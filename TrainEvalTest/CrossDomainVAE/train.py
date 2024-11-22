import matplotlib.pyplot as plt

from TrainEvalTest.GlobalConfigs import *
from TrainEvalTest.CrossDomainVAE.configs import *
from TrainEvalTest.Utils import *

from einops import rearrange

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

import os

from Dataset import DEVICE, RoadNetworkDataset
from Models import CrossDomainVAE, ClusterLoss, KLLoss


def train():
    # Dataset & DataLoader
    dataset = RoadNetworkDataset("Dataset/Tokyo_10k",
                                 batch_size=B,
                                 drop_last=True,
                                 set_name="train",
                                 enable_aug=True,
                                 img_H=16,
                                 img_W=16
                                 )

    vae = CrossDomainVAE(N_routes=dataset.N_trajs, L_route=dataset.max_L_route, N_interp=dataset.N_interp, threshold=0.5).to(DEVICE)

    loadModels("Runs/CDVAE/241121_1618_final/last.pth", vae=vae)

    cluster_loss_func = ClusterLoss()
    rec_loss_func = torch.nn.MSELoss()
    kl_loss_func = KLLoss(kl_weight=KL_WEIGHT)

    # Optimizer & Scheduler
    optimizer = AdamW(vae.parameters(), lr=LR, amsgrad=True)
    lr_scheduler = ReduceLROnPlateau(optimizer, factor=LR_REDUCE_FACTOR, patience=LR_REDUCE_PATIENCE,
                                     min_lr=LR_REDUCE_MIN, threshold=LR_REDUCE_THRESHOLD)

    # Prepare Logging
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR, exist_ok=True)
    writer = SummaryWriter(log_dir=LOG_DIR)
    mov_avg_cll = MovingAvg(MOV_AVG_LEN * len(dataset))
    mov_avg_kll = MovingAvg(MOV_AVG_LEN * len(dataset))
    mov_avg_rec = MovingAvg(MOV_AVG_LEN * len(dataset))
    global_step = 0
    best_loss = float("inf")
    plot_manager = PlotManager(4, 1, 4)

    with ProgressManager(len(dataset), EPOCHS, 5, 2, ["CLL", "KLL", "REC", "lr"]) as progress:
        for e in range(EPOCHS):
            total_loss = 0
            for i, batch in enumerate(dataset):
                batch: Dict[str, torch.Tensor]
                optimizer.zero_grad()

                # routes: (B, N_trajs, L_route, N_interp, 2)
                # routes_enc: (B, N_trajs, L_route, N_interp*2)
                # decoded: (B, N_trajs*L_route, N_interp, 2)
                # cluster_mat: (B, N_trajs*L_route, N_trajs*L_route)
                # cluster_means: (B, N_trajs*L_route, N_interp, 2)
                # coi_means: (B, ?, N_interp, 2)
                # segs: (B, N_segs, N_interp, 2)

                z_mean, z_logvar, duplicate_segs, cluster_mat, cluster_means, coi_means = vae(batch["routes"])
                kll = kl_loss_func(z_mean, z_logvar)
                rec = rec_loss_func(duplicate_segs, batch["routes"].flatten(1, 2))
                cll = cluster_loss_func(duplicate_segs.detach(), cluster_mat, batch["segs"])
                loss = kll + cll + rec

                # Backpropagation
                loss.backward()

                torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)

                optimizer.step()

                total_loss += loss
                global_step += 1
                mov_avg_cll.update(cll)
                mov_avg_kll.update(kll)
                mov_avg_rec.update(rec)

                # Progress update
                progress.update(e, i,
                                CLL=mov_avg_cll.get(),
                                KLL=mov_avg_kll.get(),
                                REC=mov_avg_rec.get(),
                                lr=optimizer.param_groups[0]['lr'])

                # Logging
                if global_step % LOG_INTERVAL == 0:
                    writer.add_scalar("CLL", mov_avg_cll.get(), global_step)
                    writer.add_scalar("KLL", mov_avg_kll.get(), global_step)
                    writer.add_scalar("REC", mov_avg_rec.get(), global_step)
                    writer.add_scalar("Learning Rate", optimizer.param_groups[0]["lr"], global_step)

            # Compute mean for each cluster
            #clusters = loss_func.getClusters(pred_segs[0], pred_cluster_mat[0])

            # Plot reconstructed segments and graphs
            plot_manager.plotSegments(batch["routes"][0], 0, 0, "Routes", color="red")
            plot_manager.plotSegments(batch["segs"][0], 0, 1, "Segs", color="blue")
            plot_manager.plotSegments(coi_means[0], 0, 2, "Pred Segs", color="green")
            plot_manager.plotSegments(duplicate_segs[0], 0, 3, "Pred Duplicate Segs")

            writer.add_figure("Reconstructed Graphs", plot_manager.getFigure(), global_step)

            # Save models
            saveModels(os.path.join(LOG_DIR, "last.pth"), vae=vae)
            if total_loss < best_loss:
                best_loss = total_loss
                saveModels(os.path.join(LOG_DIR, "best.pth"), vae=vae)

            # Step scheduler
            lr_scheduler.step(total_loss)


if __name__ == "__main__":
    train()

from TrainEvalTest.GlobalConfigs import *
from TrainEvalTest.CrossDomainVAE.configs import *
from TrainEvalTest.Utils import *

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import os

from Dataset import DEVICE, LaDeCachedDataset
from Models import CrossDomainVAE, ClusterLoss, KLLoss


def train():
    # Dataset & DataLoader
    dataset = LaDeCachedDataset(DATA_DIR, max_trajs=N_TRAJS, set_name="train")
    dataloader = DataLoader(dataset, batch_size=B, shuffle=True, collate_fn=LaDeCachedDataset.collate_fn, drop_last=True)

    # Models
    # DiT = PathsDiT(n_paths=N_TRAJS, l_path=L_PATH, d_context=2, n_layers=4, T=T).to(DEVICE)
    # ddim = DDIM(BETA_MIN, BETA_MAX, T, DEVICE, "quadratic", skip_step=1, data_dim=3)
    # loadModels("Runs/PathsDiT/241030_0853_Initial/last.pth", DiT=DiT)
    # DiT.eval()

    vae = CrossDomainVAE(N_paths=N_TRAJS, L_path=L_PATH, D_enc=4, threshold=0.5).to(DEVICE)

    # torch.set_float32_matmul_precision("high")
    # vae = torch.compile(vae)

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
    mov_avg_cll = MovingAvg(MOV_AVG_LEN * len(dataloader))
    mov_avg_kll = MovingAvg(MOV_AVG_LEN * len(dataloader))
    mov_avg_rec = MovingAvg(MOV_AVG_LEN * len(dataloader))
    global_step = 0
    best_loss = float("inf")
    plot_manager = PlotManager(5, 1, 5)

    with ProgressManager(len(dataloader), EPOCHS, 5, 2, ["CLL", "KLL", "REC", "lr"]) as progress:
        for e in range(EPOCHS):
            total_loss = 0
            for i, batch in enumerate(dataloader):
                batch: Dict[str, torch.Tensor]

                # batch |= LaDeCachedDataset.getJointsFromSegments(batch["segs"])

                # noise = torch.randn_like(batch["paths"])
                # pred_func = lambda noisy_contents, t: [DiT(*noisy_contents, batch["trajs"], t)]
                # pred_paths = ddim.diffusionBackward([noise], pred_func, mode="eps")[0]
                optimizer.zero_grad()

                batch["duplicate_segs"] = torch.cat([batch["paths"][..., 1:, :], batch["paths"][..., :-1, :]], dim=-1)  # (B, N, L-1, 4)
                batch["duplicate_segs"] = batch["duplicate_segs"].view(B, -1, 4).contiguous()  # (B, N_segs, 4)
                filter_mask = ~torch.any(batch["duplicate_segs"] == 0, dim=2, keepdim=True)
                batch["duplicate_segs"] = batch["duplicate_segs"] * filter_mask

                z_mean, z_logvar, duplicate_segs, cluster_mat, cluster_means, coi_means = vae(batch["paths"])
                kll = kl_loss_func(z_mean, z_logvar)
                rec = rec_loss_func(duplicate_segs, batch["duplicate_segs"])
                cll = cluster_loss_func(duplicate_segs.detach(), cluster_mat, batch["segs"][..., :-1])
                loss = kll + cll + rec

                # Backpropagation
                loss.backward()
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
            plot_manager.plotTrajs(batch["paths"][0], 0, 0, "Routes")
            plot_manager.plotSegments(batch["segs"][0], 0, 1, "Segs")
            plot_manager.plotSegments(coi_means[0], 0, 2, "Pred Segs")
            plot_manager.plotSegments(batch["duplicate_segs"][0], 0, 3, "Duplicate Segs")
            plot_manager.plotSegments(duplicate_segs[0], 0, 4, "Pred Duplicate Segs")

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

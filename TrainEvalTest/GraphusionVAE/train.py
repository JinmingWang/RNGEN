from TrainEvalTest.GlobalConfigs import *
from TrainEvalTest.GraphusionVAE.configs import *
from TrainEvalTest.Utils import *

from einops import rearrange

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

import os

from Dataset import DEVICE, RoadNetworkDataset
from Models import GraphusionVAE, KLLoss

def nodesAdjMatToSegs(f_nodes, adj_mat, f_edges, threshold=0.5):
    B = f_nodes.shape[0]
    batch_segs = []
    for b in range(B):
        segs = []
        for r in range(adj_mat.shape[1]):
            for c in range(r + 1, adj_mat.shape[2]):
                if adj_mat[b, r, c] >= threshold:
                    segs.append(f_edges[b, r, c].view(8, 2))
        if len(segs) == 0:
            batch_segs.append(torch.zeros(1, 8, 2, dtype=torch.float32, device=DEVICE))
        else:
            batch_segs.append(torch.stack(segs, dim=0))     # (N_segs, 8, 2)
    return batch_segs


def train():
    # Dataset & DataLoader
    dataset = RoadNetworkDataset("Dataset/Tokyo_10k_sparse",
                                 batch_size=B,
                                 drop_last=True,
                                 set_name="train",
                                 enable_aug=True,
                                 img_H=16,
                                 img_W=16,
                                 need_nodes=True
                                 )

    vae = GraphusionVAE(d_node=2, d_edge=16, d_latent=128, d_hidden=256, n_layers=8, n_heads=8).to(DEVICE)

    loadModels("Runs/GraphusionVAE/241205_2212_initial/last.pth", vae=vae)

    rec_loss_func = torch.nn.MSELoss()
    mat_loss_func = torch.nn.BCELoss()
    kl_loss_func = KLLoss(kl_weight=KL_WEIGHT)

    # Optimizer & Scheduler
    optimizer = AdamW(vae.parameters(), lr=LR, amsgrad=True)
    lr_scheduler = ReduceLROnPlateau(optimizer, factor=LR_REDUCE_FACTOR, patience=LR_REDUCE_PATIENCE,
                                     min_lr=LR_REDUCE_MIN, threshold=LR_REDUCE_THRESHOLD)

    # Prepare Logging
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR, exist_ok=True)
    writer = SummaryWriter(log_dir=LOG_DIR)
    mov_avg_kll = MovingAvg(MOV_AVG_LEN * len(dataset))
    mov_avg_node = MovingAvg(MOV_AVG_LEN * len(dataset))
    mov_avg_edge = MovingAvg(MOV_AVG_LEN * len(dataset))
    mov_avg_adj = MovingAvg(MOV_AVG_LEN * len(dataset))
    mov_avg_deg = MovingAvg(MOV_AVG_LEN * len(dataset))
    global_step = 0
    best_loss = float("inf")
    plot_manager = PlotManager(4, 1, 4)

    with ProgressManager(len(dataset), EPOCHS, 5, 2, ["KLL", "NODE", "EDGE", "ADJ", "DEG", "lr"]) as progress:
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

                z_mean, z_logvar, f_nodes, f_edges, pred_adj_mat, pred_degrees = vae(batch["nodes"], batch["edges"], batch["adj_mat"])

                kll = kl_loss_func(z_mean, z_logvar)
                loss_nodes = rec_loss_func(f_nodes, batch["nodes"])
                loss_edges = rec_loss_func(f_edges, batch["edges"])
                loss_adj_mat = mat_loss_func(pred_adj_mat, batch["adj_mat"])
                loss_degree = rec_loss_func(pred_degrees, batch["degrees"])
                loss = kll + loss_nodes + loss_adj_mat + loss_edges + loss_degree

                # Backpropagation
                loss.backward()

                torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)

                optimizer.step()

                total_loss += loss
                global_step += 1
                mov_avg_kll.update(kll.item())
                mov_avg_node.update(loss_nodes.item())
                mov_avg_edge.update(loss_edges.item())
                mov_avg_adj.update(loss_adj_mat.item())
                mov_avg_deg.update(loss_degree.item())


                # Progress update
                progress.update(e, i,
                                KLL=mov_avg_kll.get(),
                                NODE=mov_avg_node.get(),
                                EDGE=mov_avg_edge.get(),
                                ADJ=mov_avg_adj.get(),
                                DEG=mov_avg_deg.get(),
                                lr=optimizer.param_groups[0]['lr'])

                # Logging
                if global_step % LOG_INTERVAL == 0:
                    writer.add_scalar("KLL", mov_avg_kll.get(), global_step)
                    writer.add_scalar("NODE", mov_avg_node.get(), global_step)
                    writer.add_scalar("EDGE", mov_avg_edge.get(), global_step)
                    writer.add_scalar("ADJ", mov_avg_adj.get(), global_step)
                    writer.add_scalar("DEG", mov_avg_deg.get(), global_step)
                    writer.add_scalar("Learning Rate", optimizer.param_groups[0]["lr"], global_step)


            pred_segs = nodesAdjMatToSegs(f_nodes[0:1], pred_adj_mat[0:1], f_edges[0:1], threahold=0.5)
            target_segs = nodesAdjMatToSegs(batch["nodes"][0:1], batch["adj_mat"][0:1], batch["edges"][0:1], threahold=0.5)

            # Plot reconstructed segments and graphs
            plot_manager.plotNodesWithAdjMat(batch["nodes"][0], batch["adj_mat"][0], 0, 0, "Nodes")
            plot_manager.plotSegments(target_segs[0], 0, 1, "Segs", color="blue")
            plot_manager.plotNodesWithAdjMat(f_nodes[0], pred_adj_mat[0] > 0.5, 0, 2, "Pred Graph")
            plot_manager.plotSegments(pred_segs[0], 0, 3, "Pred Segments", color="red")

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

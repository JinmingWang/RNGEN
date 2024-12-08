from TrainEvalTest.Utils import *
from datetime import datetime

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


def train(
        title: str = "initial",
        dataset_path: str = "Dataset/Tokyo",
        kl_weight: float = 1e-6,
        lr: float = 1e-4,
        lr_reduce_factor: float = 0.5,
        lr_reduce_patience: int = 30,
        lr_reduce_min: float = 1e-7,
        lr_reduce_threshold: float = 1e-5,
        epochs: int = 1000,
        B: int = 32,
        mov_avg_len: int = 6,
        log_interval: int = 10,
        load_weights: str = "Runs/GraphusionVAE/241205_2212_initial/last.pth"
):
    log_dir = f"./Runs/GraphusionVAE/{datetime.now().strftime('%Y%m%d_%H%M')[2:]}_{title}/"
    # Dataset & DataLoader
    dataset = RoadNetworkDataset(folder_path=dataset_path,
                                 batch_size=B,
                                 drop_last=True,
                                 set_name="train",
                                 enable_aug=True,
                                 img_H=16,
                                 img_W=16,
                                 need_nodes=True
                                 )

    vae = GraphusionVAE(d_node=2, d_edge=16, d_latent=128, d_hidden=256, n_layers=8, n_heads=8).to(DEVICE)

    if load_weights is not None:
        loadModels(load_weights, vae=vae)

    rec_loss_func = torch.nn.MSELoss()
    mat_loss_func = torch.nn.BCELoss()
    kl_loss_func = KLLoss(kl_weight=kl_weight)

    # Optimizer & Scheduler
    optimizer = AdamW(vae.parameters(), lr=lr, amsgrad=True)
    lr_scheduler = ReduceLROnPlateau(optimizer, factor=lr_reduce_factor, patience=lr_reduce_patience,
                                     min_lr=lr_reduce_min, threshold=lr_reduce_threshold)

    # Prepare Logging
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    mov_avg_kll = MovingAvg(mov_avg_len * len(dataset))
    mov_avg_node = MovingAvg(mov_avg_len * len(dataset))
    mov_avg_edge = MovingAvg(mov_avg_len * len(dataset))
    mov_avg_adj = MovingAvg(mov_avg_len * len(dataset))
    mov_avg_deg = MovingAvg(mov_avg_len * len(dataset))
    global_step = 0
    best_loss = float("inf")
    plot_manager = PlotManager(4, 1, 4)

    with ProgressManager(len(dataset), epochs, 5, 2, ["KLL", "NODE", "EDGE", "ADJ", "DEG", "lr"]) as progress:
        for e in range(epochs):
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
                if global_step % log_interval == 0:
                    writer.add_scalar("KLL", mov_avg_kll.get(), global_step)
                    writer.add_scalar("NODE", mov_avg_node.get(), global_step)
                    writer.add_scalar("EDGE", mov_avg_edge.get(), global_step)
                    writer.add_scalar("ADJ", mov_avg_adj.get(), global_step)
                    writer.add_scalar("DEG", mov_avg_deg.get(), global_step)
                    writer.add_scalar("Learning Rate", optimizer.param_groups[0]["lr"], global_step)


            pred_segs = nodesAdjMatToSegs(f_nodes[0:1], pred_adj_mat[0:1], f_edges[0:1], threshold=0.5)
            target_segs = nodesAdjMatToSegs(batch["nodes"][0:1], batch["adj_mat"][0:1], batch["edges"][0:1], threshold=0.5)

            # Plot reconstructed segments and graphs
            plot_manager.plotNodesWithAdjMat(batch["nodes"][0], batch["adj_mat"][0], 0, 0, "Nodes")
            plot_manager.plotSegments(target_segs[0], 0, 1, "Segs", color="blue")
            plot_manager.plotNodesWithAdjMat(f_nodes[0], pred_adj_mat[0] > 0.5, 0, 2, "Pred Graph")
            plot_manager.plotSegments(pred_segs[0], 0, 3, "Pred Segments", color="red")

            writer.add_figure("Reconstructed Graphs", plot_manager.getFigure(), global_step)

            # Save models
            saveModels(os.path.join(log_dir, "last.pth"), vae=vae)
            if total_loss < best_loss:
                best_loss = total_loss
                saveModels(os.path.join(log_dir, "best.pth"), vae=vae)

            # Step scheduler
            lr_scheduler.step(total_loss)

            if optimizer.param_groups[0]["lr"] <= lr_reduce_min:
                # Stop training if learning rate is too low
                break

    return os.path.join(log_dir, "last.pth")

if __name__ == "__main__":
    train()

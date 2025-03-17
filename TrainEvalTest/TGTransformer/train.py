from TrainEvalTest.Utils import *
from datetime import datetime

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

import os

from Dataset import DEVICE, RoadNetworkDataset
from Models import TGTransformer, ClusterLoss

def train(
        title: str = "295M",
        dataset_path: str = "Dataset/Tokyo",
        lr: float = 2e-4,
        lr_reduce_factor: float = 0.5,
        lr_reduce_patience: int = 30,
        lr_reduce_min: float = 1e-7,
        lr_reduce_threshold: float = 1e-5,
        epochs: int = 1000,
        B: int = 32,
        mov_avg_len: int = 5,
        log_interval: int = 10,
        load_weights: str = None,
        *args, **kwargs
):
    log_dir = f"./Runs/TGTransformer/{datetime.now().strftime('%Y%m%d_%H%M')[2:]}_{title}/"
    # Dataset & DataLoader
    dataset = RoadNetworkDataset(folder_path=dataset_path,
                                 batch_size=B,
                                 drop_last=True,
                                 set_name="train",
                                 enable_aug=False,
                                 img_H=16,
                                 img_W=16
                                 )

    transformer = TGTransformer(
        N_routes=dataset.N_trajs,
        L_route=dataset.max_L_route,
        N_interp=dataset.N_interp,
        threshold=0.5
    )

    if load_weights is not None:
        loadModels(load_weights, transformer=transformer)

    # torch.set_float32_matmul_precision("high")
    # torch.compile(DiT)

    transformer = transformer.to(DEVICE)

    cluster_loss_func = ClusterLoss()
    rec_loss_func = torch.nn.MSELoss()

    # Optimizer & Scheduler
    optimizer = AdamW(transformer.parameters(), lr=lr, amsgrad=True)
    lr_scheduler = ReduceLROnPlateau(optimizer, factor=lr_reduce_factor, patience=lr_reduce_patience,
                                     min_lr=lr_reduce_min, threshold=lr_reduce_threshold)

    # Prepare Logging
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)
    mov_avg_cll = MovingAvg(mov_avg_len * len(dataset))
    mov_avg_rec = MovingAvg(mov_avg_len * len(dataset))
    global_step = 0
    best_loss = float("inf")
    plot_manager = PlotManager(4, 1, 4)

    with ProgressManager(len(dataset), epochs, 5, 2, ["CLL", "REC", "lr"]) as progress:
        for e in range(epochs):
            total_loss = 0
            for i, batch in enumerate(dataset):
                batch: Dict[str, Tensor]

                duplicate_segs, cluster_mat, cluster_means, coi_means = transformer(batch["trajs"])

                rec = rec_loss_func(duplicate_segs, batch["routes"].flatten(1, 2)) + \
                      rec_loss_func(cluster_means, batch["routes"].flatten(1, 2))

                cll = cluster_loss_func(cluster_means.detach(), cluster_mat, batch["segs"])
                loss = rec + cll

                # Backpropagation
                loss.backward()

                torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

                total_loss += loss.item()
                global_step += 1
                mov_avg_cll.update(cll.item())
                mov_avg_rec.update(rec.item())

                progress.update(e, i,
                                CLL=mov_avg_cll.get(),
                                REC=mov_avg_rec.get(),
                                lr=optimizer.param_groups[0]['lr'])

                if global_step % log_interval == 0:
                    writer.add_scalar("CLL", mov_avg_cll.get(), global_step)
                    writer.add_scalar("REC", mov_avg_rec.get(), global_step)
                    writer.add_scalar("Learning Rate", optimizer.param_groups[0]["lr"], global_step)

            plot_manager.plotSegments(batch["routes"][0], 0, 0, "Routes", color="red")
            plot_manager.plotSegments(batch["segs"][0], 0, 1, "Segs", color="blue")
            plot_manager.plotSegments(coi_means[0], 0, 2, "Pred Segs", color="green")
            plot_manager.plotSegments(cluster_means[0], 0, 3, "Pred Duplicate Segs")

            writer.add_figure("Reconstructed Graphs", plot_manager.getFigure(), global_step)

            saveModels(log_dir + "last.pth", transformer=transformer)
            if total_loss < best_loss:
                best_loss = total_loss
                saveModels(log_dir + "best.pth", transformer=transformer)

            lr_scheduler.step(total_loss)

            if optimizer.param_groups[0]["lr"] <= lr_reduce_min:
                # Stop training if learning rate is too low
                break

    return os.path.join(log_dir, "last.pth")

if __name__ == "__main__":
    train()

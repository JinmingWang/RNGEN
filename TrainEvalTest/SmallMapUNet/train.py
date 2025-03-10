from datetime import datetime
from TrainEvalTest.Utils import *

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import os

from Dataset import DEVICE, RoadNetworkDataset
from Models import UNet2D


def train(
        title: str = "initial",
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
        load_weights: str = "Runs/SmallMapUNet/241124_0231_sparse/last.pth"
):
    log_dir = f"./Runs/SmallMap/{datetime.now().strftime('%Y%m%d_%H%M')[2:]}_{title}/"
    # Dataset & DataLoader
    dataset = RoadNetworkDataset(folder_path=dataset_path,
                                 batch_size=B,
                                 drop_last=True,
                                 set_name="train",
                                 permute_seq=False,
                                 enable_aug=False,
                                 img_H=256,
                                 img_W=256,
                                 need_heatmap=True
                                 )

    stage_1 = UNet2D(n_repeats=2, expansion=2).to(DEVICE)
    stage_2 = UNet2D(n_repeats=2, expansion=2).to(DEVICE)

    if load_weights is not None:
        loadModels(load_weights, stage_1=stage_1, stage_2=stage_2)

    torch.set_float32_matmul_precision("high")
    torch.compile(stage_1)
    torch.compile(stage_2)

    loss_func = torch.nn.BCELoss()

    # Optimizer & Scheduler
    optimizer = AdamW(list(stage_1.parameters()) + list(stage_2.parameters()), lr=lr)
    lr_scheduler = ReduceLROnPlateau(optimizer, factor=lr_reduce_factor, patience=lr_reduce_patience,
                                     min_lr=lr_reduce_min, threshold=lr_reduce_threshold)

    # Prepare Logging
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    mov_avg_l1 = MovingAvg(mov_avg_len * len(dataset))
    mov_avg_l2 = MovingAvg(mov_avg_len * len(dataset))
    global_step = 0
    best_loss = float("inf")
    plot_manager = PlotManager(4, 1, 4)

    with ProgressManager(len(dataset), epochs, 5, 2, ["L1", "L2", "lr"]) as progress:
        for e in range(epochs):
            total_loss = 0
            for i, batch in enumerate(dataset):
                batch: Dict[str, torch.Tensor]

                optimizer.zero_grad()

                pred_1 = stage_1(batch["heatmap"])
                pred_2 = stage_2(pred_1)

                loss1 = loss_func(pred_1, batch["target_heatmaps"])
                loss2 = loss_func(pred_2, batch["target_heatmaps"])
                loss = loss1 + loss2

                # Backpropagation
                loss.backward()
                optimizer.step()

                total_loss += loss
                global_step += 1
                mov_avg_l1.update(loss1.item())
                mov_avg_l2.update(loss2.item())

                # Progress update
                progress.update(e, i,
                                L1=mov_avg_l1.get(),
                                L2=mov_avg_l2.get(),
                                lr=optimizer.param_groups[0]['lr'])

                # Logging
                if global_step % log_interval == 0:
                    writer.add_scalar("L1", mov_avg_l1.get(), global_step)
                    writer.add_scalar("L2", mov_avg_l2.get(), global_step)
                    writer.add_scalar("Learning Rate", optimizer.param_groups[0]["lr"], global_step)

            # Plot reconstructed segments and graphs
            plot_manager.plotHeatmap(batch["heatmap"][0], 0, 0, "Input")
            plot_manager.plotHeatmap(batch["target_heatmaps"][0], 0, 1, "Target")
            plot_manager.plotHeatmap(pred_1[0], 0, 2, "Pred 1")
            plot_manager.plotHeatmap(pred_2[0], 0, 3, "Pred 2")

            writer.add_figure("Figures", plot_manager.getFigure(), global_step)

            # Save models
            saveModels(os.path.join(log_dir, "last.pth"), stage_1=stage_1, stage_2=stage_2)
            if total_loss < best_loss:
                best_loss = total_loss
                saveModels(os.path.join(log_dir, "best.pth"), stage_1=stage_1, stage_2=stage_2)

            # Step scheduler
            lr_scheduler.step(total_loss)

            if optimizer.param_groups[0]["lr"] <= lr_reduce_min:
                # Stop training if learning rate is too low
                break

    return os.path.join(log_dir, "last.pth")

if __name__ == "__main__":
    train()

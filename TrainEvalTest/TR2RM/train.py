from datetime import datetime
from TrainEvalTest.Utils import *

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import os

from Dataset import DEVICE, RoadNetworkDataset
from Models import AD_Linked_Net


def train(
        title: str = "initial",
        dataset_path: str = "Dataset/Tokyo_10k_sparse",
        lr: float = 2e-4,
        lr_reduce_factor: float = 0.5,
        lr_reduce_patience: int = 30,
        lr_reduce_min: float = 1e-7,
        lr_reduce_threshold: float = 1e-5,
        epochs: int = 1000,
        B: int = 32,
        mov_avg_len: int = 5,
        log_interval: int = 10,
        load_weights: str = "Runs/TR2RM/241124_0231_sparse/last.pth"
):
    log_dir = f"./Runs/TR2RM/{datetime.now().strftime('%Y%m%d_%H%M')[2:]}_{title}/"
    # Dataset & DataLoader
    dataset = RoadNetworkDataset(folder_path=dataset_path,
                                 batch_size=B,
                                 drop_last=True,
                                 set_name="train",
                                 permute_seq=False,
                                 enable_aug=False,
                                 img_H=256,
                                 img_W=256
                                 )

    model = AD_Linked_Net(d_in=4, H=256, W=256).to(DEVICE)

    if load_weights is not None:
        loadModels("Runs/TR2RM/241124_0231_sparse/last.pth", ADLinkedNet=model)

    torch.set_float32_matmul_precision("high")
    torch.compile(model)

    loss_func = torch.nn.BCELoss()

    # Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=lr)
    lr_scheduler = ReduceLROnPlateau(optimizer, factor=lr_reduce_factor, patience=lr_reduce_patience,
                                     min_lr=lr_reduce_min, threshold=lr_reduce_threshold)

    # Prepare Logging
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    mov_avg_loss = MovingAvg(mov_avg_len * len(dataset))
    global_step = 0
    best_loss = float("inf")
    plot_manager = PlotManager(4, 1, 4)

    with ProgressManager(len(dataset), epochs, 5, 2, ["Loss", "lr"]) as progress:
        for e in range(epochs):
            total_loss = 0
            for i, batch in enumerate(dataset):
                batch: Dict[str, torch.Tensor]

                optimizer.zero_grad()

                pred_heatmap = model(torch.cat([batch["heatmap"], batch["image"]], dim=1))

                loss = loss_func(pred_heatmap, batch["target_heatmaps"])

                # Backpropagation
                loss.backward()
                optimizer.step()

                total_loss += loss
                global_step += 1
                mov_avg_loss.update(loss)

                # Progress update
                progress.update(e, i,
                                Loss=mov_avg_loss.get(),
                                lr=optimizer.param_groups[0]['lr'])

                # Logging
                if global_step % log_interval == 0:
                    writer.add_scalar("Loss", mov_avg_loss.get(), global_step)
                    writer.add_scalar("Learning Rate", optimizer.param_groups[0]["lr"], global_step)

            # Plot reconstructed segments and graphs
            plot_manager.plotHeatmap(batch["heatmap"][0], 0, 0, "Input")
            plot_manager.plotRGB(batch["image"][0], 0, 1, "Image")
            plot_manager.plotHeatmap(batch["target_heatmaps"][0], 0, 2, "Target")
            plot_manager.plotHeatmap(pred_heatmap[0], 0, 3, "Prediction")

            writer.add_figure("Figures", plot_manager.getFigure(), global_step)

            # Save models
            saveModels(os.path.join(log_dir, "last.pth"), ADLinkedNet=model)
            if total_loss < best_loss:
                best_loss = total_loss
                saveModels(os.path.join(log_dir, "best.pth"), ADLinkedNet=model)

            # Step scheduler
            lr_scheduler.step(total_loss)

            if optimizer.param_groups[0]["lr"] <= lr_reduce_min:
                # Stop training if learning rate is too low
                break

    return os.path.join(log_dir, "last.pth")


if __name__ == "__main__":
    train()

from TrainEvalTest.GlobalConfigs import *
from TrainEvalTest.TR2RM.configs import *
from TrainEvalTest.Utils import *

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import os

from Dataset import DEVICE, RoadNetworkDataset
from Models import AD_Linked_Net


def train():
    # Dataset & DataLoader
    dataset = RoadNetworkDataset("Dataset/Tokyo_10k",
                                 batch_size=B,
                                 drop_last=True,
                                 set_name="train",
                                 permute_seq=False,
                                 enable_aug=False,
                                 img_H=256,
                                 img_W=256
                                 )

    model = AD_Linked_Net(d_in=4, H=256, W=256).to(DEVICE)

    torch.set_float32_matmul_precision("high")
    torch.compile(model)

    loss_func = torch.nn.BCELoss()

    # Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=LR)
    lr_scheduler = ReduceLROnPlateau(optimizer, factor=LR_REDUCE_FACTOR, patience=LR_REDUCE_PATIENCE,
                                     min_lr=LR_REDUCE_MIN, threshold=LR_REDUCE_THRESHOLD)

    # Prepare Logging
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR, exist_ok=True)
    writer = SummaryWriter(log_dir=LOG_DIR)
    mov_avg_loss = MovingAvg(MOV_AVG_LEN * len(dataset))
    global_step = 0
    best_loss = float("inf")
    plot_manager = PlotManager(4, 1, 4)

    with ProgressManager(len(dataset), EPOCHS, 5, 2, ["Loss", "lr"]) as progress:
        for e in range(EPOCHS):
            total_loss = 0
            for i, batch in enumerate(dataset):
                batch: Dict[str, torch.Tensor]

                H, W = batch["heatmap"].shape[-2:]
                batch |= RoadNetworkDataset.getTargetHeatmaps(batch, H, W)

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
                if global_step % LOG_INTERVAL == 0:
                    writer.add_scalar("Loss", mov_avg_loss.get(), global_step)
                    writer.add_scalar("Learning Rate", optimizer.param_groups[0]["lr"], global_step)

            # Plot reconstructed segments and graphs
            plot_manager.plotHeatmap(batch["heatmap"][0], 0, 0, "Input")
            plot_manager.plotRGB(batch["image"][0], 0, 1, "Image")
            plot_manager.plotHeatmap(batch["target_heatmaps"][0], 0, 2, "Target")
            plot_manager.plotHeatmap(pred_heatmap[0], 0, 3, "Prediction")

            writer.add_figure("Figures", plot_manager.getFigure(), global_step)

            # Save models
            saveModels(os.path.join(LOG_DIR, "last.pth"), ADLinkedNet=model)
            if total_loss < best_loss:
                best_loss = total_loss
                saveModels(os.path.join(LOG_DIR, "best.pth"), ADLinkedNet=model)

            # Step scheduler
            lr_scheduler.step(total_loss)


if __name__ == "__main__":
    train()

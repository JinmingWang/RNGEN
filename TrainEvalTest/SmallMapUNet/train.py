from TrainEvalTest.GlobalConfigs import *
from TrainEvalTest.SmallMapUNet.configs import *
from TrainEvalTest.Utils import *

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import os

from Dataset import DEVICE, LaDeCachedDataset
from Models import UNet2D


def train():
    # Dataset & DataLoader
    dataset = LaDeCachedDataset(DATA_DIR, max_trajs=N_TRAJS, set_name="train")
    dataloader = DataLoader(dataset, batch_size=B, shuffle=True, collate_fn=LaDeCachedDataset.collate_fn, drop_last=True)

    stage_1 = UNet2D(n_repeats=2, expansion=2).to(DEVICE)
    stage_2 = UNet2D(n_repeats=2, expansion=2).to(DEVICE)


    loss_func = torch.nn.MSELoss()

    # Optimizer & Scheduler
    optimizer = AdamW(list(stage_1.parameters()) + list(stage_2.parameters()), lr=LR)
    lr_scheduler = ReduceLROnPlateau(optimizer, factor=LR_REDUCE_FACTOR, patience=LR_REDUCE_PATIENCE,
                                     min_lr=LR_REDUCE_MIN, threshold=LR_REDUCE_THRESHOLD)

    # Prepare Logging
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR, exist_ok=True)
    writer = SummaryWriter(log_dir=LOG_DIR)
    mov_avg_l1 = MovingAvg(MOV_AVG_LEN * len(dataloader))
    mov_avg_l2 = MovingAvg(MOV_AVG_LEN * len(dataloader))
    global_step = 0
    best_loss = float("inf")
    plot_manager = PlotManager(4, 1, 4)

    with ProgressManager(len(dataloader), EPOCHS, 5, 2, ["L1", "L2", "lr"]) as progress:
        for e in range(EPOCHS):
            total_loss = 0
            for i, batch in enumerate(dataloader):
                batch: Dict[str, torch.Tensor]

                H, W = batch["heatmaps"].shape[-2:]
                batch |= LaDeCachedDataset.getTargetHeatmaps(batch["segs"], H, W)

                # noise = torch.randn_like(batch["paths"])
                # pred_func = lambda noisy_contents, t: [DiT(*noisy_contents, batch["trajs"], t)]
                # pred_paths = ddim.diffusionBackward([noise], pred_func, mode="eps")[0]
                optimizer.zero_grad()

                pred_1 = stage_1(batch["heatmaps"])
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
                if global_step % LOG_INTERVAL == 0:
                    writer.add_scalar("L1", mov_avg_l1.get(), global_step)
                    writer.add_scalar("L2", mov_avg_l2.get(), global_step)
                    writer.add_scalar("Learning Rate", optimizer.param_groups[0]["lr"], global_step)

            # Plot reconstructed segments and graphs
            plot_manager.plotHeatmap(batch["heatmaps"][0], 0, 0, "Input")
            plot_manager.plotHeatmap(batch["target_heatmaps"][0], 0, 1, "Target")
            plot_manager.plotHeatmap(pred_1[0], 0, 2, "Pred 1")
            plot_manager.plotHeatmap(pred_2[0], 0, 3, "Pred 2")

            writer.add_figure("Figures", plot_manager.getFigure(), global_step)

            # Save models
            saveModels(os.path.join(LOG_DIR, "last.pth"), stage_1=stage_1, stage_2=stage_2)
            if total_loss < best_loss:
                best_loss = total_loss
                saveModels(os.path.join(LOG_DIR, "best.pth"), stage_1=stage_1, stage_2=stage_2)

            # Step scheduler
            lr_scheduler.step(total_loss)


if __name__ == "__main__":
    train()

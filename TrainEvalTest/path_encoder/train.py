from TrainEvalTest.GlobalConfigs import *
from TrainEvalTest.path_encoder.configs import *
from TrainEvalTest.Utils import *

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import os

from Dataset import DEVICE, LaDeCachedDataset
from Models import PathEncoder


def train():
    # Dataset & DataLoader
    dataset = LaDeCachedDataset(DATA_DIR, max_trajs=N_TRAJS, set_name="train")
    dataloader = DataLoader(dataset, batch_size=B, shuffle=True, collate_fn=LaDeCachedDataset.collate_fn,
                            drop_last=True)

    # Models
    encoder = PathEncoder(N_TRAJS, L_TRAJ, L_PATH).to(DEVICE)
    torch.set_float32_matmul_precision('high')
    encoder = torch.compile(encoder)

    loss_func = torch.nn.MSELoss()

    # Optimizer & Scheduler
    optimizer = AdamW(encoder.parameters(), lr=LR_ENCODER)
    lr_scheduler = ReduceLROnPlateau(optimizer, factor=LR_REDUCE_FACTOR, patience=LR_REDUCE_PATIENCE,
                                     min_lr=LR_REDUCE_MIN, threshold=LR_REDUCE_THRESHOLD)

    # Prepare Logging
    os.makedirs(LOG_DIR)
    writer = SummaryWriter(log_dir=LOG_DIR)
    # plot_manager = PlotManager(5, 1, 1)
    mov_avg_loss = MovingAvg(MOV_AVG_LEN)
    global_step = 0
    best_loss = float("inf")
    plot_manager = PlotManager(5, 2, 2)

    with ProgressManager(len(dataloader), EPOCHS, 5, 2, ["Loss", "lr"]) as progress:
        for e in range(EPOCHS):
            total_loss = 0
            for i, batch in enumerate(dataloader):
                batch: Dict[str, Tensor]

                optimizer.zero_grad()
                pred_paths = encoder(batch["trajs"])

                loss = loss_func(pred_paths, batch["paths"])

                loss.backward()
                optimizer.step()

                total_loss += loss
                global_step += 1
                mov_avg_loss.update(loss)

                progress.update(e, i, Loss=mov_avg_loss.get(), lr=optimizer.param_groups[0]['lr'])

                if global_step % 5 == 0:
                    writer.add_scalar("loss", mov_avg_loss.get(), global_step)
                    writer.add_scalar("lr", optimizer.param_groups[0]["lr"], global_step)


            plot_manager.plotSegments(batch["graphs"][0], 0, 0, "Graph")
            plot_manager.plotTrajs(batch["trajs"][0], 0, 1, "Trajs")
            plot_manager.plotTrajs(batch["paths"][0], 1, 0, "Paths")
            plot_manager.plotTrajs(pred_paths[0], 1, 1, "Reconstructed Paths")
            writer.add_figure("Figure", plot_manager.getFigure(), global_step)

            saveModels(LOG_DIR + "last.pth", encoder)
            if total_loss < best_loss:
                best_loss = total_loss
                saveModels(LOG_DIR + "best.pth", encoder)

            lr_scheduler.step(total_loss)


if __name__ == "__main__":
    train()

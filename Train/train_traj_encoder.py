import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import numpy as np
from tqdm import tqdm
from datetime import datetime
import os

from Dataset import DEVICE, LaDeCachedDataset
from Models import TrajAutoEncoder
from Train.Utils import renderPlotHeatmap, renderPlotTraj


def train():
    epochs = 100

    torch.autograd.set_detect_anomaly(True)

    dataset = LaDeCachedDataset("./Dataset/Shanghai_5k")

    dataloader = DataLoader(dataset, batch_size=20, shuffle=True, collate_fn=LaDeCachedDataset.collate_fn)

    TAE = TrajAutoEncoder().to(DEVICE)

    # encoder, decoder = torch.load("Runs/2024-09-30_07-59-11/last.pth")

    optimizer = AdamW(TAE.parameters(), lr=2e-4)

    lr_scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6)

    now_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = f"./Runs/{now_str}/"
    os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    iterations = 0
    best_loss = float("inf")
    recent_loss = [0] * 10

    for e in range(epochs):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Training Epoch {e + 1}/{epochs}")
        for batch_trajs, batch_graph, batch_heatmap in pbar:
            loss, output = TAE.trainStep(optimizer, trajs=batch_trajs, heatmap=batch_heatmap)

            total_loss += loss
            iterations += 1
            recent_loss.pop(0)
            recent_loss.append(loss)

            pbar.set_postfix_str(f"loss={np.mean(recent_loss):.7f}, lr={optimizer.param_groups[0]['lr']:.5e}")

            if iterations % 5 == 0:
                writer.add_scalar("loss", np.mean(recent_loss), iterations)
                writer.add_scalar("lr", optimizer.param_groups[0]["lr"], iterations)

            if iterations % 50 == 0:
                # writer.add_figure("Plots",
                #                   renderPlot(batch_graph[0], batch_trajs[0], batch_heatmap[0], output[0]),
                #                   iterations)
                writer.add_figure("Plots",
                                  renderPlotTraj(batch_graph[0], batch_trajs[0], output[0]),
                                  iterations)

        torch.save(TAE, log_dir + "last.pth")
        if total_loss < best_loss:
            best_loss = total_loss
            torch.save(TAE, log_dir + "best.pth")

        lr_scheduler.step(total_loss)


if __name__ == "__main__":
    train()

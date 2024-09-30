import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from tqdm import tqdm
from datetime import datetime
import os

from Dataset import DEVICE, LaDeCachedDataset
from Models import TrajEncoder, TrajDecoder, BiasLoss
from Train.Utils import renderPlot


def train():
    epochs = 500

    torch.autograd.set_detect_anomaly(True)

    dataset = LaDeCachedDataset("./Dataset/Shanghai_500_Cache.pth")


    dataloader = DataLoader(dataset, batch_size=20, shuffle=True, collate_fn=LaDeCachedDataset.collate_fn)

    loss_func = nn.MSELoss()

    encoder = TrajEncoder().to(DEVICE)
    decoder = TrajDecoder().to(DEVICE)

    encoder, decoder = torch.load("Runs/2024-09-30_07-59-11/last.pth")

    optimizer = AdamW([{"params": encoder.parameters()}, {"params": decoder.parameters()}], lr=2e-4)

    lr_scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10, min_lr=1e-6)

    now_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = f"./Runs/{now_str}/"
    os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    iterations = 0
    best_loss = float("inf")

    for e in range(epochs):
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Training Epoch {e + 1}/{epochs}")
        for batch_trajs, batch_graph, batch_heatmap in pbar:
            optimizer.zero_grad()

            batch_traj_enc = encoder(batch_trajs)
            pred_heatmap = decoder(batch_traj_enc)

            loss = loss_func(pred_heatmap, batch_heatmap)
            loss.backward()

            optimizer.step()

            total_loss += loss.item()
            iterations += 1

            pbar.set_postfix_str(f"loss={loss.item():.7f}, lr={optimizer.param_groups[0]['lr']:.5e}")

            if iterations % 5 == 0:
                writer.add_scalar("loss", loss, iterations)
                writer.add_scalar("lr", optimizer.param_groups[0]["lr"], iterations)

            if iterations % 50 == 0:
                writer.add_figure("Plots",
                                  renderPlot(batch_graph[0], batch_trajs[0], batch_heatmap[0], pred_heatmap[0]),
                                  iterations)

        torch.save([encoder, decoder], log_dir + "last.pth")
        if total_loss < best_loss:
            best_loss = total_loss
            torch.save([encoder, decoder], log_dir + "best.pth")

        lr_scheduler.step(total_loss)


if __name__ == "__main__":
    train()

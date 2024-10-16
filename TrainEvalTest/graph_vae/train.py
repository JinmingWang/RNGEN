from TrainEvalTest.GlobalConfigs import *
from TrainEvalTest.graph_vae.configs import *
from TrainEvalTest.Utils import *

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import os

from Dataset import DEVICE, LaDeCachedDataset
from Models import GraphEncoder, GraphDecoder, VAELoss


def train():
    # Dataset & DataLoader
    dataset = LaDeCachedDataset(DATA_DIR, max_trajs=N_TRAJS, set_name="train")
    dataloader = DataLoader(dataset, batch_size=B, shuffle=True, collate_fn=LaDeCachedDataset.collate_fn, drop_last=True)

    # Models
    encoder = GraphEncoder(d_in=D_IN, d_latent=D_LATENT, d_head=D_HEAD, d_expand=D_EXPAND,
                           d_hidden=D_HIDDEN, n_heads=N_HEADS, n_layers=N_LAYERS, dropout=0.0).to(DEVICE)
    decoder = GraphDecoder(d_latent=D_LATENT, d_out=D_IN, d_head=D_HEAD, d_expand=D_EXPAND,
                           d_hidden=D_HIDDEN, n_heads=N_HEADS, n_layers=N_LAYERS, dropout=0.0).to(DEVICE)

    loss_func = VAELoss(kl_weight=KL_WEIGHT)

    # Optimizer & Scheduler
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = AdamW(params, lr=LR)
    lr_scheduler = ReduceLROnPlateau(optimizer, factor=LR_REDUCE_FACTOR, patience=LR_REDUCE_PATIENCE, min_lr=LR_REDUCE_MIN)

    # Prepare Logging
    os.makedirs(LOG_DIR, exist_ok=True)
    writer = SummaryWriter(log_dir=LOG_DIR)
    mov_avg_kld = MovingAvg(MOV_AVG_LEN)
    mov_avg_mse = MovingAvg(MOV_AVG_LEN)
    global_step = 0
    best_loss = float("inf")
    plot_manager = PlotManager(5, 1, 2)

    with ProgressManager(len(dataloader), EPOCHS, 5, 2, ["KLD", "MSE", "lr"]) as progress:
        for e in range(EPOCHS):
            total_loss = 0
            for i, batch in enumerate(dataloader):
                batch: Dict[str, torch.Tensor]

                # count number of non-zero tokens for each batch graph
                segments = batch["graphs"].flatten(2)  # (B, G, 2, 2) -> (B, G, 4)
                valid_mask = torch.sum(torch.abs(segments), dim=-1) > 0  # (B, G)
                # add a dimension for segments indicating if it's a valid segment or padding
                segments = torch.cat([segments, valid_mask.unsqueeze(-1).float()], dim=-1)  # (B, G, 5)
                batch["segs"] = segments

                optimizer.zero_grad()

                # Forward pass through encoder and decoder
                z_mean, z_logvar = encoder(batch["segs"])  # Pass through encoder
                z = encoder.reparameterize(z_mean, z_logvar)           # Reparameterization trick
                reconstructed = decoder(z)                    # Pass through decoder

                # Compute VAE loss
                loss, mse_loss, kl_loss = loss_func(reconstructed, batch["segs"], z_mean, z_logvar)

                # Backpropagation
                loss.backward()
                optimizer.step()

                total_loss += loss
                global_step += 1
                mov_avg_mse.update(mse_loss)
                mov_avg_kld.update(kl_loss)

                # Progress update
                progress.update(e, i, KLD=mov_avg_mse.get(), MSE=mov_avg_kld.get(),
                                lr=optimizer.param_groups[0]['lr'])

                # Logging
                if global_step % 5 == 0:
                    writer.add_scalar("loss/KLD", mov_avg_kld.get(), global_step)
                    writer.add_scalar("loss/MSE", mov_avg_mse.get(), global_step)
                    writer.add_scalar("Learning Rate", optimizer.param_groups[0]["lr"], global_step)

            # Plot reconstructed segments and graphs
            plot_manager.plotSegments(batch["graphs"][0], 0, 0, "Original")
            plot_manager.plotSegments(reconstructed[0, :, :4].view(-1, 2, 2), 0, 1, "Reconstructed")
            writer.add_figure("Reconstructed Graphs", plot_manager.getFigure(), global_step)

            # Save models
            saveModels(os.path.join(LOG_DIR, "last.pth"), encoder, decoder)
            if total_loss < best_loss:
                best_loss = total_loss
                saveModels(os.path.join(LOG_DIR, "best.pth"), encoder, decoder)

            # Step scheduler
            lr_scheduler.step(total_loss)


if __name__ == "__main__":
    train()

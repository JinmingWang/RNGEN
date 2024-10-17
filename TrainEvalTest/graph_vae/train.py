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
from Models import GraphEncoder, GraphDecoder, KLLoss, HungarianLoss, HungarianMode


def train():
    # Dataset & DataLoader
    dataset = LaDeCachedDataset(DATA_DIR, max_trajs=N_TRAJS, set_name="train")
    dataloader = DataLoader(dataset, batch_size=B, shuffle=True, collate_fn=LaDeCachedDataset.collate_fn, drop_last=True)

    # Models
    encoder = GraphEncoder(d_latent=D_LATENT, d_head=D_HEAD, d_expand=D_EXPAND,
                           d_hidden=D_HIDDEN, n_heads=N_HEADS, n_layers=8, dropout=0.0).to(DEVICE)
    decoder = GraphDecoder(d_latent=D_LATENT, d_head=D_HEAD, d_expand=D_EXPAND,
                           d_hidden=D_HIDDEN, n_heads=N_HEADS, n_layers=4, dropout=0.0).to(DEVICE)

    torch.set_float32_matmul_precision('high')
    encoder = torch.compile(encoder)
    decoder = torch.compile(decoder)

    kl_loss_func = KLLoss(kl_weight=KL_WEIGHT)
    hu_loss_func = HungarianLoss(mode=HungarianMode.Seq, feature_weight=[1.0, 1.0, 0.0])
    mse_loss_func = torch.nn.MSELoss()

    # Optimizer & Scheduler
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = AdamW(params, lr=LR)
    lr_scheduler = ReduceLROnPlateau(optimizer, factor=LR_REDUCE_FACTOR, patience=LR_REDUCE_PATIENCE,
                                     min_lr=LR_REDUCE_MIN, threshold=LR_REDUCE_THRESHOLD)

    # Prepare Logging
    os.makedirs(LOG_DIR, exist_ok=True)
    writer = SummaryWriter(log_dir=LOG_DIR)
    mov_avg_kld = MovingAvg(MOV_AVG_LEN)
    mov_avg_nodes = MovingAvg(MOV_AVG_LEN)
    mov_avg_segs = MovingAvg(MOV_AVG_LEN)
    global_step = 0
    best_loss = float("inf")
    plot_manager = PlotManager(5, 2, 2)

    with ProgressManager(len(dataloader), EPOCHS, 5, 2, ["KLD", "Nodes", "Segs", "lr"]) as progress:
        for e in range(EPOCHS):
            total_loss = 0
            for i, batch in enumerate(dataloader):
                batch: Dict[str, torch.Tensor]

                batch |= LaDeCachedDataset.SegmentsToNodesAdj(batch["graphs"], N_NODES)

                # count number of non-zero tokens for each batch graph
                segments = batch["graphs"].flatten(2)  # (B, G, 2, 2) -> (B, G, 4)
                valid_mask = torch.sum(torch.abs(segments), dim=-1) > 0  # (B, G)
                # add a dimension for segments indicating if it's a valid segment or padding
                segments = torch.cat([segments, valid_mask.unsqueeze(-1).float()], dim=-1)  # (B, G, 5)
                # randomly permute the segments along the graph dimension
                perm = torch.randperm(segments.shape[1])
                batch["segs"] = segments[:, perm, :]

                # randomly permute the segments along the graph dimension
                perm = torch.randperm(batch["segs"].shape[1])
                batch["segs"] = batch["segs"][:, perm, :]

                optimizer.zero_grad()

                # Forward pass through encoder and decoder
                z_mean, z_logvar = encoder(batch["segs"])  # Pass through encoder
                z = encoder.reparameterize(z_mean, z_logvar)           # Reparameterization trick
                pred_nodes, pred_segs = decoder(z)                    # Pass through decoder

                # Compute VAE loss
                kl_loss = kl_loss_func(z_mean, z_logvar)
                nodes_loss = hu_loss_func(pred_nodes, batch["nodes"])
                segs_loss = mse_loss_func(pred_segs, batch["segs"])
                loss = kl_loss + nodes_loss + segs_loss

                # Backpropagation
                loss.backward()
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
                optimizer.step()

                total_loss += loss
                global_step += 1
                if kl_loss.item() >= 0.1:
                    continue
                mov_avg_kld.update(kl_loss)
                mov_avg_nodes.update(nodes_loss)
                mov_avg_segs.update(segs_loss)

                # Progress update
                progress.update(e, i, KLD=mov_avg_kld.get(), Nodes=mov_avg_nodes.get(), Segs=mov_avg_segs.get(),
                                lr=optimizer.param_groups[0]['lr'])

                # Logging
                if global_step % 5 == 0:
                    writer.add_scalar("loss/KLD", mov_avg_kld.get(), global_step)
                    writer.add_scalar("loss/Nodes", mov_avg_nodes.get(), global_step)
                    writer.add_scalar("loss/Segs", mov_avg_segs.get(), global_step)
                    writer.add_scalar("Learning Rate", optimizer.param_groups[0]["lr"], global_step)

            # Plot reconstructed segments and graphs
            plot_manager.plotSegments(batch["segs"][0], 0, 0, "Segs")
            plot_manager.plotSegments(pred_segs[0], 0, 1, "Pred Segs")
            plot_manager.plotNodes(batch["nodes"][0], 1, 0, "Nodes")
            plot_manager.plotNodes(pred_nodes[0], 1, 1, "Pred Nodes")
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

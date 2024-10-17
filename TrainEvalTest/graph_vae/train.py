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

    # torch.set_float32_matmul_precision('high')
    # encoder = torch.compile(encoder)
    # decoder = torch.compile(decoder)

    kl_loss_func = KLLoss(kl_weight=KL_WEIGHT)
    segs_loss_func = torch.nn.MSELoss()
    joints_loss_func = torch.nn.BCELoss()

    # Optimizer & Scheduler
    params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = AdamW(params, lr=LR)
    lr_scheduler = ReduceLROnPlateau(optimizer, factor=LR_REDUCE_FACTOR, patience=LR_REDUCE_PATIENCE,
                                     min_lr=LR_REDUCE_MIN, threshold=LR_REDUCE_THRESHOLD)

    # Prepare Logging
    os.makedirs(LOG_DIR, exist_ok=True)
    writer = SummaryWriter(log_dir=LOG_DIR)
    mov_avg_kld = MovingAvg(MOV_AVG_LEN)
    mov_avg_joints = MovingAvg(MOV_AVG_LEN)
    mov_avg_segs = MovingAvg(MOV_AVG_LEN)
    global_step = 0
    best_loss = float("inf")
    plot_manager = PlotManager(5, 2, 2)

    with ProgressManager(len(dataloader), EPOCHS, 5, 2, ["KLD", "Joints", "Segs", "lr"]) as progress:
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

                batch |= LaDeCachedDataset.getJointsFromSegments(batch["segs"])

                optimizer.zero_grad()

                # Forward pass through encoder and decoder
                z_mean, z_logvar = encoder(batch["segs"])  # Pass through encoder
                z = encoder.reparameterize(z_mean, z_logvar)           # Reparameterization trick
                pred_segs, pred_joints = decoder(z)                    # Pass through decoder

                # Compute VAE loss
                kl_loss = kl_loss_func(z_mean, z_logvar)
                joints_loss = joints_loss_func(pred_joints, batch["joints"])
                segs_loss = segs_loss_func(pred_segs, batch["segs"])
                loss = kl_loss + joints_loss + segs_loss

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
                mov_avg_joints.update(joints_loss)
                mov_avg_segs.update(segs_loss)

                # Progress update
                progress.update(e, i, KLD=mov_avg_kld.get(), Joints=mov_avg_joints.get(), Segs=mov_avg_segs.get(),
                                lr=optimizer.param_groups[0]['lr'])

                # Logging
                if global_step % 5 == 0:
                    writer.add_scalar("loss/KLD", mov_avg_kld.get(), global_step)
                    writer.add_scalar("loss/Joints", mov_avg_joints.get(), global_step)
                    writer.add_scalar("loss/Segs", mov_avg_segs.get(), global_step)
                    writer.add_scalar("Learning Rate", optimizer.param_groups[0]["lr"], global_step)

            # Plot reconstructed segments and graphs
            pred_segs[0] = matchJoints(pred_segs[0], pred_joints[0])
            plot_manager.plotSegments(batch["segs"][0], 0, 0, "Segs")
            plot_manager.plotSegments(pred_segs[0], 0, 1, "Pred Segs")
            plot_manager.plotHeatmap(batch["joints"][0], 1, 0, "Joints")
            plot_manager.plotHeatmap(pred_joints[0], 1, 1, "Pred Joints")
            writer.add_figure("Reconstructed Graphs", plot_manager.getFigure(), global_step)

            # Save models
            saveModels(os.path.join(LOG_DIR, "last.pth"), encoder, decoder)
            if total_loss < best_loss:
                best_loss = total_loss
                saveModels(os.path.join(LOG_DIR, "best.pth"), encoder, decoder)

            # Step scheduler
            lr_scheduler.step(total_loss)


def matchJoints(segments: torch.Tensor, joints: torch.Tensor) -> torch.Tensor:
    # segs: (N, 5), joints: (N, N)
    joints = (joints >= 0.5).to(torch.float32)
    # remove diagonal
    joints = joints - torch.eye(joints.shape[0], device=joints.device)

    segs = segments[:, :4]  # Only use (x1, y1, x2, y2)
    N = segs.shape[0]

    for seg_i in range(N):
        neighbor_ids = torch.nonzero(joints[seg_i]).flatten()  # Get neighbors for segment i
        if len(neighbor_ids) == 0:
            continue  # No neighbors, move to the next segment

        # Compute pairwise distances between p1 and p2 of seg_i and the neighbors
        dist_p1n1 = torch.norm(segs[seg_i, 0:2] - segs[neighbor_ids, 0:2], dim=1)  # Distance between p1 of seg_i and p1 of neighbors
        dist_p1n2 = torch.norm(segs[seg_i, 0:2] - segs[neighbor_ids, 2:4], dim=1)  # Distance between p1 of seg_i and p2 of neighbors
        dist_p2n1 = torch.norm(segs[seg_i, 2:4] - segs[neighbor_ids, 0:2], dim=1)  # Distance between p2 of seg_i and p1 of neighbors
        dist_p2n2 = torch.norm(segs[seg_i, 2:4] - segs[neighbor_ids, 2:4], dim=1)  # Distance between p2 of seg_i and p2 of neighbors

        # Stack the distances and find the minimum for each neighbor
        distances = torch.stack([dist_p1n1, dist_p1n2, dist_p2n1, dist_p2n2], dim=0)  # Shape (4, N_i)
        min_dist, min_idx = torch.min(distances, dim=0)  # Get minimum distance index for each neighbor

        # Identify matching points
        p1n1_match = min_idx == 0
        p1n2_match = min_idx == 1
        p2n1_match = min_idx == 2
        p2n2_match = min_idx == 3

        # Compute the mean of the matched points
        p1_sum = segs[seg_i, 0:2] + segs[neighbor_ids][p1n1_match, 0:2].sum(dim=0) + segs[neighbor_ids][p1n2_match][:, 2:4].sum(dim=0)
        p1_mean = p1_sum / (1 + p1n1_match.sum() + p1n2_match.sum())  # Take average

        p2_sum = segs[seg_i, 2:4] + segs[neighbor_ids][p2n1_match, 0:2].sum(dim=0) + segs[neighbor_ids][p2n2_match][:, 2:4].sum(dim=0)
        p2_mean = p2_sum / (1 + p2n1_match.sum() + p2n2_match.sum())  # Take average

        # Update current segment's points
        segs[seg_i, 0:2] = p1_mean
        segs[seg_i, 2:4] = p2_mean

        # Update the matched neighbors' points accordingly
        segs[neighbor_ids[p1n1_match], 0:2] = p1_mean  # p1n1: update p2 of neighbors
        segs[neighbor_ids[p1n2_match], 2:4] = p1_mean  # p1n2: update p1 of neighbors
        segs[neighbor_ids[p2n1_match], 0:2] = p2_mean  # p2n1: update p2 of neighbors
        segs[neighbor_ids[p2n2_match], 2:4] = p2_mean  # p2n2: update p1 of neighbors

        # Mark processed neighbors to avoid double-processing
        joints[neighbor_ids, seg_i] = 0

    return torch.cat([segs, segments[:, 4:]], dim=-1)



if __name__ == "__main__":
    train()

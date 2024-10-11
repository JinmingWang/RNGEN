from TrainEvalTest.GlobalConfigs import *
from TrainEvalTest.segment_model.configs import *
from TrainEvalTest.Utils import *
from TrainEvalTest.segment_model.eval import eval

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import os

from Dataset import DEVICE, LaDeCachedDataset
from Models import SegmentsModel, Encoder, HungarianLoss_Sequential
from Diffusion import DDPM


def train():
    # Dataset & DataLoader
    dataset = LaDeCachedDataset(DATA_DIR, max_trajs=N_TRAJS, set_name="train")
    dataloader = DataLoader(dataset, batch_size=B, shuffle=True, collate_fn=LaDeCachedDataset.collate_fn, drop_last=True)

    # Models
    encoder = Encoder(N_TRAJS, L_TRAJ, D_TRAJ_ENC).to(DEVICE)
    diffusion_net = SegmentsModel(n_seg=N_SEGS, d_seg=5, d_traj_enc=D_TRAJ_ENC, n_traj=N_TRAJS, T=T).to(DEVICE)
    # encoder, diffusion_net = loadModels("Runs/NodeEdgeModel_2024-10-07_04-50-30/last.pth", encoder, diffusion_net)
    ddpm = DDPM(BETA_MIN, BETA_MAX, T, DEVICE, "quadratic")
    # loss_func = torch.nn.MSELoss()
    loss_func = HungarianLoss_Sequential('l1')

    # Optimizer & Scheduler
    optimizer = AdamW([{"params": diffusion_net.parameters(), "lr": LR_DIFFUSION},
                       {"params": encoder.parameters(), "lr": LR_ENCODER}], lr=LR_DIFFUSION)
    lr_scheduler = ReduceLROnPlateau(optimizer, factor=LR_REDUCE_FACTOR, patience=LR_REDUCE_PATIENCE, min_lr=LR_REDUCE_MIN, threshold=LR_REDUCE_THRESHOLD)

    # Prepare Logging
    os.makedirs(LOG_DIR)
    writer = SummaryWriter(log_dir=LOG_DIR)
    mov_avg_loss = MovingAvg(MOV_AVG_LEN)
    global_step = 0
    best_loss = float("inf")

    with ProgressManager(len(dataloader), EPOCHS, 5, 2, ["Loss", "lr"]) as progress:
        for e in range(EPOCHS):
            total_loss = 0
            for i, batch in enumerate(dataloader):
                batch: Dict[str, Tensor]

                # count number of non-zero tokens for each batch graph
                segments = batch["graphs"].flatten(2)   # (B, G, 2, 2) -> (B, G, 4)
                valid_mask = torch.sum(torch.abs(segments), dim=-1) > 0 # (B, G)
                # add a dimension for segments indicating if it's a valid segment or padding
                segments = torch.cat([segments, valid_mask.unsqueeze(-1).float()], dim=-1)  # (B, G, 5)
                batch["segs"] = segments

                noise = torch.randn_like(batch["segs"])

                t = torch.randint(0, T, (B,)).to(DEVICE)
                noisy_segs = ddpm.diffusionForward(batch["segs"], t, noise)

                s = t - 1
                less_noisy_segs = torch.zeros_like(noisy_segs)
                less_noisy_segs[t!=0] = ddpm.diffusionForward(batch["segs"][t!=0], s[t!=0], noise[t!=0])
                less_noisy_segs[t==0] = batch["segs"][t==0]

                optimizer.zero_grad()
                traj_enc = encoder(batch["trajs"])
                pred_noise = diffusion_net(noisy_segs, traj_enc, t)

                pred_segs = ddpm.diffusionBackwardStep(noisy_segs, t, pred_noise, disable_noise=True)

                loss = loss_func(pred_segs, less_noisy_segs)

                loss.backward()

                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(diffusion_net.parameters(), 1.0)

                optimizer.step()

                total_loss += loss.item()
                global_step += 1
                mov_avg_loss.update(loss)

                progress.update(e, i, Loss=mov_avg_loss.get(), lr=optimizer.param_groups[0]['lr'])

                if global_step % LOG_INTERVAL == 0:
                    writer.add_scalar("loss/Loss", mov_avg_loss.get(), global_step)
                    writer.add_scalar("lr", optimizer.param_groups[0]["lr"], global_step)

                if global_step % EVAL_INTERVAL == 0:
                    figure, eval_loss = eval(batch, encoder, diffusion_net, ddpm)
                    writer.add_figure("Evaluation", figure, global_step)
                    writer.add_scalar("loss/eval", eval_loss.item(), global_step)

            saveModels(LOG_DIR + "last.pth", encoder, diffusion_net)
            if total_loss < best_loss:
                best_loss = total_loss
                saveModels(LOG_DIR + "best.pth", encoder, diffusion_net)

            lr_scheduler.step(total_loss)


if __name__ == "__main__":
    train()

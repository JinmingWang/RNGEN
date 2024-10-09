from TrainEvalTest.GlobalConfigs import *
from TrainEvalTest.node_edge_model.configs import *
from TrainEvalTest.Utils import *
from TrainEvalTest.node_edge_model.eval import eval

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import os

from Dataset import DEVICE, LaDeCachedDataset
from Models import DiffusionNetwork, Encoder, HungarianLoss
from Diffusion import DDPM


def train():
    # Dataset & DataLoader
    dataset = LaDeCachedDataset(DATA_DIR, max_trajs=N_TRAJS, set_name="train")
    dataloader = DataLoader(dataset, batch_size=B, shuffle=True, collate_fn=LaDeCachedDataset.collate_fn, drop_last=True)

    # Models
    encoder = Encoder(N_TRAJS, L_TRAJ, D_TRAJ_ENC).to(DEVICE)
    diffusion_net = DiffusionNetwork(num_nodes=N_NODES, traj_encoding_c=D_TRAJ_ENC, traj_num=N_TRAJS, T=T).to(DEVICE)
    # encoder, diffusion_net = loadModels("Runs/NodeEdgeModel_2024-10-07_04-50-30/last.pth", encoder, diffusion_net)
    ddpm = DDPM(BETA_MIN, BETA_MAX, T, DEVICE, "quadratic")
    # loss_func = torch.nn.MSELoss()
    loss_func = HungarianLoss('l1')

    # Optimizer & Scheduler
    optimizer = AdamW([{"params": diffusion_net.parameters(), "lr": LR_DIFFUSION},
                       {"params": encoder.parameters(), "lr": LR_ENCODER}], lr=LR_DIFFUSION)
    lr_scheduler = ReduceLROnPlateau(optimizer, factor=LR_REDUCE_FACTOR, patience=LR_REDUCE_PATIENCE, min_lr=LR_REDUCE_MIN)

    # Prepare Logging
    os.makedirs(LOG_DIR)
    writer = SummaryWriter(log_dir=LOG_DIR)
    mov_avg_node_loss = MovingAvg(MOV_AVG_LEN)
    mov_avg_mat_loss = MovingAvg(MOV_AVG_LEN)
    global_step = 0
    best_loss = float("inf")

    with ProgressManager(len(dataloader), EPOCHS, 5, 2, ["NodeLoss", "MatLoss", "lr"]) as progress:
        for e in range(EPOCHS):
            total_loss = 0
            for i, batch in enumerate(dataloader):
                batch: Dict[str, Tensor]

                batch |= LaDeCachedDataset.SegmentsToNodesAdj(batch["graphs"], N_NODES)

                node_noise = torch.randn_like(batch["nodes"])
                adj_mat_noise = torch.randn_like(batch["adj_mats"])
                setPaddingToZero(batch["n_nodes"], [node_noise], [adj_mat_noise])

                t = torch.randint(0, T, (B,)).to(DEVICE)
                noisy_nodes = ddpm.diffusionForward(batch["nodes"], t, node_noise)
                noisy_adj_mat = ddpm.diffusionForward(batch["adj_mats"], t, adj_mat_noise)

                s = t - 1
                less_noisy_nodes = torch.zeros_like(noisy_nodes)
                less_noisy_nodes[t!=0] = ddpm.diffusionForward(batch["nodes"][t!=0], s[t!=0], node_noise[t!=0])
                less_noisy_nodes[t==0] = batch["nodes"][t==0]

                less_noisy_adj_mat = torch.zeros_like(noisy_adj_mat)
                less_noisy_adj_mat[t!=0] = ddpm.diffusionForward(batch["adj_mats"][t!=0], s[t!=0], adj_mat_noise[t!=0])
                less_noisy_adj_mat[t==0] = batch["adj_mats"][t==0]

                optimizer.zero_grad()
                traj_enc = encoder(batch["trajs"])
                pred_node_noise, pred_adj_mat_noise = diffusion_net(noisy_nodes, noisy_adj_mat, traj_enc, t)
                setPaddingToZero(batch["n_nodes"], [pred_node_noise], [pred_adj_mat_noise])

                pred_node = ddpm.diffusionBackwardStep(noisy_nodes, t, pred_node_noise, disable_noise=True)
                pred_adj_mat = ddpm.diffusionBackwardStep(noisy_adj_mat, t, pred_adj_mat_noise, disable_noise=True)

                node_loss = loss_func(pred_node, less_noisy_nodes)
                adj_mat_loss = loss_func(pred_adj_mat, less_noisy_adj_mat)

                loss = node_loss + adj_mat_loss

                loss.backward()

                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), 2.0)
                torch.nn.utils.clip_grad_norm_(diffusion_net.parameters(), 2.0)

                optimizer.step()

                total_loss += loss.item()
                global_step += 1
                mov_avg_node_loss.update(node_loss)
                mov_avg_mat_loss.update(adj_mat_loss)

                progress.update(e, i, NodeLoss=mov_avg_node_loss.get(), MatLoss=mov_avg_mat_loss.get(), lr=optimizer.param_groups[0]['lr'])

                if global_step % LOG_INTERVAL == 0:
                    writer.add_scalar("loss/nodes", mov_avg_node_loss.get(), global_step)
                    writer.add_scalar("loss/adj_mat", mov_avg_mat_loss.get(), global_step)
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

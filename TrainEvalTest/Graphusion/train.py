from TrainEvalTest.GlobalConfigs import *
from TrainEvalTest.Graphusion.configs import *
from TrainEvalTest.Utils import *
from TrainEvalTest.Graphusion.eval import getEvalFunction

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

import os

from Dataset import DEVICE, RoadNetworkDataset
from Models import Graphusion, GraphusionVAE
from Diffusion import DDIM

def train():
    # Dataset & DataLoader
    dataset = RoadNetworkDataset("Dataset/Tokyo_10k_sparse",
                                 batch_size=B,
                                 drop_last=True,
                                 set_name="train",
                                 enable_aug=True,
                                 img_H=16,
                                 img_W=16,
                                 need_nodes=True
                                 )

    # Models
    vae = GraphusionVAE(d_node=2, d_edge=16, d_latent=128, d_hidden=256, n_layers=8, n_heads=8).to(DEVICE)
    loadModels("Runs/GraphusionVAE/241206_1058_initial/last.pth", vae=vae)
    vae.eval()

    graphusion = Graphusion(D_in=128,
                            L_enc=dataset.max_N_nodes,
                            N_trajs=dataset.N_trajs,
                            L_traj=dataset.max_L_traj,
                            d_context=2,
                            n_layers=6,
                            T=T)

    torch.set_float32_matmul_precision("high")
    torch.compile(graphusion)

    loadModels("Runs/Graphusion/241206_2341_initial/last.pth", graphusion=graphusion)

    graphusion = graphusion.to(DEVICE)

    eval = getEvalFunction(vae)

    ddim = DDIM(BETA_MIN, BETA_MAX, T, DEVICE, "quadratic", skip_step=10, data_dim=2)
    loss_func = torch.nn.MSELoss()

    # Optimizer & Scheduler
    optimizer = AdamW(graphusion.parameters(), lr=LR, amsgrad=True)
    lr_scheduler = ReduceLROnPlateau(optimizer, factor=LR_REDUCE_FACTOR, patience=LR_REDUCE_PATIENCE,
                                     min_lr=LR_REDUCE_MIN, threshold=LR_REDUCE_THRESHOLD)

    # Prepare Logging
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    writer = SummaryWriter(log_dir=LOG_DIR)
    mov_avg_loss = MovingAvg(MOV_AVG_LEN * len(dataset))
    global_step = 0
    best_loss = float("inf")

    with ProgressManager(len(dataset), EPOCHS, 5, 2, ["Loss", "lr"]) as progress:
        for e in range(EPOCHS):
            total_loss = 0
            for i, batch in enumerate(dataset):
                batch: Dict[str, Tensor]

                with torch.no_grad():
                    latent, _ = vae.encode(batch["nodes"], batch["edges"], batch["adj_mat"])    # (B, N_nodes, 128)

                latent_noise = torch.randn_like(latent)

                t = torch.randint(0, T, (B,)).to(DEVICE)
                latent_noisy = ddim.diffusionForward(latent, t, latent_noise)

                latent_noise_pred = graphusion(latent_noisy, batch["trajs"], t)

                loss = loss_func(latent_noise_pred, latent_noise) * 100
                loss.backward()
                torch.nn.utils.clip_grad_norm_(graphusion.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

                total_loss += loss.item()
                global_step += 1
                mov_avg_loss.update(loss.item())

                progress.update(e, i, Loss=mov_avg_loss.get(), lr=optimizer.param_groups[0]['lr'])

                if global_step % LOG_INTERVAL == 0:
                    writer.add_scalar("loss/Loss", mov_avg_loss.get(), global_step)
                    writer.add_scalar("lr", optimizer.param_groups[0]["lr"], global_step)

            if e % EVAL_INTERVAL == 0:
                figure, eval_loss = eval(graphusion, ddim)
                writer.add_figure("Evaluation", figure, global_step)
                writer.add_scalar("loss/eval", eval_loss, global_step)

            saveModels(LOG_DIR + "last.pth", graphusion=graphusion)
            if total_loss < best_loss:
                best_loss = total_loss
                saveModels(LOG_DIR + "best.pth", graphusion=graphusion)

            lr_scheduler.step(total_loss)


if __name__ == "__main__":
    train()

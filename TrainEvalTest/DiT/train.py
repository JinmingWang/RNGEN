from TrainEvalTest.GlobalConfigs import *
from TrainEvalTest.DiT.configs import *
from TrainEvalTest.Utils import *
from TrainEvalTest.DiT.eval import getEvalFunction

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

import os

from Dataset import DEVICE, RoadNetworkDataset
from Models import RoutesDiT, CrossDomainVAE
from Diffusion import DDIM

def train():
    # Dataset & DataLoader
    dataset = RoadNetworkDataset("Dataset/Tokyo_10k_sparse",
                                 batch_size=B,
                                 drop_last=True,
                                 set_name="train",
                                 enable_aug=True,
                                 img_H=16,
                                 img_W=16
                                 )

    # Models
    vae = CrossDomainVAE(N_routes=dataset.N_trajs, L_route=dataset.max_L_route,
                         N_interp=dataset.N_interp, threshold=0.5).to(DEVICE)
    loadModels("Runs/CDVAE/241127_1833_sparse_kl1e-6/last.pth", vae=vae)
    vae.eval()

    DiT = RoutesDiT(D_in=dataset.N_interp * 2,
                    N_routes=dataset.N_trajs,
                    L_route=dataset.max_L_route,
                    L_traj=dataset.max_L_traj,
                    d_context=2,
                    n_layers=8,
                    T=T)

    torch.set_float32_matmul_precision("high")
    torch.compile(DiT)

    DiT = DiT.to(DEVICE)

    eval = getEvalFunction(vae)

    ddim = DDIM(BETA_MIN, BETA_MAX, T, DEVICE, "quadratic", skip_step=10, data_dim=3)
    loss_func = torch.nn.MSELoss()

    # Optimizer & Scheduler
    optimizer = AdamW(DiT.parameters(), lr=LR, amsgrad=True)
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
                    latent, _ = vae.encode(batch["routes"])    # (B, N_trajs, L_route, N_interp*2)

                latent_noise = torch.randn_like(latent)

                t = torch.randint(0, T, (B,)).to(DEVICE)
                latent_noisy = ddim.diffusionForward(latent, t, latent_noise)

                latent_noise_pred = DiT(latent_noisy, batch["trajs"], t)

                loss = loss_func(latent_noise_pred, latent_noise) * 100
                loss.backward()
                torch.nn.utils.clip_grad_norm_(DiT.parameters(), 1.0)
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
                figure, eval_loss = eval(DiT, ddim)
                writer.add_figure("Evaluation", figure, global_step)
                writer.add_scalar("loss/eval", eval_loss, global_step)

            saveModels(LOG_DIR + "last.pth", DiT=DiT)
            if total_loss < best_loss:
                best_loss = total_loss
                saveModels(LOG_DIR + "best.pth", DiT=DiT)

            lr_scheduler.step(total_loss)


if __name__ == "__main__":
    train()

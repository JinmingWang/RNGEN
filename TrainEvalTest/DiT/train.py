from TrainEvalTest.GlobalConfigs import *
from TrainEvalTest.DiT.configs import *
from TrainEvalTest.Utils import *
from TrainEvalTest.DiT.eval import getEvalFunction

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import os

from Dataset import DEVICE, LaDeCachedDataset
from Models import PathsDiT, CrossDomainVAE
from Diffusion import DDIM


def prepareModels() -> Dict[str, torch.nn.Module]:
    vae = CrossDomainVAE(N_paths=N_TRAJS, L_path=L_PATH, D_enc=4, threshold=0.5).to(DEVICE)
    vae.eval()

    DiT = PathsDiT(n_paths=N_TRAJS, l_path=L_PATH, d_context=2, n_layers=4, T=T).to(DEVICE)

    # loadModels("Runs/SegmentsModel/20241029_021329_Use_Paths/last.pth", SegmentsModel=DiT, PathEncoder=traj_encoder)
    return {"DiT": DiT, "VAE": vae}


def train():
    # Dataset & DataLoader
    dataset = LaDeCachedDataset(DATA_DIR, max_trajs=N_TRAJS, set_name="train")
    dataloader = DataLoader(dataset, batch_size=B, shuffle=True, collate_fn=LaDeCachedDataset.collate_fn, drop_last=True)

    # Models
    models = prepareModels()

    eval = getEvalFunction(models["VAE"])

    ddim = DDIM(BETA_MIN, BETA_MAX, T, DEVICE, "quadratic", skip_step=1, data_dim=3)
    loss_func = torch.nn.MSELoss()

    # Optimizer & Scheduler
    optimizer = AdamW(models["DiT"].parameters(), lr=LR, amsgrad=True)
    lr_scheduler = ReduceLROnPlateau(optimizer, factor=LR_REDUCE_FACTOR, patience=LR_REDUCE_PATIENCE,
                                     min_lr=LR_REDUCE_MIN, threshold=LR_REDUCE_THRESHOLD)

    # Prepare Logging
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    writer = SummaryWriter(log_dir=LOG_DIR)
    mov_avg_loss = MovingAvg(MOV_AVG_LEN * len(dataloader))
    global_step = 0
    best_loss = float("inf")

    with ProgressManager(len(dataloader), EPOCHS, 5, 2, ["Loss", "lr"]) as progress:
        for e in range(EPOCHS):
            total_loss = 0
            for i, batch in enumerate(dataloader):
                batch: Dict[str, Tensor]

                with torch.no_grad():
                    latent, _ = models["VAE"].encode(batch["paths"])    # (B, N, L, 4)

                latent_noise = torch.randn_like(latent)

                t = torch.randint(0, T, (B,)).to(DEVICE)
                latent_noisy = ddim.diffusionForward(latent, t, latent_noise)

                optimizer.zero_grad()

                latent_pred = models["DiT"](latent_noisy, batch["trajs"], t)

                loss = loss_func(latent_pred, latent)

                loss.backward()

                # Gradient Clipping
                # torch.nn.utils.clip_grad_norm_(models["DiT"].parameters(), 1.2)

                optimizer.step()

                total_loss += loss.item()
                global_step += 1
                mov_avg_loss.update(loss)

                progress.update(e, i, Loss=mov_avg_loss.get(), lr=optimizer.param_groups[0]['lr'])

                if global_step % LOG_INTERVAL == 0:
                    writer.add_scalar("loss/Loss", mov_avg_loss.get(), global_step)
                    writer.add_scalar("lr", optimizer.param_groups[0]["lr"], global_step)

            if e % EVAL_INTERVAL == 0:
                figure, eval_loss = eval(models, ddim)
                writer.add_figure("Evaluation", figure, global_step)
                writer.add_scalar("loss/eval", eval_loss.item(), global_step)

            saveModels(LOG_DIR + "last.pth", PathsDiT=models["DiT"])
            if total_loss < best_loss:
                best_loss = total_loss
                saveModels(LOG_DIR + "best.pth", PathsDiT=models["DiT"])

            lr_scheduler.step(total_loss)


if __name__ == "__main__":
    train()

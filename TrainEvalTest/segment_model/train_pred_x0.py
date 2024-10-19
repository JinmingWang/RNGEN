from TrainEvalTest.GlobalConfigs import *
from TrainEvalTest.segment_model.configs import *
from TrainEvalTest.Utils import *
from TrainEvalTest.segment_model.eval_pred_x0 import eval

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import os

from Dataset import DEVICE, LaDeCachedDataset
from Models import SegmentsModel, TrajEncoder, HungarianLoss, HungarianMode, GraphEncoder, GraphDecoder
from Diffusion import DDIM


def prepareModels() -> Dict[str, torch.nn.Module]:
    traj_encoder = TrajEncoder(N_TRAJS, L_TRAJ, 128).to(DEVICE)
    graph_encoder = GraphEncoder(d_latent=16, d_head=64, d_expand=512, d_hidden=128, n_heads=16, n_layers=8, dropout=0.0).to(DEVICE)
    graph_decoder = GraphDecoder(d_latent=16, d_head=64, d_expand=512, d_hidden=128, n_heads=16, n_layers=4, dropout=0.0).to(DEVICE)
    DiT = SegmentsModel(d_seg=16, n_seg=N_SEGS, d_traj_enc=128, n_traj=N_TRAJS, T=T, pred_x0=True).to(DEVICE)

    # Load pre-trained graph VAE, it will be always frozen
    loadModels(GRAPH_VAE_WEIGHT, graph_encoder, graph_decoder)
    graph_encoder.eval()
    graph_decoder.eval()

    traj_encoder.eval()

    # torch.set_float32_matmul_precision('high')
    # traj_encoder = torch.compile(traj_encoder)
    # graph_encoder = torch.compile(graph_encoder)
    # graph_decoder = torch.compile(graph_decoder)
    # DiT = torch.compile(DiT)

    return {"traj_encoder": traj_encoder, "graph_encoder": graph_encoder, "graph_decoder": graph_decoder, "DiT": DiT}


def train():
    # Dataset & DataLoader
    dataset = LaDeCachedDataset(DATA_DIR, max_trajs=N_TRAJS, set_name="train")
    dataloader = DataLoader(dataset, batch_size=B, shuffle=True, collate_fn=LaDeCachedDataset.collate_fn, drop_last=True)

    # Models
    models = prepareModels()

    ddim = DDIM(BETA_MIN, BETA_MAX, T, DEVICE, "quadratic", skip_step=1)
    # loss_func = torch.nn.MSELoss()
    loss_func = HungarianLoss(HungarianMode.Seq)

    # Optimizer & Scheduler
    optimizer = AdamW([
        {"params": models["DiT"].parameters(), "lr": LR},
        {"params": models["traj_encoder"].parameters(), "lr":1e-7}
        ],
        lr=LR)
    lr_scheduler = ReduceLROnPlateau(optimizer, factor=LR_REDUCE_FACTOR, patience=LR_REDUCE_PATIENCE,
                                     min_lr=LR_REDUCE_MIN, threshold=LR_REDUCE_THRESHOLD)

    # Prepare Logging
    os.makedirs(LOG_DIR)
    writer = SummaryWriter(log_dir=LOG_DIR)
    mov_avg_loss = MovingAvg(MOV_AVG_LEN)
    global_step = 0
    best_loss = float("inf")

    with ProgressManager(len(dataloader), EPOCHS, 5, 2, ["Loss", "lr"]) as progress:
        for e in range(EPOCHS):
            if e == RELEASE_PATH_ENC:
                models["traj_encoder"].train()
                optimizer.param_groups[1]["lr"] = optimizer.param_groups[0]["lr"]
            total_loss = 0
            for i, batch in enumerate(dataloader):
                batch: Dict[str, Tensor]

                # count number of non-zero tokens for each batch graph
                segments = batch["graphs"].flatten(2)   # (B, G, 2, 2) -> (B, G, 4)
                valid_mask = torch.sum(torch.abs(segments), dim=-1) > 0 # (B, G)
                # add a dimension for segments indicating if it's a valid segment or padding
                segments = torch.cat([segments, valid_mask.unsqueeze(-1).float()], dim=-1)  # (B, G, 5)
                batch["segs"] = segments

                with torch.no_grad():
                    batch["graph_enc"], _ = models["graph_encoder"](segments)

                noise = torch.randn_like(batch["graph_enc"])

                t = torch.randint(0, T, (B,)).to(DEVICE)
                noisy_graph_enc = ddim.diffusionForward(batch["graph_enc"], t, noise)

                # s = t - 1
                # less_noisy_segs = torch.zeros_like(noisy_segs)
                # less_noisy_segs[t!=0] = ddim.diffusionForward(batch["segs"][t!=0], s[t!=0], noise[t!=0])
                # less_noisy_segs[t==0] = batch["segs"][t==0]

                optimizer.zero_grad()

                traj_enc = models["traj_encoder"](batch["trajs"])
                pred_graph_enc = models["DiT"](noisy_graph_enc, traj_enc, t)

                # pred_less_noisy_segs = ddim.diffusionBackwardStepWithx0(pred_segs, t, s, pred_noise)

                loss = loss_func(pred_graph_enc, batch["graph_enc"])

                loss.backward()

                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(models["DiT"].parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(models["traj_encoder"].parameters(), 1.0)

                optimizer.step()

                total_loss += loss.item()
                global_step += 1
                mov_avg_loss.update(loss)

                progress.update(e, i, Loss=mov_avg_loss.get(), lr=optimizer.param_groups[0]['lr'])

                if global_step % LOG_INTERVAL == 0:
                    writer.add_scalar("loss/Loss", mov_avg_loss.get(), global_step)
                    writer.add_scalar("lr", optimizer.param_groups[0]["lr"], global_step)

            if e % EVAL_INTERVAL == 0:
                figure, eval_loss = eval(batch, models, ddim)
                writer.add_figure("Evaluation", figure, global_step)
                writer.add_scalar("loss/eval", eval_loss.item(), global_step)

            saveModels(LOG_DIR + "last.pth", models["DiT"], models["traj_encoder"])
            if total_loss < best_loss:
                best_loss = total_loss
                saveModels(LOG_DIR + "best.pth", models["DiT"], models["traj_encoder"])

            lr_scheduler.step(total_loss)


if __name__ == "__main__":
    train()

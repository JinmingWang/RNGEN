from TrainEvalTest.GlobalConfigs import *
from TrainEvalTest.path_encoder.configs import *
from TrainEvalTest.Utils import *
from TrainEvalTest.path_encoder.eval import eval

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import os

from Dataset import DEVICE, LaDeCachedDataset
from Models import SegmentsModel, PathEncoder, GraphEncoder, GraphDecoder, HungarianLoss, HungarianMode
from Diffusion import DDIM


def prepareModels() -> Dict[str, torch.nn.Module]:
    traj_encoder = PathEncoder(N_TRAJS, L_TRAJ, 9, 128).to(DEVICE)
    graph_encoder = GraphEncoder(d_latent=16, d_head=64, d_expand=512, d_hidden=128, n_heads=16, n_layers=8, dropout=0.0).to(DEVICE)
    graph_decoder = GraphDecoder(d_latent=16, d_head=64, d_expand=512, d_hidden=128, n_heads=16, n_layers=4, dropout=0.0).to(DEVICE)
    DiT = SegmentsModel(d_in=16, d_traj_enc=128, n_layers=12, T=T, pred_x0=True).to(DEVICE)

    # Model-S:
    # traj_encoder: d_encode = 128
    # DiT: d_in = 16, d_traj_enc = 128, n_layers = 12

    # Model-M:
    # traj_encoder: d_encode = 192
    # DiT: d_in = 16, d_traj_enc = 192, n_layers = 14

    # Model-L:
    # traj_encoder: d_encode = 256
    # DiT: d_in = 16, d_traj_enc = 256, n_layers = 16

    # Load pre-trained graph VAE, it will be always frozen
    loadModels(GRAPH_VAE_WEIGHT, graph_encoder, graph_decoder)
    graph_encoder.eval()
    graph_decoder.eval()

    traj_encoder.eval()

    torch.set_float32_matmul_precision('high')
    traj_encoder = torch.compile(traj_encoder)
    graph_encoder = torch.compile(graph_encoder)
    graph_decoder = torch.compile(graph_decoder)
    DiT = torch.compile(DiT)

    return {"traj_encoder": traj_encoder, "graph_encoder": graph_encoder, "graph_decoder": graph_decoder, "DiT": DiT}


def train():
    # Dataset & DataLoader
    dataset = LaDeCachedDataset(DATA_DIR, max_trajs=N_TRAJS, set_name="train")
    dataloader = DataLoader(dataset, batch_size=B, shuffle=True, collate_fn=LaDeCachedDataset.collate_fn, drop_last=True)

    # Models
    models = prepareModels()

    loss_func = HungarianLoss(HungarianMode.Seq)
    # loss_func = torch.nn.MSELoss()

    # Optimizer & Scheduler
    optimizer = AdamW([
        {"params": models["DiT"].parameters(), "lr": LR},
        {"params": models["traj_encoder"].parameters(), "lr":LR}
        ],
        lr=LR)
    lr_scheduler = ReduceLROnPlateau(optimizer, factor=LR_REDUCE_FACTOR, patience=LR_REDUCE_PATIENCE,
                                     min_lr=LR_REDUCE_MIN, threshold=LR_REDUCE_THRESHOLD)

    # Prepare Logging
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
                    batch["graph_enc"], _ = models["graph_encoder"](batch["segs"])

                noise = torch.randn_like(batch["graph_enc"])

                t = torch.zeros((B,), device=DEVICE, dtype=torch.long)

                optimizer.zero_grad()

                traj_enc = models["traj_encoder"](batch["trajs"])
                pred = models["DiT"](noise, traj_enc, t)

                loss = loss_func(pred, batch["graph_enc"])

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
                with torch.no_grad():
                    pred_segs, pred_joints = models["graph_decoder"](pred)

                pred_segs_jointed = matchJoints(pred_segs[0], pred_joints[0])
                joints = LaDeCachedDataset.getJointsFromSegments(batch["segs"][0:1])["joints"]

                eval_loss = loss_func(pred_segs, batch["segs"])

                plot_manager = PlotManager(5, 2, 3)
                plot_manager.plotSegments(batch["segs"][0], 0, 0, "Segs")
                plot_manager.plotSegments(pred_segs[0], 0, 1, f"Pred segs (Loss: {loss.item():.3e})")
                plot_manager.plotSegments(pred_segs_jointed, 0, 2, "Pred segs jointed")
                plot_manager.plotTrajs(batch["trajs"][0], 1, 0, "Trajectories")
                plot_manager.plotHeatmap(joints[0], 1, 1, "Joints")
                plot_manager.plotHeatmap(pred_joints[0], 1, 2, "Pred joints")

                writer.add_figure("Evaluation", plot_manager.getFigure(), global_step)
                writer.add_scalar("loss/eval", eval_loss.item(), global_step)

            saveModels(LOG_DIR + "last.pth", models["DiT"], models["traj_encoder"])
            if total_loss < best_loss:
                best_loss = total_loss
                saveModels(LOG_DIR + "best.pth", models["DiT"], models["traj_encoder"])

            lr_scheduler.step(total_loss)


if __name__ == "__main__":
    train()

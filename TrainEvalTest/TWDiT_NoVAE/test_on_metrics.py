from TrainEvalTest.Utils import *
from TrainEvalTest.Metrics import *
from Diffusion import DDIM
from tqdm import tqdm
import cv2

import torch

from Dataset import DEVICE, RoadNetworkDataset
from Models import TWDiT_MHSA, Deduplicator, heuristicDeduplication


def segsToHeatmaps(batch_segs: Tensor, batch_trajs: Tensor, traj_lens: Tensor, img_H: int, img_W: int, line_width: int):
    # segs: B * (N_segs, N_interp, 2)
    B = len(batch_segs)

    heatmaps = []

    for i in range(B):
        # Get the bounding box of the segment
        trajs = batch_trajs[i]
        L_traj = traj_lens[i]
        points = torch.cat([trajs[i, :L_traj[i]] for i in range(48)], dim=0)
        min_point = torch.min(points, dim=0, keepdim=True).values
        max_point = torch.max(points, dim=0, keepdim=True).values
        point_range = max_point - min_point

        segs = batch_segs[i]  # (N_segs, N_interp, 2)
        segs = segs[torch.all(segs.flatten(1) != 0, dim=1)]

        segs = (segs - min_point.view(1, 1, 2)) / point_range.view(1, 1, 2)

        segs[..., 0] *= img_W
        segs[..., 1] *= img_H

        heatmap = np.zeros((1, img_H, img_W), dtype=np.float32)

        lons = segs[:, :, 0].cpu().numpy().astype(np.int32)
        lats = segs[:, :, 1].cpu().numpy().astype(np.int32)

        for j in range(len(segs)):
            # Draw the polyline
            cv2.polylines(heatmap[0], [np.stack([lons[j], lats[j]], axis=1)], False,
                          1.0, thickness=line_width)

        heatmaps.append(torch.tensor(heatmap))

    return torch.stack(heatmaps, dim=0).to(DEVICE)


def pred_func(noisy_contents: List[Tensor], t: Tensor, model: torch.nn.Module, trajs: Tensor):
    pred = model(*noisy_contents, trajs, t)
    return [pred]


def test(
        T,
        beta_min,
        beta_max,
        data_path,
        model_path,
        report_to,
        deduplicator_path = None,
):
    B = 50
    dataset = RoadNetworkDataset(folder_path=data_path,
                                 batch_size=B,
                                 drop_last=True,
                                 set_name="test",
                                 permute_seq=False,
                                 enable_aug=False,
                                 shuffle=False,
                                 img_H=256,
                                 img_W=256,
                                 need_heatmap=True
                                 )

    DiT = TWDiT_MHSA(D_in=dataset.N_interp * 2,
                N_routes=dataset.N_trajs,
                L_route=dataset.max_L_route,
                L_traj=dataset.max_L_traj,
                d_context=2,
                n_layers=8,
                T=T).to(DEVICE)
    #loadModels("Runs/RoutesDiT/241129_0108_300M/last.pth", TRDiT=TRDiT)
    loadModels(model_path, DiT=DiT)

    if deduplicator_path is None:
        heuristic_cluster = True
    else:
        heuristic_cluster = False
        deduplicator = Deduplicator(dataset.N_interp * 2, 0.1).to(DEVICE)
        loadModels(deduplicator_path, deduplicator=deduplicator)
        deduplicator.eval()

    ddim = DDIM(beta_min, beta_max, T, DEVICE, "quadratic", skip_step=20, data_dim=3)

    titles = ["hungarian_mae", "hungarian_mse", "chamfer_mae", "chamfer_mse", "diff_seg_count", "diff_seg_len"]

    if heuristic_cluster:
        name = "GraphWalker_NoVAE_heuristic"
    else:
        name = "GraphWalker_NoVAE"

    # Maybe add another module named "HeuristicDuplicationRemoval"
    # Implements a heuristic algo

    with open(f"{report_to}/Report_{name}.csv", "w") as f:
        f.write(",".join(titles) + "\n")
        for bi, batch in enumerate(tqdm(dataset, desc="Testing")):

            with torch.no_grad():
                noise = torch.randn_like(batch["routes"].flatten(3))
                # latent, _ = vae.encode(batch["routes"])
                # latent_noise = torch.randn_like(latent)
                routes_pred = ddim.diffusionBackward([noise], pred_func, mode="eps", model=DiT, trajs=batch["trajs"])[0]
                if heuristic_cluster:
                    duplicate_segs, unique_seqs = heuristicDeduplication(routes_pred.flatten(1, 2), 0.1)
                else:
                    duplicate_segs, unique_seqs = deduplicator(routes_pred.flatten(1, 2))

            # norm_pred_segs = duplicate_segs  # (1, N_segs, N_interp, 2)
            # max_point = torch.max(norm_pred_segs.view(-1, 2), dim=0).values.view(1, 1, 2)
            # min_point = torch.min(norm_pred_segs.view(-1, 2), dim=0).values.view(1, 1, 2)
            # point_range = max_point - min_point

            # pred_heatmaps = segsToHeatmaps(coi_means, batch["trajs"], batch["L_traj"], 256, 256, 3)

            batch_scores = reportAllMetrics(unique_seqs, [batch["segs"][b][:batch["N_segs"][b]] for b in range(B)])

            batch_scores = np.array(batch_scores).T

            for scores in batch_scores:
                f.write(",".join([f"{s}" for s in scores]) + "\n")
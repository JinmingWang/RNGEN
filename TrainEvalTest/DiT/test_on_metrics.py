from TrainEvalTest.DiT.configs import *
from TrainEvalTest.GlobalConfigs import *
from TrainEvalTest.Utils import *
from TrainEvalTest.Metrics import *
from Diffusion import DDIM
from tqdm import tqdm
import cv2

import torch

import os

from Dataset import DEVICE, RoadNetworkDataset
from Models import CrossDomainVAE, RoutesDiT


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

        for j in range(len(segs)):
            lons = segs[j, :, 0].cpu().numpy().astype(np.int32)
            lats = segs[j, :, 1].cpu().numpy().astype(np.int32)
            # Draw the polyline
            cv2.polylines(heatmap[0], [np.stack([lons, lats], axis=1)], False,
                          1.0, thickness=line_width)

        heatmaps.append(torch.tensor(heatmap))

    return torch.stack(heatmaps, dim=0).to(DEVICE)


def test():
    dataset = RoadNetworkDataset("Dataset/Tokyo_10k_sparse",
                                 batch_size=100,
                                 drop_last=True,
                                 set_name="test",
                                 permute_seq=False,
                                 enable_aug=False,
                                 img_H=256,
                                 img_W=256
                                 )

    vae = CrossDomainVAE(N_routes=dataset.N_trajs, L_route=dataset.max_L_route,
                         N_interp=dataset.N_interp, threshold=0.5).to(DEVICE)
    loadModels("Runs/CDVAE/241125_0625_sparse/last.pth", vae=vae)
    vae.eval()

    DiT = RoutesDiT(D_in=dataset.N_interp * 2,
                    N_routes=dataset.N_trajs,
                    L_route=dataset.max_L_route,
                    L_traj=dataset.max_L_traj,
                    d_context=2,
                    n_layers=6,
                    T=T).to(DEVICE)
    loadModels("Runs/PathsDiT/241125_2244_Sparse/last.pth", PathsDiT=DiT)
    DiT.eval()

    ddim = DDIM(BETA_MIN, BETA_MAX, T, DEVICE, "quadratic", skip_step=10, data_dim=3)

    titles = ["heatmap_accuracy", "heatmap_precision", "heatmap_recall", "heatmap_f1",
                "hungarian_mae", "hungarian_mse", "chamfer_mae", "chamfer_mse"]

    name = "CDVAE+GLDiT"

    def pred_func(noisy_contents: List[Tensor], t: Tensor):
        pred = DiT(*noisy_contents, batch["trajs"], t)
        return [pred]

    with open(f"Report_{name}.csv", "w") as f:
        f.write(",".join(titles) + "\n")
        for batch in tqdm(dataset, desc="Testing"):
            batch |= RoadNetworkDataset.getTargetHeatmaps(batch, 256, 256, 3)

            batch_segs = batch["segs"]  # (1, N_segs, N_interp, 2)
            max_point = torch.max(batch_segs.view(-1, 2), dim=0).values.view(1, 1, 2)
            min_point = torch.min(batch_segs.view(-1, 2), dim=0).values.view(1, 1, 2)
            point_range = max_point - min_point
            norm_segs = []
            for b, segs in enumerate(batch_segs):
                norm_segs.append((segs[:batch["N_segs"][b]] - min_point) / point_range)


            with torch.no_grad():
                latent, _ = vae.encode(batch["routes"])
                latent_noise = torch.randn_like(latent)
                latent_pred = ddim.diffusionBackward([latent_noise], pred_func, mode="eps")[0]
                duplicate_segs, cluster_mat, cluster_means, coi_means = vae.decode(latent_pred)

            norm_pred_segs = duplicate_segs  # (1, N_segs, N_interp, 2)
            max_point = torch.max(norm_pred_segs.view(-1, 2), dim=0).values.view(1, 1, 2)
            min_point = torch.min(norm_pred_segs.view(-1, 2), dim=0).values.view(1, 1, 2)
            point_range = max_point - min_point
            for i in range(len(coi_means)):
                coi_means[i] = ((coi_means[i]- min_point) / point_range)

            pred_heatmaps = segsToHeatmaps(norm_pred_segs, batch["trajs"], batch["L_traj"], 256, 256, 1)
            batch_scores = reportAllMetrics(pred_heatmaps, batch["target_heatmaps"], coi_means, norm_segs)

            batch_scores = np.array(batch_scores).T

            for scores in batch_scores:
                f.write(",".join([f"{s}" for s in scores]) + "\n")


if __name__ == "__main__":
    test()
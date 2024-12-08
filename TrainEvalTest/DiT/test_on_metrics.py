from TrainEvalTest.DiT.configs import *
from TrainEvalTest.GlobalConfigs import *
from TrainEvalTest.Utils import *
from TrainEvalTest.Metrics import *
from Diffusion import DDIM
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

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


def test():
    dataset = RoadNetworkDataset("Dataset/Tokyo_10k_sparse",
                                 batch_size=100,
                                 drop_last=True,
                                 set_name="test",
                                 permute_seq=False,
                                 enable_aug=False,
                                 shuffle=False,
                                 img_H=256,
                                 img_W=256,
                                 need_heatmap=True
                                 )

    vae = CrossDomainVAE(N_routes=dataset.N_trajs, L_route=dataset.max_L_route,
                         N_interp=dataset.N_interp, threshold=0.7).to(DEVICE)
    loadModels("Runs/CDVAE/241127_1833_sparse_kl1e-6/last.pth", vae=vae)
    vae.eval()

    DiT = RoutesDiT(D_in=dataset.N_interp * 2,
                    N_routes=dataset.N_trajs,
                    L_route=dataset.max_L_route,
                    L_traj=dataset.max_L_traj,
                    d_context=2,
                    n_layers=8,
                    T=T).to(DEVICE)
    #loadModels("Runs/RoutesDiT/241129_0108_300M/last.pth", DiT=DiT)
    loadModels("Runs/RoutesDiT/241129_2126_295M/last.pth", DiT=DiT)

    # state_dict = torch.load("Runs/PathsDiT/241128_0811_KL1e-6/last.pth")
    # DiT.load_state_dict(state_dict)
    # DiT.eval()

    ddim = DDIM(BETA_MIN, BETA_MAX, T, DEVICE, "quadratic", skip_step=10, data_dim=3)

    titles = ["heatmap_accuracy", "heatmap_precision", "heatmap_recall", "heatmap_f1",
                "hungarian_mae", "hungarian_mse", "chamfer_mae", "chamfer_mse"]

    name = "DiT_295M"

    with open(f"Report_{name}.csv", "w") as f:
        f.write(",".join(titles) + "\n")
        for batch in tqdm(dataset, desc="Testing"):

            # batch_segs = batch["segs"]  # (1, N_segs, N_interp, 2)
            # max_point = torch.max(batch_segs.view(-1, 2), dim=0).values.view(1, 1, 2)
            # min_point = torch.min(batch_segs.view(-1, 2), dim=0).values.view(1, 1, 2)
            # point_range = max_point - min_point
            # norm_segs = []
            # for b, segs in enumerate(batch_segs):
            #     norm_segs.append((segs[:batch["N_segs"][b]] - min_point) / point_range)


            with torch.no_grad():
                latent, _ = vae.encode(batch["routes"])
                latent_noise = torch.randn_like(latent)
                latent_pred = ddim.diffusionBackward([latent_noise], pred_func, mode="eps", model=DiT, trajs=batch["trajs"])[0]
                duplicate_segs, cluster_mat, cluster_means, coi_means = vae.decode(latent_pred)

            # plot_manager = PlotManager(4, 2, 5)
            # plot_manager.plotSegments(batch["routes"][0], 0, 0, "Routes", color="red")
            # plot_manager.plotSegments(batch["segs"][0], 0, 1, "Segs", color="blue")
            # plot_manager.plotSegments(coi_means[0], 0, 2, "Pred Segs", color="green")

            # norm_pred_segs = duplicate_segs  # (1, N_segs, N_interp, 2)
            # max_point = torch.max(norm_pred_segs.view(-1, 2), dim=0).values.view(1, 1, 2)
            # min_point = torch.min(norm_pred_segs.view(-1, 2), dim=0).values.view(1, 1, 2)
            # point_range = max_point - min_point

            pred_heatmaps = segsToHeatmaps(coi_means, batch["trajs"], batch["L_traj"], 256, 256, 3)

            # for i in range(len(coi_means)):
            #     coi_means[i] = ((coi_means[i] - min_point) / point_range)

            batch_scores = reportAllMetrics(pred_heatmaps, batch["target_heatmaps"],
                                            coi_means,
                                            [batch["segs"][b][:batch["N_segs"][b]] for b in range(100)])

            # plot_manager.plotSegments(duplicate_segs[0], 0, 3, "Pred Duplicate Segs")
            # plot_manager.plotTrajs(batch["trajs"][0], 0, 4, "Trajectories")
            # plot_manager.plotHeatmap(batch["target_heatmaps"][0], 1, 0, "Target Heatmap")
            # plot_manager.plotHeatmap(pred_heatmaps[0], 1, 1, "Predict Heatmap")
            #
            # plt.savefig("Result.png", dpi=100)

            batch_scores = np.array(batch_scores).T

            for scores in batch_scores:
                f.write(",".join([f"{s}" for s in scores]) + "\n")


if __name__ == "__main__":
    test()
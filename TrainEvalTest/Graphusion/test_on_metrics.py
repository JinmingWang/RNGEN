from TrainEvalTest.Utils import *
from TrainEvalTest.Metrics import *
from Diffusion import DDIM
from tqdm import tqdm
import cv2

import torch

from Dataset import DEVICE, RoadNetworkDataset
from Models import Graphusion, GraphusionVAE


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


def nodesAdjMatToSegs(f_nodes, adj_mat, f_edges, threshold=0.5):
    B = f_nodes.shape[0]
    batch_segs = []
    for b in range(B):
        segs = []
        for r in range(adj_mat.shape[1]):
            for c in range(r + 1, adj_mat.shape[2]):
                if adj_mat[b, r, c] >= threshold:
                    segs.append(f_edges[b, r, c].view(8, 2))
        if len(segs) == 0:
            batch_segs.append(torch.zeros(1, 8, 2, dtype=torch.float32, device=DEVICE))
        else:
            batch_segs.append(torch.stack(segs, dim=0))     # (N_segs, 8, 2)
    return batch_segs



def pred_func(noisy_contents: List[Tensor], t: Tensor, model: torch.nn.Module, trajs: Tensor):
    pred = model(*noisy_contents, trajs, t)
    return [pred]


def test(
        T,
        beta_min,
        beta_max,
        dataset_path,
        vae_path,
        model_path,
        report_to
):
    B = 50
    dataset = RoadNetworkDataset(folder_path=dataset_path,
                                 batch_size=B,
                                 drop_last=True,
                                 set_name="test",
                                 permute_seq=False,
                                 enable_aug=False,
                                 img_H=256,
                                 img_W=256,
                                 need_heatmap=True,
                                 need_nodes=True
                                 )

    # Models
    vae = GraphusionVAE(d_node=2, d_edge=16, d_latent=128, d_hidden=256, n_layers=8, n_heads=8).to(DEVICE)
    loadModels(vae_path, vae=vae)
    vae.eval()

    graphusion = Graphusion(D_in=128,
                            L_enc=dataset.max_N_nodes,
                            N_trajs=dataset.N_trajs,
                            L_traj=dataset.max_L_traj,
                            d_context=2,
                            n_layers=6,
                            T=T).to(DEVICE)
    loadModels(model_path, graphusion=graphusion)
    graphusion.eval()

    ddim = DDIM(beta_min, beta_max, T, DEVICE, "quadratic", skip_step=10, data_dim=2)

    titles = ["hungarian_mae", "hungarian_mse", "chamfer_mae", "chamfer_mse", "diff_seg_count", "diff_seg_len"]

    name = "Graphusion"

    with open(f"{report_to}/Report_{name}.csv", "w") as f:
        f.write(",".join(titles) + "\n")
        for batch in tqdm(dataset, desc="Testing"):

            with torch.no_grad():
                latent, _ = vae.encode(batch["nodes"], batch["edges"], batch["adj_mat"])
                latent_noise = torch.randn_like(latent)
                latent_pred = ddim.diffusionBackward([latent_noise], pred_func, mode="eps",
                                                     model=graphusion, trajs=batch["trajs"])[0]
                f_nodes, f_edges, pred_adj_mat, pred_degrees = vae.decode(latent_pred)

            pred_segs = nodesAdjMatToSegs(f_nodes, pred_adj_mat, f_edges)

            # pred_heatmaps = segsToHeatmaps(pred_segs, batch["trajs"], batch["L_traj"], 256, 256, 3)

            batch_scores = reportAllMetrics(pred_segs,
                                            [batch["segs"][b][:batch["N_segs"][b]] for b in range(B)])

            batch_scores = np.array(batch_scores).T

            for scores in batch_scores:
                f.write(",".join([f"{s}" for s in scores]) + "\n")

            # plot_manager = PlotManager(4, 2, 5)
            # plot_manager.plotSegments(batch["routes"][0], 0, 0, "Routes", color="red")
            # plot_manager.plotSegments(batch["segs"][0], 0, 1, "Segs", color="blue")
            # plot_manager.plotSegments(pred_segs[0], 0, 3, "Pred Segs")
            # plot_manager.plotTrajs(batch["trajs"][0], 0, 4, "Trajectories")
            # plot_manager.plotHeatmap(batch["target_heatmaps"][0], 1, 0, "Target Heatmap")
            # # plot_manager.plotHeatmap(pred_heatmaps[0], 1, 1, "Predict Heatmap")
            # plt.savefig("Graphusion_visualize.png", dpi=100)


if __name__ == "__main__":
    test()
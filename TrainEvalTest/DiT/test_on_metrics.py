from TrainEvalTest.DiT.configs import *
from TrainEvalTest.GlobalConfigs import *
from TrainEvalTest.Utils import *
from TrainEvalTest.Metrics import *
from Diffusion import DDIM
from tqdm import tqdm

import torch

import os

from Dataset import DEVICE, RoadNetworkDataset
from Models import CrossDomainVAE, RoutesDiT





def test():
    dataset = RoadNetworkDataset("Dataset/Tokyo_10k",
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
    loadModels("Runs/CDVAE/241121_1618_final/last.pth", vae=vae)
    vae.eval()

    DiT = RoutesDiT(D_in=dataset.N_interp * 2,
                    N_routes=dataset.N_trajs,
                    L_route=dataset.max_L_route,
                    L_traj=dataset.max_L_traj,
                    d_context=2,
                    n_layers=6,
                    T=T).to(DEVICE)
    loadModels("Runs/PathsDiT/241122_1428_DiT_Tokyo10k/last.pth", DiT=DiT)
    DiT.eval()

    ddim = DDIM(BETA_MIN, BETA_MAX, T, DEVICE, "quadratic", skip_step=1, data_dim=3)

    titles = ["name", "heatmap_accuracy", "heatmap_precision", "heatmap_recall", "heatmap_f1",
                "hungarian_mae", "hungarian_mse", "chamfer_mae", "chamfer_mse"]

    if not os.path.exists("Report.csv"):
        with open("Report.csv", "w") as f:
            f.write(",".join(titles) + "\n")

    name = "CDVAE+GLDiT"

    def pred_func(noisy_contents: List[Tensor], t: Tensor):
        pred = DiT(*noisy_contents, batch["trajs"], t)
        return [pred]

    with open("Report.csv", "a") as f:
        for batch in tqdm(dataset, desc="Testing"):
            batch |= RoadNetworkDataset.getTargetHeatmaps(batch, 256, 256, 1)

            norm_segs = batch["segs"]  # (1, N_segs, N_interp, 2)
            max_point = torch.max(norm_segs.view(-1, 2), dim=0).values.view(1, 1, 1, 2)
            min_point = torch.min(norm_segs.view(-1, 2), dim=0).values.view(1, 1, 1, 2)
            point_range = max_point - min_point
            norm_segs = ((norm_segs - min_point) / point_range)


            with torch.no_grad():
                latent, _ = vae.encode(batch["routes"])
                latent_noise = torch.randn_like(latent)
                latent_pred = ddim.diffusionBackward([latent_noise], pred_func, mode="eps")[0]
                duplicate_segs, cluster_mat, cluster_means, coi_means = vae.decode(latent_pred)

            norm_pred_segs = duplicate_segs  # (1, N_segs, N_interp, 2)
            max_point = torch.max(norm_pred_segs.view(-1, 2), dim=0).values.view(1, 1, 1, 2)
            min_point = torch.min(norm_pred_segs.view(-1, 2), dim=0).values.view(1, 1, 1, 2)
            point_range = max_point - min_point
            for i in range(len(coi_means)):
                coi_means[i] = ((coi_means[i]- min_point) / point_range)

            # TODO: Generate heatmap from seqs
            pred_heatmaps = None
            batch_scores = reportAllMetrics(pred_heatmaps, batch["target_heatmaps"], coi_means, norm_segs)

            batch_scores = np.array(batch_scores).T

            for scores in batch_scores:
                f.write(name + ",")
                f.write(",".join([f"{s}" for s in scores]) + "\n")


if __name__ == "__main__":
    test()
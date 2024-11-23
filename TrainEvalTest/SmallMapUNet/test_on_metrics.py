from TrainEvalTest.Utils import *
from TrainEvalTest.Metrics import *
from tqdm import tqdm

import torch

import os

from Dataset import DEVICE, RoadNetworkDataset
from Models import UNet2D

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

    stage_1 = UNet2D(n_repeats=2, expansion=2).to(DEVICE)
    stage_2 = UNet2D(n_repeats=2, expansion=2).to(DEVICE)

    loadModels("Runs/SmallMapUNet/241123_0238_initial/last.pth", stage_1=stage_1, stage_2=stage_2)

    stage_1.eval()
    stage_2.eval()

    titles = ["name", "heatmap_accuracy", "heatmap_precision", "heatmap_recall", "heatmap_f1",
                "hungarian_mae", "hungarian_mse", "chamfer_mae", "chamfer_mse"]

    if not os.path.exists("Report.csv"):
        with open("Report.csv", "w") as f:
            f.write(",".join(titles) + "\n")

    name = "SmallMap"

    with open("Report.csv", "a") as f:
        for batch in tqdm(dataset, desc="Testing"):
            batch |= RoadNetworkDataset.getTargetHeatmaps(batch, 256, 256, 1)

            with torch.no_grad():
                pred_1 = stage_1(batch["heatmap"])
                pred_2 = stage_2(pred_1)

            norm_segs = batch["segs"]  # (1, N_segs, N_interp, 2)
            max_point = torch.max(norm_segs.view(-1, 2), dim=0).values.view(1, 1, 1, 2)
            min_point = torch.min(norm_segs.view(-1, 2), dim=0).values.view(1, 1, 1, 2)
            point_range = max_point - min_point
            norm_segs = ((norm_segs - min_point) / point_range)

            pred_segs = heatmapsToSegments(pred_2)
            if pred_segs is None:
                batch_scores = reportAllMetrics(pred_2, batch["target_heatmaps"], torch.zeros_like(norm_segs), norm_segs)
            else:
                batch_scores = reportAllMetrics(pred_2, batch["target_heatmaps"], pred_segs, norm_segs)

            batch_scores = np.array(batch_scores).T

            for scores in batch_scores:
                f.write(name + ",")
                f.write(",".join([f"{s}" for s in scores]) + "\n")


if __name__ == "__main__":
    test()
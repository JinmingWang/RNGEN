from TrainEvalTest.Utils import *
from TrainEvalTest.Metrics import *
from tqdm import tqdm

import torch

import os

from Dataset import DEVICE, RoadNetworkDataset
from Models import AD_Linked_Net

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

    model = AD_Linked_Net(d_in=4, H=256, W=256).to(DEVICE)

    loadModels("Runs/TR2RM/241124_1849_sparse/last.pth", ADLinkedNet=model)

    model.eval()

    titles = ["heatmap_accuracy", "heatmap_precision", "heatmap_recall", "heatmap_f1",
                "hungarian_mae", "hungarian_mse", "chamfer_mae", "chamfer_mse"]

    name = "TR2RM"

    with open(f"Report_{name}.csv", "w") as f:
        f.write(",".join(titles) + "\n")
        for batch in tqdm(dataset, desc="Testing"):
            batch |= RoadNetworkDataset.getTargetHeatmaps(batch, 256, 256)

            with torch.no_grad():
                pred_heatmap = model(torch.cat([batch["heatmap"], batch["image"]], dim=1))

            batch_segs = batch["segs"]  # (1, N_segs, N_interp, 2)
            max_point = torch.max(batch_segs.view(-1, 2), dim=0).values.view(1, 1, 2)
            min_point = torch.min(batch_segs.view(-1, 2), dim=0).values.view(1, 1, 2)
            point_range = max_point - min_point
            norm_segs = []
            for b, segs in enumerate(batch_segs):
                norm_segs.append((segs[:batch["N_segs"][b]] - min_point) / point_range)

            pred_segs = heatmapsToSegments(pred_heatmap)
            batch_scores = reportAllMetrics(pred_heatmap, batch["target_heatmaps"], pred_segs, norm_segs)

            batch_scores = np.array(batch_scores).T

            for scores in batch_scores:
                f.write(",".join([f"{s}" for s in scores]) + "\n")


if __name__ == "__main__":
    test()
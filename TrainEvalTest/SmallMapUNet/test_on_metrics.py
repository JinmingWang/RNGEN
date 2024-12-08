from TrainEvalTest.Utils import *
from TrainEvalTest.Metrics import *
from tqdm import tqdm

import torch

import os

from Dataset import DEVICE, RoadNetworkDataset
from Models import UNet2D, NodeExtractor

def test():
    dataset = RoadNetworkDataset("Dataset/Tokyo_10k_sparse",
                                 batch_size=100,
                                 drop_last=True,
                                 set_name="test",
                                 permute_seq=False,
                                 enable_aug=False,
                                 img_H=256,
                                 img_W=256,
                                 need_heatmap=True,
                                 need_image=True,
                                 )

    stage_1 = UNet2D(n_repeats=2, expansion=2).to(DEVICE)
    stage_2 = UNet2D(n_repeats=2, expansion=2).to(DEVICE)
    node_extractor = NodeExtractor().to(DEVICE)

    loadModels("Runs/SmallMapUNet/241124_1849_sparse/last.pth", stage_1=stage_1, stage_2=stage_2)
    loadModels("Runs/NodeExtractor/241126_2349_initial/last.pth", node_model=node_extractor)

    stage_1.eval()
    stage_2.eval()
    node_extractor.eval()

    titles = ["heatmap_accuracy", "heatmap_precision", "heatmap_recall", "heatmap_f1",
                "hungarian_mae", "hungarian_mse", "chamfer_mae", "chamfer_mse"]

    name = "SmallMap"

    with open(f"Report_{name}.csv", "w") as f:
        f.write(",".join(titles) + "\n")
        for batch in tqdm(dataset, desc="Testing"):

            with torch.no_grad():
                pred_1 = stage_1(batch["heatmap"])
                pred_2 = stage_2(pred_1)
                pred_nodemap = node_extractor(pred_2)


            pred_segs = heatmapsToSegments(pred_2, pred_nodemap)
            batch_scores = reportAllMetrics(pred_2, batch["target_heatmaps"], pred_segs,
                                            [batch["segs"][b][:batch["N_segs"][b]] for b in range(100)])

            batch_scores = np.array(batch_scores).T

            for scores in batch_scores:
                f.write(",".join([f"{s}" for s in scores]) + "\n")


if __name__ == "__main__":
    test()
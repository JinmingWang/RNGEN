from TrainEvalTest.GlobalConfigs import *
from TrainEvalTest.SmallMapUNet.configs import *
from TrainEvalTest.Utils import *
from TrainEvalTest.Metrics import *

import torch

import os

from Dataset import DEVICE, RoadNetworkDataset
from Models import UNet2D

def test():
    dataset = RoadNetworkDataset("Dataset/Tokyo_10k",
                                 batch_size=10,
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

    for batch in dataset:
        batch |= RoadNetworkDataset.getTargetHeatmaps(batch, 256, 256, 1)

        with torch.no_grad():
            pred_1 = stage_1(batch["heatmap"])
            pred_2 = stage_2(pred_1)

        pred_segs = heatmapsToSegments(pred_2)

        (heatmap_accuracy, heatmap_precision, heatmap_recall, heatmap_f1,
         hungarian_mae, hungarian_mse, chamfer_mae, chamfer_mse) = reportAllMetrics(pred_2,
                                                                                    batch["target_heatmaps"],
                                                                                    pred_segs,
                                                                                    batch["segs"])

        break


if __name__ == "__main__":
    test()
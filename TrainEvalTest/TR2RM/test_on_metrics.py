from TrainEvalTest.Utils import *
from TrainEvalTest.Metrics import *
from tqdm import tqdm

import torch

import os

from Dataset import DEVICE, RoadNetworkDataset
from Models import AD_Linked_Net, NodeExtractor

def test(
        dataset_path: str = "Dataset/Tokyo",
        model_path: str = "Runs/TR2RM/241124_1849_sparse/last.pth",
        node_extractor_path: str = "Runs/NodeExtractor/241126_2349_initial/last.pth",
):
    dataset = RoadNetworkDataset(folder_path=dataset_path,
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

    model = AD_Linked_Net(d_in=4, H=256, W=256).to(DEVICE)

    node_extractor = NodeExtractor().to(DEVICE)

    loadModels(model_path, ADLinkedNet=model)
    loadModels(node_extractor_path, node_model=node_extractor)

    model.eval()
    node_extractor.eval()

    titles = ["hungarian_mae", "hungarian_mse", "chamfer_mae", "chamfer_mse", "diff_seg_count", "diff_seg_len"]

    name = "TR2RM"

    with open(f"Report_{name}.csv", "w") as f:
        f.write(",".join(titles) + "\n")
        for batch in tqdm(dataset, desc="Testing"):

            with torch.no_grad():
                pred_heatmap = model(torch.cat([batch["heatmap"], batch["image"]], dim=1))
                pred_nodemap = node_extractor(pred_heatmap)

            # batch_segs = batch["segs"]  # (1, N_segs, N_interp, 2)
            # max_point = torch.max(batch_segs.view(-1, 2), dim=0).values.view(1, 1, 2)
            # min_point = torch.min(batch_segs.view(-1, 2), dim=0).values.view(1, 1, 2)
            # point_range = max_point - min_point
            # norm_segs = []
            # for b, segs in enumerate(batch_segs):
            #     norm_segs.append((segs[:batch["N_segs"][b]] - min_point) / point_range)

            pred_segs = heatmapsToSegments(pred_heatmap, pred_nodemap)
            batch_scores = reportAllMetrics(pred_segs,
                                            [batch["segs"][b][:batch["N_segs"][b]] for b in range(100)])

            batch_scores = np.array(batch_scores).T

            for scores in batch_scores:
                f.write(",".join([f"{s}" for s in scores]) + "\n")


if __name__ == "__main__":
    test()
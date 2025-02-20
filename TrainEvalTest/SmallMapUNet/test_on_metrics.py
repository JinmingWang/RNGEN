from TrainEvalTest.Utils import *
from TrainEvalTest.Metrics import *
from tqdm import tqdm

import torch

import os

from Dataset import DEVICE, RoadNetworkDataset
from Models import UNet2D, NodeExtractor

def test(
        dataset_path: str,
        model_path: str,
        node_extractor_path: str,
        report_to: str
):
    B = 1
    dataset = RoadNetworkDataset(folder_path=dataset_path,
                                 batch_size=B,
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
    # node_extractor = NodeExtractor().to(DEVICE)

    loadModels(model_path, stage_1=stage_1, stage_2=stage_2)
    # loadModels(node_extractor_path, node_model=node_extractor)

    stage_1.eval()
    stage_2.eval()
    # node_extractor.eval()

    titles = ["hungarian_mae", "hungarian_mse", "chamfer_mae", "chamfer_mse", "diff_seg_count", "diff_seg_len"]

    name = "SmallMap"

    with open(f"{report_to}/Report_{name}.csv", "w") as f:
        f.write(",".join(titles) + "\n")
        for bi, batch in enumerate(tqdm(dataset, desc="Testing")):

            with torch.no_grad():
                pred_1 = stage_1(batch["heatmap"])
                pred_2 = stage_2(pred_1)
                # pred_nodemap = node_extractor(pred_2)

            pred_segs, temp_map = heatmapsToSegments(pred_2, None)
            batch_scores = reportAllMetrics(pred_segs,
                                            [batch["segs"][b][:batch["N_segs"][b]] for b in range(B)])

            batch_scores = np.array(batch_scores).T

            for scores in batch_scores:
                f.write(",".join([f"{s}" for s in scores]) + "\n")

            plot_manager = PlotManager(4, 1, 1)
            plot_manager.plotHeatmap(pred_2[-1], 0, 0, "pred_heatmap")
            plt.tight_layout()
            plt.savefig(f"./reports/special/{bi}_pred_heatmap.png", dpi=300)

            plot_manager = PlotManager(4, 1, 1)
            plot_manager.plotSegments(pred_segs[-1], 0, 0, "pred_segs")
            plt.tight_layout()
            plt.savefig(f"./reports/special/{bi}_pred_segs.png", dpi=300)



            # plot_manager.plotHeatmap(batch["heatmap"][-1], 0, 1, "input_heatmap")
            # plot_manager.plotSegments(pred_segs[-1], 0, 3, "pred_segs")
            # plot_manager.plotHeatmap(torch.tensor(temp_map), 0, 4, "temp_map")
            # plot_manager.plotSegments(batch["segs"][-1], 1, 0, "target_segs")
            # plot_manager.save("Smallmap_visualize.png")


if __name__ == "__main__":
    test()
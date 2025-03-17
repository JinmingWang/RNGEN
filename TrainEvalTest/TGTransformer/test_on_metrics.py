from TrainEvalTest.Utils import *
from TrainEvalTest.Metrics import *
from tqdm import tqdm

import torch

from Dataset import DEVICE, RoadNetworkDataset
from Models import TGTransformer

def test(
        dataset_path: str,
        model_path: str,
        report_to: str
):
    B = 10
    dataset = RoadNetworkDataset(folder_path=dataset_path,
                                 batch_size=B,
                                 drop_last=True,
                                 set_name="test",
                                 permute_seq=False,
                                 enable_aug=False,
                                 img_H=16,
                                 img_W=16,
                                 need_heatmap=True,
                                 need_image=True,
                                 more_noise_std=0.1
                                 )

    transformer = TGTransformer(
        N_routes=dataset.N_trajs,
        L_route=dataset.max_L_route,
        N_interp=dataset.N_interp,
        threshold=0.2
    ).to(DEVICE)

    loadModels(model_path, transformer=transformer)

    transformer.eval()

    titles = ["hungarian_mae", "hungarian_mse", "chamfer_mae", "chamfer_mse", "diff_seg_count", "diff_seg_len"]

    name = "TGTransformer"

    with open(f"{report_to}/Report_{name}.csv", "w") as f:
        f.write(",".join(titles) + "\n")
        for batch in tqdm(dataset, desc="Testing"):

            with torch.no_grad():
                duplicate_segs, cluster_mat, cluster_means, coi_means = transformer(batch["trajs"])

            batch_scores = reportAllMetrics(coi_means,
                                            [batch["segs"][b][:batch["N_segs"][b]] for b in range(B)])

            batch_scores = np.array(batch_scores).T

            for scores in batch_scores:
                f.write(",".join([f"{s}" for s in scores]) + "\n")


if __name__ == "__main__":
    test()
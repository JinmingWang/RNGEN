from Dataset import RoadNetworkDataset
# Accuracy, Precision, Recall, F1
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.optimize import linear_sum_assignment
import torch
import os
import numpy as np
from jaxtyping import Float32 as F32
from typing import Tuple, List

Tensor = torch.Tensor

def heatmapMetric(pred_heatmap: F32[Tensor, "B 1 H W"],
                     target_heatmap: F32[Tensor, "B 1 H W"],
                     threshold: float = 0.5) -> Tuple[float, ...]:
    """
    Compute the scores between road network projected to 2D (heatmaps)

    :param pred_heatmap: the predicted road network heatmap (after sigmoid)
    :param target_heatmap: the target road network heatmap (binary)
    :param threshold: threshold for binarization
    :return: accuracy, precision, recall, f1
    """
    B, N, _ = pred_heatmap.shape
    pred_flatten = pred_heatmap.view(B, -1).cpu().numpy() > threshold
    target_flatten = target_heatmap.view(B, -1).cpu().numpy()

    accuracy = accuracy_score(target_flatten, pred_flatten)
    precision = precision_score(target_flatten, pred_flatten)
    recall = recall_score(target_flatten, pred_flatten)
    f1 = f1_score(target_flatten, pred_flatten)

    return accuracy, precision, recall, f1


def hungarianMetric(batch_pred_segs: List[F32[Tensor, "P N_interp 2"]],
                    batch_target_segs: List[F32[Tensor, "Q N_interp 2"]]) -> Tuple[float, float]:
    """
    Find hungarian matching between predicted and target segments.
    Then compute MAE and MSE between matched segments.

    Hungarian matching is used to find global optimal matching between predicted and target segments.

    :param batch_pred_segs: list of predicted segments
    :param batch_target_segs: list of target segments
    :return: MAE, MSE
    """

    B = len(batch_pred_segs)
    mae = 0
    mse = 0

    for b in range(B):
        pred_segs = batch_pred_segs[b].flatten(1)   # (P, N_interp*2)
        target_segs = batch_target_segs[b].flatten(1)   # (Q, N_interp*2)
        cost_matrix = torch.cdist(pred_segs, target_segs, p=2).cpu().detach().numpy()  # (P, Q)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Matched segments, compute MAE and MSE normally for matched segments
        matched_pred_segs = pred_segs[row_ind]
        matched_target_segs = target_segs[col_ind]

        # Unmatched segments, The match target will be 0 if the segment is unmatched
        unmatched_pred_segs = np.delete(pred_segs, row_ind, axis=0)
        unmatched_target_segs = np.delete(target_segs, col_ind, axis=0)

        # Compute MAE and MSE
        mae += (np.abs(matched_pred_segs - matched_target_segs).mean() +
                np.abs(unmatched_pred_segs).mean() +
                np.abs(unmatched_target_segs).mean())

        mse += ((np.abs(matched_pred_segs - matched_target_segs) ** 2).mean() +
                (np.abs(unmatched_pred_segs) ** 2).mean() +
                (np.abs(unmatched_target_segs) ** 2).mean())

    return mae / B, mse / B


def chamferMetric(batch_pred_segs: List[F32[Tensor, "P N_interp 2"]],
                  batch_target_segs: List[F32[Tensor, "Q N_interp 2"]]) -> Tuple[float, float]:
    """
    Find chamfer matching between predicted and target segments.
    Then compute MAE and MSE between matched segments.

    Chamfer matching is to find the best match for each segment in the predicted segments.

    :param batch_pred_segs: list of predicted segments
    :param batch_target_segs: list of target segments
    :return: Chamfer distance
    """
    B = len(batch_pred_segs)
    mae = 0
    mse = 0

    for b in range(B):
        pred_segs = batch_pred_segs[b].flatten(1)  # (P, N_interp*2)
        target_segs = batch_target_segs[b].flatten(1)  # (Q, N_interp*2)

        # Compute pairwise distance matrix
        cost_matrix = torch.cdist(pred_segs, target_segs, p=2).cpu().detach().numpy()  # (P, Q)

        # Find the best match for each segment in the predicted segments
        row_ind = np.argmin(cost_matrix, axis=1)

        # Compute MAE and MSE (match the predicted segments to the target segments)
        mae_p2t = (np.abs(pred_segs - target_segs[row_ind]).mean())
        mse_p2t = ((np.abs(pred_segs - target_segs[row_ind]) ** 2).mean())

        # Compute MAE and MSE (match the target segments to the predicted segments)
        row_ind = np.argmin(cost_matrix, axis=0)
        mae_t2p = (np.abs(target_segs - pred_segs[row_ind]).mean())
        mse_t2p = ((np.abs(target_segs - pred_segs[row_ind]) ** 2).mean())

        # Why do we compute p2t and also t2p?
        # When we match p to t, some segments in p may not have a match in t, they are not counted in the loss.
        # When we match t to p, some segments in t may not have a match in p, they are not counted in the loss.
        # So we need to compute both to make sure all segments are counted in the loss.
        mae += (mae_p2t + mae_t2p)
        mse += (mse_p2t + mse_t2p)

    return mae / B, mse / B


def reportAllMetrics(pred_heatmap: F32[Tensor, "B 1 H W"],
                     target_heatmap: F32[Tensor, "B 1 H W"],
                     batch_pred_segs: List[F32[Tensor, "P N_interp 2"]],
                     batch_target_segs: List[F32[Tensor, "Q N_interp 2"]],
                     log_path: str = "report.csv") -> Tuple[str, str]:
    """
    Compute all metrics for road network prediction

    :param log_path: path to save the log
    :param pred_heatmap: the predicted road network heatmap (after sigmoid)
    :param target_heatmap: the target road network heatmap (binary)
    :param batch_pred_segs: list of predicted segments
    :param batch_target_segs: list of target segments
    :return: accuracy, precision, recall, f1, MAE, MSE
    """
    heatmap_accuracy, heatmap_precision, heatmap_recall, heatmap_f1 = heatmapMetric(pred_heatmap, target_heatmap)
    hungarian_mae, hungarian_mse = hungarianMetric(batch_pred_segs, batch_target_segs)
    chamfer_mae, chamfer_mse = chamferMetric(batch_pred_segs, batch_target_segs)

    write_title = not os.path.exists(log_path)

    title = "heatmap_accuracy,heatmap_precision,heatmap_recall,heatmap_f1,hungarian_mae,hungarian_mse,chamfer_mae,chamfer_mse\n"
    content = f"{heatmap_accuracy},{heatmap_precision},{heatmap_recall},{heatmap_f1},{hungarian_mae},{hungarian_mse},{chamfer_mae},{chamfer_mse}\n"

    with open(log_path, "a") as f:
        if write_title:
            f.write(title)
        f.write(content)

    return title, content
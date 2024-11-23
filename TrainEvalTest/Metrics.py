from Dataset import RoadNetworkDataset
# Accuracy, Precision, Recall, F1
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy.optimize import linear_sum_assignment
import torch
import os
import numpy as np
from jaxtyping import Float32 as F32
from typing import Tuple, List
from shapely.geometry import LineString
import cv2
import networkx as nx

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


def findAndAddPath(graph, heatmap, src, dst, visualize) -> LineString:
    # Reached another keynode
    search_grid = np.int32(heatmap > 0)
    H, W = search_grid.shape

    # skip if edge already exists
    if graph.has_edge(f"{src[0]}_{src[1]}", f"{dst[0]}_{dst[1]}"):
        return None
    # Try to find a path from (kx, ky) to (x, y) with greedy search
    path = [src]
    current = src
    while current != dst:
        # Take the pixel with the smallest distance to the target
        neighbors = [(current[0] - 1, current[1]), (current[0] + 1, current[1]),
                     (current[0], current[1] - 1), (current[0], current[1] + 1)]
        if len(path) > 1:
            neighbors.remove((path[-2][0], path[-2][1]))
        neighbors = list(filter(lambda p: 0 <= p[0] < W and 0 <= p[1] < H and search_grid[p[1], p[0]] != 0, neighbors))

        # Compute heuristic distance to the target
        # grid_values are the values of the heatmap at the neighbors
        # higher values means more visited
        grid_values = np.array([search_grid[p[1], p[0]] for p in neighbors])
        distance = np.linalg.norm(np.array(neighbors) - np.array([dst]), axis=1) + grid_values

        nearest_id = np.argmin(distance)
        current = neighbors[nearest_id]

        # update the search grid
        search_grid[current[1], current[0]] += 1

        path.append(current)

        if visualize:
            tmp = heatmap.copy()
            tmp[current[1], current[0]] = 255
            cv2.imshow("search_map", tmp)
            cv2.waitKey(1)

    geometry = LineString(path)

    length = geometry.length
    interp_times = np.linspace(0, length, 8)
    geometry = LineString([geometry.interpolate(i) for i in interp_times])

    graph.add_edge(f"{src[0]}_{src[1]}", f"{dst[0]}_{dst[1]}", geometry=geometry)


def heatmapsToSegments(pred_heatmaps: F32[Tensor, "B 1 H W"], visualize: bool = False) -> List[F32[Tensor, "P N_interp 2"]]:
    B, _, H, W = pred_heatmaps.shape

    segs = []

    for i in range(B):
        pred_heatmap = pred_heatmaps[i, 0].cpu().numpy()

        # Corner detection on predicted heatmap
        temp = cv2.dilate(cv2.cornerHarris(pred_heatmap, 2, 3, 0.08), None, iterations=1)
        corner_blobs = (temp > 0.05 * temp.max())

        num_labels, _, _, centroids = cv2.connectedComponentsWithStats(np.uint8(corner_blobs), connectivity=8)
        corner_map = np.zeros_like(np.uint8(corner_blobs))
        keynodes = centroids.astype(np.int32)[1:]
        graph = nx.Graph()
        for (x, y) in keynodes:
            corner_map[y, x] = 255
            graph.add_node(f"{x}_{y}", pos=(x, y))

        # Now extract the edges (1-pixel wide) from the predicted heatmap
        edge_map = (pred_heatmap > 0.5) | corner_blobs
        edge_map = np.uint8(edge_map) * 127

        if visualize:
            cv2.imshow("edge_map", edge_map)
            cv2.waitKey(0)

        temp_map = edge_map.copy()

        # Flood fill from a keynode until it reaches another keynode
        for ki, (kx, ky) in enumerate(keynodes):
            frontier = {(kx, ky)}
            while frontier:
                new_frontier = set()
                for (x, y) in frontier:
                    temp_map[y, x] = 64

                    # If this pixel is close to another keynode, stop, connect this keynode and the reached keynode
                    distances = np.linalg.norm(keynodes - np.array([x, y]), 2, axis=1)  # (num_keynodes,)
                    # set distance to itself to infinity
                    distances[ki] = np.inf
                    nearest_id = np.argmin(distances)
                    if distances[nearest_id] < 7:
                        src = (kx, ky)
                        dst = (keynodes[nearest_id][0], keynodes[nearest_id][1])
                        findAndAddPath(graph, temp_map, src, dst, visualize)
                        continue

                    # Check neighbors
                    if x > 0 and temp_map[y, x - 1] == 127:
                        new_frontier.add((x - 1, y))
                    if x < W - 1 and temp_map[y, x + 1] == 127:
                        new_frontier.add((x + 1, y))
                    if y > 0 and temp_map[y - 1, x] == 127:
                        new_frontier.add((x, y - 1))
                    if y < H - 1 and temp_map[y + 1, x] == 127:
                        new_frontier.add((x, y + 1))
                frontier = new_frontier
                if visualize:
                    cv2.imshow("temp_map", temp_map)
                    cv2.waitKey(1)

        segs.append(torch.tensor([data["geometry"].coords for u, v, data in graph.edges(data=True)], dtype=torch.float32, device=pred_heatmaps.device))

        # Normalize segments to 0-1
        max_point = torch.max(segs[-1].flatten(0, 1), dim=0).values
        min_point = torch.min(segs[-1].flatten(0, 1), dim=0).values
        point_range = max_point - min_point
        segs[-1] = ((segs[-1] - min_point) / point_range)

    return segs


def reportAllMetrics(pred_heatmap: F32[Tensor, "B 1 H W"],
                     target_heatmap: F32[Tensor, "B 1 H W"],
                     batch_pred_segs: List[F32[Tensor, "P N_interp 2"]],
                     batch_target_segs: List[F32[Tensor, "Q N_interp 2"]]) -> Tuple[str, str]:
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

    title = "heatmap_accuracy,heatmap_precision,heatmap_recall,heatmap_f1,hungarian_mae,hungarian_mse,chamfer_mae,chamfer_mse\n"
    content = f"{heatmap_accuracy},{heatmap_precision},{heatmap_recall},{heatmap_f1},{hungarian_mae},{hungarian_mse},{chamfer_mae},{chamfer_mse}\n"

    return title, content
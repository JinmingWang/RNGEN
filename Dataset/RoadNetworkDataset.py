import torch
from .Utils import *
from tqdm import tqdm
import os
import numpy as np
import cv2


class RoadNetworkDataset():
    def __init__(self,
                 folder_path: str,
                 batch_size: int = 32,
                 drop_last: bool = True,
                 set_name: str = "train",
                 permute_seq: bool = True,
                 enable_aug: bool = False,
                 shuffle:bool = True,
                 img_H: int = 256,
                 img_W: int = 256) -> None:
        """
        Initialize the dataset, this class loads data from a cache file
        The cache file is created by using LaDeDatasetCacheGenerator class
        :param path: the path to the cache file
        :param max_trajs: the maximum number of trajectories to use
        :param set_name: the name of the set, can be "train", "test", "debug" or "all"
        """
        print("Loading RoadNetworkDataset")

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.set_name = set_name
        self.permute_seq = permute_seq
        self.enable_aug = enable_aug
        self.shuffle = shuffle
        self.img_H = img_H
        self.img_W = img_W

        dataset = torch.load(os.path.join(folder_path, "dataset.pt"))

        # (N_data, N_trajs, L_traj, 2)
        self.trajs = dataset["trajs"]
        data_count = len(self.trajs)
        if set_name == "train":
            slicing = slice(data_count - 1000)
        elif set_name == "test":
            slicing = slice(data_count - 1000, None)
        elif set_name == "debug":
            slicing = slice(300)
        elif set_name == "all":
            slicing = slice(data_count)

        # Data Loading

        self.trajs = self.trajs[slicing]
        # (N_data, N_trajs, L_route, N_interp, 2)
        self.routes = dataset["routes"][slicing]
        # (N_data, N_segs, N_interp, 2)
        self.segs = dataset["segs"][slicing]
        # (N_data, 3, H, W)
        self.images = dataset["images"][slicing]
        self.images = torch.nn.functional.interpolate(self.images, (img_H, img_W), mode="bilinear")
        # (N_data, 1, H, W)
        self.heatmaps = dataset["heatmaps"][slicing]
        self.heatmaps = torch.nn.functional.interpolate(self.heatmaps, (img_H, img_W), mode="nearest")

        self.L_traj = dataset["traj_lens"]
        self.L_route = dataset["route_lens"]
        self.N_segs = dataset["seg_nums"]

        self.mean_norm = dataset["point_mean"]
        self.std_norm = dataset["point_std"]

        self.bboxes = dataset["bboxes"]

        # Get the data dimensions

        self.N_data, self.N_trajs, self.max_L_traj = self.trajs.shape[:3]
        self.max_L_route = self.routes.shape[2]
        self.max_N_segs, self.N_interp = self.segs.shape[1:3]

        print(str(self))

    def __str__(self):
        return f"RoadNetworkDataset: {self.set_name} set with {self.N_data} samples packed to {len(self)} batches"


    def __repr__(self):
        return self.__str__().replace("\n", ", ")


    def __len__(self) -> int:
        if self.drop_last:
            return self.N_data // self.batch_size
        else:
            if self.N_data % self.batch_size == 0:
                return self.N_data // self.batch_size
            else:
                return self.N_data // self.batch_size + 1


    def augmentation(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Apply data augmentation to the given sample
        :param trajs: (N, 128, 2)
        :param paths: (N, 11, 2)
        :param graph: (G, 2, 2)
        :param heatmap: (2, H, W)
        :return: The augmented sample
        """

        B = batch["trajs"].shape[0]
        point_shift = torch.randn(B, 1, 1, 2).to(DEVICE) * 0.05
        batch["trajs"] += point_shift * (batch["trajs"] != 0)
        batch["routes"] += point_shift.unsqueeze(1) * (batch["routes"] != 0)
        batch["segs"] += point_shift * (batch["segs"] != 0)

        return batch

    def __getitem__(self, idx) -> Dict[str, Tensor]:
        """
        Return the sample at the given index
        :param idx: the index of the sample
        :return: the sample at the given index
        """
        if isinstance(idx, int):
            idx = [idx]

        trajs = self.trajs[idx].to(DEVICE)
        routes = self.routes[idx].to(DEVICE)
        segs = self.segs[idx].to(DEVICE)
        L_traj = self.L_traj[idx].to(DEVICE)
        L_route = self.L_route[idx].to(DEVICE)

        if self.permute_seq:
            # permute the order of the trajectories and routes
            traj_perm = torch.randperm(trajs.shape[1])
            trajs = trajs[:, traj_perm]
            routes = routes[:, traj_perm]
            L_traj = L_traj[:, traj_perm]
            L_route = L_route[:, traj_perm]

            # permute the order of the segments
            segs_perm = torch.randperm(segs.shape[1])
            segs = segs[:, segs_perm]

        batch_data = {
            "trajs": trajs,
            "routes": routes,
            "segs": segs,
            "heatmap": self.heatmaps[idx].to(DEVICE),
            "image": self.images[idx].to(DEVICE),
            "L_traj": L_traj,
            "L_route": L_route,
            "N_segs": self.N_segs[idx].to(DEVICE),
            "mean_point": self.mean_norm[idx].to(DEVICE),
            "std_point": self.std_norm[idx].to(DEVICE),
            "bbox": self.bboxes[idx].to(DEVICE)
        }

        if self.enable_aug:
            return self.augmentation(batch_data)
        return batch_data


    def __iter__(self):
        if self.shuffle:
            shuffled_indices = torch.randperm(self.N_data)
        else:
            shuffled_indices = torch.arange(self.N_data)

        if self.drop_last:
            end = self.N_data - self.N_data % self.batch_size
        else:
            end = self.N_data

        for i in range(0, end, self.batch_size):
            yield self[shuffled_indices[i:i+self.batch_size]]

    @staticmethod
    def sequencesToSegments(seqs: torch.Tensor, L_seg: int) -> torch.Tensor:
        # seqs: (B, N_seqs, L_seq, D_token)
        B, N_seqs, L_seq, D_token = seqs.shape

        result = torch.cat([
            seqs[:, :, :-1].view(B, N_seqs, -1, L_seg - 1, 2),  # (B, N_seqs, N_segs, L_seg-1, 2)
            seqs[:, :, L_seg - 1::L_seg - 1].unsqueeze(3)],  # (B, N_seqs, N_segs, 1, 2)
            dim=-2)

        # (B, N_seqs, N_segs, L_seg, 2)

        return result.flatten(-2, -1)  # (B, N_seqs, N_segs, L_seg * 2)


    @staticmethod
    def SegmentsToNodesAdj(graphs: Tensor, nodes_pad_len: int) -> Dict[str, Tensor]:
        B, N, _, _ = graphs.shape  # B: batch size, N: number of line segments
        nodes_padded = []
        adj_padded = []
        nodes_counts = []

        for i in range(B):
            graph = graphs[i]  # Shape: (N, 2, 2)
            # Reshape to (2 * N, 2) to get all endpoints
            points = graph.view(-1, 2)  # (2N, 2)

            # Extract unique nodes (2D points)
            unique_nodes, inverse_indices = torch.unique(points, dim=0, return_inverse=True)
            num_nodes = unique_nodes.shape[0]

            nodes_counts.append(num_nodes)

            # Create node tensor with an additional channel indicating if it's a valid node or padding
            # Shape: (nodes_pad_len, 3) -> (x, y, is_valid_node)
            node_tensor = torch.zeros((nodes_pad_len, 3))
            node_tensor[:num_nodes, :2] = unique_nodes
            node_tensor[:num_nodes, 2] = 1  # Set is_valid_node to 1 for valid nodes

            nodes_padded.append(node_tensor)

            # Initialize adjacency matrix of size (nodes_pad_len, nodes_pad_len)
            adj_matrix = torch.zeros((nodes_pad_len, nodes_pad_len), dtype=torch.int32)

            # Fill adjacency matrix
            for j in range(N):
                # Get the indices of the two points of the line segment in the unique nodes list
                p1_idx = inverse_indices[2 * j]  # First point of the line segment
                p2_idx = inverse_indices[2 * j + 1]  # Second point of the line segment
                adj_matrix[p1_idx, p2_idx] = 1
                adj_matrix[p2_idx, p1_idx] = 1  # Undirected graph

            adj_padded.append(adj_matrix)

        # Convert lists to tensors using torch.stack
        nodes_padded_tensor = torch.stack(nodes_padded).to(DEVICE)  # Shape: (B, nodes_pad_len, 3)
        adj_padded_tensor = torch.stack(adj_padded).to(torch.float32).to(DEVICE)  # Shape: (B, nodes_pad_len, nodes_pad_len)
        nodes_count_tensor = torch.tensor(nodes_counts, dtype=torch.long, device=DEVICE)

        return {"nodes": nodes_padded_tensor, "adj_mats": adj_padded_tensor, "n_nodes": nodes_count_tensor}


    @staticmethod
    def getJointsFromSegments(segments) -> Dict[str, Tensor]:
        """
        Computes the adjacency (joint) matrix for a batch of line segments.

        Args:
            segments (torch.Tensor): A tensor of shape (B, N, D), where D=5 (x1, y1, x2, y2, flag).

        Returns:
            torch.Tensor: A joint matrix of shape (B, N, N) where each entry (i, j) is 1 if segments i and j are joint, 0 otherwise.
        """
        B, N, _ = segments.shape

        p1 = segments[:, :, 0:2]    # (B, N, 2)
        p2 = segments[:, :, 2:4]    # (B, N, 2)

        # p1p1_match[i, j] = 1 if p1[i] == p1[j]
        p1p1_match = torch.cdist(p1, p1) < 1e-5   # (B, N, N)
        p1p2_match = torch.cdist(p1, p2) < 1e-5   # (B, N, N)
        p2p1_match = torch.cdist(p2, p1) < 1e-5   # (B, N, N)
        p2p2_match = torch.cdist(p2, p2) < 1e-5   # (B, N, N)

        # Combine the matches
        joint_matrix = p1p1_match | p1p2_match | p2p1_match | p2p2_match

        return {"joints": joint_matrix.to(torch.float32)}

    @staticmethod
    def getTargetHeatmaps(batch,
                          H: int,
                          W: int,
                          line_width: int = 3) -> Dict[str, Float[Tensor, "B 1 H W"]]:
        """
        Compute the target heatmaps for the given segments
        :param segs: the segments tensor
        :param bboxes: (min_lat, max_lat, min_lon, max_lon)
        :param H: the height of the heatmap
        :param W: the width of the heatmap
        :param line_width: the width of the line
        :return: the target heatmaps of shape (B, 1, H, W)
        """

        B = batch["trajs"].shape[0]

        heatmaps = np.zeros((B, 1, H, W), dtype=np.float32)

        for i in range(B):
            # Get the bounding box of the segment
            trajs = batch["trajs"][i]
            L_traj = batch["L_traj"][i]
            points = torch.cat([trajs[j, :L_traj[j]] for j in range(trajs.shape[0])], dim=0)
            min_point = torch.min(points, dim=0, keepdim=True).values
            max_point = torch.max(points, dim=0, keepdim=True).values
            point_range = max_point - min_point

            segs = batch["segs"][i]     # (N_segs, N_interp, 2)
            segs = segs[torch.all(segs.flatten(1) != 0, dim=1)]

            segs = (segs - min_point.view(1, 1, 2)) / point_range.view(1, 1, 2)

            segs[..., 0] *= W
            segs[..., 1] *= H

            for j in range(len(segs)):
                lons = segs[j, :, 0].cpu().numpy().astype(np.int32)
                lats = segs[j, :, 1].cpu().numpy().astype(np.int32)
                # Draw the polyline
                cv2.polylines(heatmaps[i, 0], [np.stack([lons, lats], axis=1)], False,
                              1.0, thickness=line_width)


        return {"target_heatmaps": torch.tensor(heatmaps, dtype=torch.float32, device=DEVICE)}

    @staticmethod
    def heatmapsToSegments(heatmaps: Float[Tensor, "B 1 H W"]):
        pass
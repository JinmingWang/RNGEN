# LaDe Dataset: https://arxiv.org/abs/2306.10675
import torch
from torch.utils.data import Dataset
from .Utils import *
from random import randint


class LaDeCachedDataset(Dataset):
    def __init__(self,
                 folder_path: str,
                 max_trajs: int = 32,
                 set_name: str = "train",
                 no_padding: bool = False) -> None:
        """
        Initialize the dataset, this class loads data from a cache file
        The cache file is created by using LaDeDatasetCacheGenerator class
        :param path: the path to the cache file
        :param max_trajs: the maximum number of trajectories to use
        :param set_name: the name of the set, either "train" or "test"
        """
        self.max_trajs = max_trajs
        self.set_name = set_name
        self.enable_augmentation = set_name == "train"
        self.no_padding = no_padding

        self.trajs = torch.load(folder_path + "/trajs.pth")
        data_count = len(self.trajs)
        slicing = slice(int(data_count * 0.8)) if set_name == "train" else slice(int(data_count * 0.8), None)

        self.trajs = self.trajs[slicing]
        self.paths = torch.load(folder_path + "/paths.pth")[slicing]
        self.graph_tensor = torch.load(folder_path + "/graphs.pth")[slicing]
        self.heatmap = torch.load(folder_path + "/heatmaps.pth")[slicing]
        self.paths_lengths = torch.load(folder_path + "/paths_lengths.pth")[slicing]
        self.trajs_lengths = torch.load(folder_path + "/trajs_lengths.pth")[slicing]
        self.segs_count = torch.load(folder_path + "/segs_count.pth")[slicing]

        path_lens = torch.cat(self.paths_lengths).to(torch.float32)
        traj_lens = torch.cat(self.trajs_lengths).to(torch.float32)
        seg_counts = torch.cat(self.segs_count).to(torch.float32)
        print(f"max_path_len = {torch.max(path_lens).item()}")
        print(f"avg_path_len = {torch.mean(path_lens).item()}")
        print(f"max_traj_len = {torch.max(traj_lens).item()}")
        print(f"avg_traj_len = {torch.mean(traj_lens).item()}")
        print(f"max_segs = {torch.max(seg_counts).item()}")
        print(f"avg_segs = {torch.mean(seg_counts).item()}")

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset
        :return: the number of samples in the dataset
        """
        return len(self.heatmap)


    def augmentation(self, trajs: Tensor, paths: Tensor, graph: Tensor, heatmap: Tensor):
        """
        Apply data augmentation to the given sample
        :param trajs: (N, 128, 2)
        :param paths: (N, 11, 2)
        :param graph: (G, 2, 2)
        :param heatmap: (2, H, W)
        :return: The augmented sample
        """

        match (randint(0, 2)):
            case 0:
                pass
            case 1:     # left-right flip
                # For the coodrinates, flipping is done by negation
                trajs[..., 0] = -trajs[..., 0]
                paths[..., 0] = -paths[..., 0]
                graph[..., 0] = -graph[..., 0]
                # For the heatmap, flipping is done by reversing the tensor
                heatmap = torch.flip(heatmap, dims=(2,))
            case 2:     # top-bottom flip
                trajs[..., 1] = -trajs[..., 1]
                paths[..., 1] = -paths[..., 1]
                graph[..., 1] = -graph[..., 1]
                heatmap = torch.flip(heatmap, dims=(1,))

        # graph data is unordered, so we need to shuffle it
        perm = torch.randperm(graph.shape[0])
        graph = graph[perm]
        # The two nodes of each line segment are unordered, so we need to shuffle them
        ordering = torch.randint(0, 2, (graph.shape[0],))
        first_points = graph[torch.arange(graph.shape[0]), ordering]
        second_points = graph[torch.arange(graph.shape[0]), 1 - ordering]
        graph = torch.stack([first_points, second_points], dim=1)

        # traj and path data are ordered, so we need to shuffle them in the same way
        perm = torch.randperm(trajs.shape[0])
        trajs = trajs[perm]
        paths = paths[perm]

        return trajs, paths, graph, heatmap

    def __getitem__(self, idx: int) -> Tuple[Tensor, ...]:
        """
        Return the sample at the given index
        :param idx: the index of the sample
        :return: the sample at the given index
        """
        trajs = self.trajs[idx][:self.max_trajs].to(DEVICE)     # (N, 128, 2)
        paths = self.paths[idx][:self.max_trajs].to(DEVICE)     # (N, 21, 2)
        graph = self.graph_tensor[idx].to(DEVICE)   # (G, 2, 2)
        heatmap = self.heatmap[idx].to(DEVICE)  # (2, H, W)

        if self.enable_augmentation:
            trajs, paths, graph, heatmap = self.augmentation(trajs, paths, graph, heatmap)
        return (trajs, paths, graph, heatmap)

    @staticmethod
    def collate_fn(batch_list: List[Tuple[Tensor, ...]]) -> Dict[str, Tensor]:
        """
        Collate function for the dataset, this function is used by DataLoader to collate a batch of samples
        :param batch_list: a list of samples
        :return: a batch of samples
        """
        trajs_list, paths_list, graph_list, heatmap_list = zip(*batch_list)

        segments = torch.stack(graph_list).flatten(2)  # (B, G, 2, 2) -> (B, G, 4)
        valid_mask = torch.sum(torch.abs(segments), dim=-1) > 0  # (B, G)
        segments = torch.cat([segments, valid_mask.unsqueeze(-1).float()], dim=-1)  # (B, G, 5)

        return {"trajs": torch.stack(trajs_list),
                "paths": torch.stack(paths_list),
                "segs": segments,
                "heatmaps": torch.stack(heatmap_list)}


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
    def xyxy2xydl(xyxy: Tensor) -> Tensor:
        """
        Convert a segment (x1, y1, x2, y2) to a segment (x_center, y_center, direction, length)
        Why? Because a segment in x1y1x2y2 format can also be represented in x2y2x1y1 format,
        this non-uniqueness is problematic. Imagine a segment (x1, y1, x2, y2), if the model predicts
        (x2, y2, x1, y1) instead, it should be considered correct. Or, if two given segments are (x1, y1, x2, y2) and
        (x2, y2, x1, y1), the attention mechanism should be able to match them as if they are the same segment.
        :param xyxy: The segments (B, N, 5), where 5 is (x1, y1, x2, y2, is_valid)
        :return: The segments in (B, N, 5) format
        """
        # Unbind the input tensor to get x1, y1, x2, y2
        x1, y1, x2, y2, valid_mask = torch.unbind(xyxy, dim=-1)

        # Compute center coordinates
        x_c, y_c = (x1 + x2) / 2, (y1 + y2) / 2

        # Compute the length of each segment
        lengths = torch.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        # Compute the direction and ensure it's in the range [0, pi)
        directions = torch.atan2(y2 - y1, x2 - x1)
        directions = torch.remainder(directions, torch.pi)

        # Return the concatenated result
        return torch.stack([x_c, y_c, directions, lengths, valid_mask], dim=-1)


    @staticmethod
    def xydl2xyxy(xydl: Tensor) -> Tensor:
        """
        Convert a segment (x_center, y_center, direction, length) to a segment (x1, y1, x2, y2)
        :param xydl: The segments (B, N, 5), where 5 is (x_center, y_center, direction, length, is_valid)
        :return: The segments in (B, N, 5) format
        """
        # Unbind the input tensor to get x_c, y_c, directions, lengths
        x_c, y_c, directions, lengths, valid_mask = torch.unbind(xydl, dim=-1)

        # Compute half-length components
        half_dx = (lengths / 2) * torch.cos(directions)
        half_dy = (lengths / 2) * torch.sin(directions)

        # Compute x1, y1, x2, y2 based on the center and direction
        x1, y1 = x_c - half_dx, y_c - half_dy
        x2, y2 = x_c + half_dx, y_c + half_dy

        # Return the concatenated result
        return torch.stack([x1, y1, x2, y2, valid_mask], dim=-1)







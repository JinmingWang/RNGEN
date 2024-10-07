# LaDe Dataset: https://arxiv.org/abs/2306.10675
from torch.utils.data import Dataset
from .Utils import *
from random import randint


class LaDeCachedDataset(Dataset):
    def __init__(self,
                 folder_path: str,
                 max_trajs: int = 32,
                 set_name: str = "train") -> None:
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

        self.trajs = torch.load(folder_path + "/trajs.pth")
        data_count = len(self.trajs)
        slicing = slice(int(data_count * 0.8)) if set_name == "train" else slice(int(data_count * 0.8), None)

        self.trajs = self.trajs[slicing]
        self.paths = torch.load(folder_path + "/paths.pth")[slicing]
        self.graph_tensor = torch.load(folder_path + "/graphs.pth")[slicing]
        self.heatmap = torch.load(folder_path + "/heatmaps.pth")[slicing]


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

        # traj and path data are ordered, so we need to shuffle them in the same way
        perm = torch.randperm(trajs.shape[0])
        trajs = trajs[perm]
        paths = paths[perm]

        return trajs, paths, graph, heatmap

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Return the sample at the given index
        :param idx: the index of the sample
        :return: the sample at the given index
        """
        trajs = self.trajs[idx][:self.max_trajs].to(DEVICE)     # (N, 128, 2)
        paths = self.paths[idx][:self.max_trajs].to(DEVICE)     # (M, 21, 2)
        graph = self.graph_tensor[idx].to(DEVICE)   # (G, 2, 2)
        heatmap = self.heatmap[idx].to(DEVICE)  # (2, H, W)

        if self.enable_augmentation:
            return self.augmentation(trajs, paths, graph, heatmap)
        return (trajs, paths, graph, heatmap)

    @staticmethod
    def collate_fn(batch_list: List[Tuple[Tensor, ...]]) -> Dict[str, Tensor]:
        """
        Collate function for the dataset, this function is used by DataLoader to collate a batch of samples
        :param batch_list: a list of samples
        :return: a batch of samples
        """
        trajs_list, paths_list, graph_list, heatmap_list = zip(*batch_list)
        return {"trajs": torch.stack(trajs_list),
                "paths": torch.stack(paths_list),
                "graphs": torch.stack(graph_list),
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
            node_tensor = torch.zeros((nodes_pad_len, 2))
            node_tensor[:num_nodes] = unique_nodes  # Assign unique nodes

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


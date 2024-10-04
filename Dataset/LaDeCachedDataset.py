# LaDe Dataset: https://arxiv.org/abs/2306.10675
from torch.utils.data import Dataset
from .Utils import *


class LaDeCachedDataset(Dataset):
    def __init__(self, graph_path: str, heatmap_path: str, traj_path: str, path_path: str, max_trajs: int = 32) -> None:
        """
        Initialize the dataset, this class loads data from a cache file
        The cache file is created by using LaDeDatasetCacheGenerator class
        :param path: the path to the cache file
        """
        self.trajs = torch.load(traj_path)
        self.paths = torch.load(path_path)
        self.graph_tensor = torch.load(graph_path)
        self.heatmap = torch.load(heatmap_path)
        self.max_trajs = max_trajs

    def __len__(self) -> int:
        """
        Return the number of samples in the dataset
        :return: the number of samples in the dataset
        """
        return len(self.heatmap)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Return the sample at the given index
        :param idx: the index of the sample
        :return: the sample at the given index
        """
        return (self.trajs[idx][:self.max_trajs].to(DEVICE), self.paths[idx][:self.max_trajs].to(DEVICE),
                self.graph_tensor[idx].to(DEVICE), self.heatmap[idx].to(DEVICE))

    @staticmethod
    def collate_fn(batch_list: List[Tuple[Tensor, Tensor, Tensor, Tensor]]):
        """
        Collate function for the dataset, this function is used by DataLoader to collate a batch of samples
        :param batch_list: a list of samples
        :return: a batch of samples
        """
        trajs_list, paths_list, graph_list, heatmap_list = zip(*batch_list)
        return torch.stack(trajs_list), torch.stack(paths_list), list(graph_list), torch.stack(heatmap_list, dim=0)
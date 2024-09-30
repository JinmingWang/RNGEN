import os.path

from torch.utils.data import Dataset
from .Utils import *
from tqdm import tqdm


class LaDeCachedDataset(Dataset):
    def __init__(self, path: str):
        assert os.path.exists(path), (f"File {path} does not exist, please check it or use "
                                      f"{self.__class__.__name__}.createCache(path) to create cache data")
        self.trajs, self.graph_tensor, self.heatmap = torch.load(path)

    def __len__(self):
        return len(self.heatmap)

    def __getitem__(self, idx: int):
        return self.trajs[idx].to(DEVICE), self.graph_tensor[idx].to(DEVICE), self.heatmap[idx].to(DEVICE)

    @staticmethod
    def collate_fn(batch_list: List[Tuple[Tensor, Tensor, Tensor]]):
        trajs_list, graph_list, heatmap_list = zip(*batch_list)
        return list(trajs_list), list(graph_list), torch.stack(heatmap_list, dim=0)
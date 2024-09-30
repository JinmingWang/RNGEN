import torch
from typing import List, Tuple
import matplotlib.pyplot as plt
import random

Tensor = torch.Tensor

Node = Tensor  # (2,)
Trajectory = Tensor  # (T, 2)

DATASET_ROOT = "/home/jimmy/Data/LaDe"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DEVICE = "cpu"

class Segment():
    def __init__(self, node_1, node_2):
        self.nodes = torch.stack([node_1, node_2])

    def __repr__(self):
        return (f"({self[0, 0].item(), self[0, 1].item()}) --- "
                f"({self[1, 0].item(), self[1, 1].item()})")

    def __getitem__(self, idx):
        return self.nodes[idx]

    def __eq__(self, other: 'Segment'):
        # the nodes are unordered
        return torch.equal(self.nodes, other.nodes) or torch.equal(self.nodes[[1, 0]], other.nodes)

    def __sub__(self, other: Node):
        return self.nodes - other

    def __contains__(self, item: Node):
        return torch.equal(self.nodes[0], item) or torch.equal(self.nodes[1], item)

    def draw(self, **kwargs):
        plt.plot(self.nodes[:, 0].cpu().numpy(), self.nodes[:, 1].cpu().numpy(), **kwargs)

    @classmethod
    def fromTensor(cls, tensor: Tensor):
        return cls(tensor[0], tensor[1])


class SegmentGraph(list):
    def __init__(self, segments: List[Segment] = None):
        if segments is not None:
            super(SegmentGraph, self).__init__(segments)
        else:
            super(SegmentGraph, self).__init__()

    @property
    def num_nodes(self):
        nodes = torch.stack(self).view(-1, 2)
        return torch.unique(nodes, dim=0).size(0)

    def getNeighbors(self, node: Node, exceptions: List[int] = None) -> List[int]:
        exceptions = exceptions if exceptions is not None else []
        return [i for i, segment in enumerate(self) if i not in exceptions and node in segment]

    def getRandomPath(self, start_segment_id: int) -> List[Node]:
        """
        Get a path from the graph growing from the start segment
        :param start_segment_id: the id of the start segment
        :return: a list of nodes in the path
        """
        def growPath(visited_sid: List[int], path: List[Node]) -> List[Node]:
            """
            Recursively grow the path
            :param visited_sid: the list of visited segment ids already in the path
            :param path: the list of nodes in the path
            :return: a list of nodes in the path
            """
            # get the neighbors of the start node
            left_neighbors = self.getNeighbors(path[0], exceptions=visited_sid)
            if len(left_neighbors) > 0:
                # randomly choose a neighbor
                left_neighbor = left_neighbors[random.randint(0, len(left_neighbors) - 1)]
                visited_sid.append(left_neighbor)
                if torch.equal(self[left_neighbor][0], path[0]):
                    path.insert(0, self[left_neighbor][1])
                else:
                    path.insert(0, self[left_neighbor][0])

            # get the neighbors of the end node
            right_neighbors = self.getNeighbors(path[-1], exceptions=visited_sid)
            if len(right_neighbors) > 0:
                right_neighbor = right_neighbors[random.randint(0, len(right_neighbors) - 1)]
                visited_sid.append(right_neighbor)
                if torch.equal(self[right_neighbor][0], path[-1]):
                    path.append(self[right_neighbor][1])
                else:
                    path.append(self[right_neighbor][0])

            if len(left_neighbors) == 0 and len(right_neighbors) == 0:
                return path
            return growPath(visited_sid, path)

        return growPath([start_segment_id], [self[start_segment_id][0], self[start_segment_id][1]])


    def getCenter(self) -> Node:
        nodes = torch.stack([seg.nodes for seg in self]).view(-1, 2)   # (N, 2)
        return torch.mean(nodes, dim=0, keepdim=True)


    def normalize(self) -> 'SegmentGraph':
        """
        The min-max normalization limits the graph from -3 to 3
        :return:
        """
        nodes = torch.stack([seg.nodes for seg in self]).view(-1, 2)  # (N, 2)
        min_vals = torch.min(nodes, dim=0, keepdim=True)[0]
        max_vals = torch.max(nodes, dim=0, keepdim=True)[0]
        val_range = max_vals - min_vals

        for i in range(self.__len__()):
            self[i].nodes = (self[i].nodes - min_vals) / val_range * 6 - 3

        return self


    def transform(self, rotate_angle: Tensor, scale_factor: Tensor) -> 'SegmentGraph':
        center = self.getCenter()

        cos = torch.cos(rotate_angle)
        sin = torch.sin(rotate_angle)
        rot_mat = torch.tensor([[cos, -sin], [sin, cos]], device=DEVICE)

        for i in range(self.__len__()):
            self[i].nodes = torch.mm(self[i].nodes - center, rot_mat) * scale_factor + center

        return self

    def toTensor(self) -> Tensor:
        return torch.stack([segment.nodes for segment in self]).to(DEVICE)

    @classmethod
    def fromTensor(cls, graph_tensor: Tensor) -> 'SegmentGraph':
        result = SegmentGraph()
        for each in graph_tensor:
            result.append(Segment.fromTensor(each))
        return result

    def draw(self, **kwargs):
        for segment in self:
            segment.draw(**kwargs)


def visualizeTraj(traj: Trajectory) -> None:
    """
    Visualize a trajectory
    :param traj: tensor of shape (T, 2)
    :return:
    """
    traj = traj.cpu().numpy()
    plt.scatter(traj[:, 0], traj[:, 1], c="#0000FF", marker='.', s=10, alpha=0.5)
    plt.plot(traj[:, 0], traj[:, 1], c="#000000", alpha=0.1, linewidth=1)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")



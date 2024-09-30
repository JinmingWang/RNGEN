import matplotlib.pyplot as plt
from .Utils import *

def visualizeTraj(traj: Trajectory) -> None:
    """
    Visualize a trajectory
    :param traj: tensor of shape (T, 2)
    :return:
    """
    plt.scatter(traj[:, 0], traj[:, 1], c="#0000FF", marker='.', s=10, alpha=0.5)
    plt.plot(traj[:, 0], traj[:, 1], c="#000000", alpha=0.1, linewidth=1)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")


def visualizeGraph(graph: SegmentGraph, color: str="#000000") -> None:
    """
    Visualize a graph
    :param segments: list of segments, each segment is a list of two tensors of shape (2,)
    :return:
    """
    for segment in graph:
        plt.plot(segment[:, 0], segment[:, 1], c=color, alpha=0.5, linewidth=1, marker='o', markersize=1)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
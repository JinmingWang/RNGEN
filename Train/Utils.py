import matplotlib.pyplot as plt
from Dataset import SegmentGraph, visualizeTraj
import torch

def renderPlotHeatmap(graph, trajs, heatmap, pred_heatmap):
    figure = plt.figure(figsize=(15, 10))
    plt.subplot(2, 3, 1)
    plt.title("Graph")
    seg_graph = SegmentGraph.fromTensor(graph)
    seg_graph.draw(color="#ff0000", linewidth=5, alpha=0.1)

    plt.subplot(2, 3, 2)
    plt.title("Trajetories")
    for traj in trajs:
        traj = traj[torch.all(traj != 0, dim=1)]
        visualizeTraj(traj)

    plt.subplot(2, 3, 4)
    plt.title("Trajectory Heatmap")
    plt.imshow(heatmap[0].cpu().numpy(), origin="lower")
    plt.colorbar()

    plt.subplot(2, 3, 5)
    plt.title("Nodes Heatmap")
    plt.imshow(heatmap[1].cpu().numpy(), origin="lower")
    plt.colorbar()

    plt.subplot(2, 3, 3)
    plt.title("Predicted Heatmap")
    plt.imshow(pred_heatmap[0].detach().cpu().numpy(), origin="lower")
    plt.colorbar()

    plt.subplot(2, 3, 6)
    plt.title("Predicted Nodes Heatmap")
    plt.imshow(pred_heatmap[1].detach().cpu().numpy(), origin="lower")
    plt.colorbar()

    # plt.savefig("visual_0.png", dpi=100)
    return figure


def renderPlotTraj(graph, trajs, reconstruct_trajs):
    figure = plt.figure(figsize=(15, 10))
    plt.subplot(1, 3, 1)
    plt.title("Graph")
    seg_graph = SegmentGraph.fromTensor(graph)
    seg_graph.draw(color="#ff0000", linewidth=5, alpha=0.1)

    plt.subplot(1, 3, 2)
    plt.title("Trajetories")
    for traj in trajs:
        traj = traj[torch.all(traj != 0, dim=1)]
        visualizeTraj(traj)

    plt.subplot(1, 3, 3)
    plt.title("Reconstructed Trajectories")
    for traj in reconstruct_trajs:
        traj = traj[torch.all(traj != 0, dim=1)]
        visualizeTraj(traj)

    # plt.savefig("visual_0.png", dpi=100)
    return figure
from Dataset import *
import matplotlib.pyplot as plt

def test_LaDeDataset():
    dataset = LaDeDataset(30)
    graph = dataset[0]

    aug_graph = dataset.graphAugmentation(graph, rotation=True, scaling_range=0.2)

    graph.draw(color="#000000", alpha=0.5, linewidth=1, marker='o', markersize=1)
    aug_graph.draw(color="#FF0000", alpha=0.5, linewidth=1, marker='o', markersize=1)

    plt.show()

    trajs = dataset.generateTrajsFromGraph(graph, 50.0, 15.0, 10.0)

    for traj in trajs:
        visualizeTraj(traj)
    plt.show()



if __name__ == '__main__':
    test_LaDeDataset()
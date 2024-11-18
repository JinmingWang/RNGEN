from Dataset import *
from TrainEvalTest.Utils import PlotManager
import matplotlib.pyplot as plt
import torch

def test_LaDeDataset():
    dataset = torch.load("Dataset/RoadsGetter/dataset.pt")

    batch = {
        "segs": dataset["segs"][0:5],
        "routes": dataset["routes"][0:5]
    }

    for i in range(1):
        for route in batch["routes"][i].flatten(0, -3):
            plt.plot(route[:, 0], route[:, 1])
    plt.savefig("routes_orig.png", dpi=100)
    plt.show()


    # dataset = RoadNetworkDataset("Dataset/RoadsGetter",
    #                              batch_size=5,
    #                              drop_last=False,
    #                              set_name="debug",
    #                              enable_aug=False,
    #                              img_H=16,
    #                              img_W=16
    #                              )
    # batch = dataset[0:5]

    plot_manager = PlotManager(4, 2, 5)

    for i in range(5):
        plot_manager.plotSegments(batch["segs"][i], 0, i, f"Segs{i}")
        plot_manager.plotSegments(batch["routes"][i], 1, i, f"Routes{i}")

    plt.savefig("routes.png", dpi=100)



if __name__ == '__main__':
    test_LaDeDataset()
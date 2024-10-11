from Dataset import *
import matplotlib.pyplot as plt

def test_LaDeDataset():
    dataset = LaDeCachedDataset("./Dataset/Shanghai_20k", max_trajs=32, set_name="train")

    max_node_count = 0

    for batch in dataset:
        n_nodes = LaDeCachedDataset.SegmentsToNodesAdj(batch[2].unsqueeze(0), 128)["n_nodes"]

        if n_nodes.item() > max_node_count:
            max_node_count = n_nodes.item()
            print(max_node_count)



if __name__ == '__main__':
    test_LaDeDataset()
from Train.train_traj_encoder import train
from Dataset import LaDeDataset, LaDeCachedDataset


def createCache():
    raw_dataset = LaDeDataset(graph_depth=20,
                              min_trajs=16,
                              rotation=True,
                              scaling_range=0.2,
                              traj_step_mean=0.3,
                              traj_step_std=0.1,
                              traj_noise_std=0.1)

    LaDeCachedDataset.createCache(raw_dataset, 500, "Dataset/Shanghai_500_Cache.pth")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # createCache()
    train()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

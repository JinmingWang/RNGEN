from Dataset import *
import matplotlib.pyplot as plt

def test_LaDeDataset():
    dataset = LaDeCachedDataset("./Dataset/Shanghai_20k_Lv1", max_trajs=32, set_name="train")



if __name__ == '__main__':
    test_LaDeDataset()
from TrainEvalTest.RGVAE.train import train as train_cdvae
from TrainEvalTest.TRDiT.train import train as train_DiT
from TrainEvalTest.SmallMapUNet.train import train as train_SmallMap
from TrainEvalTest.TR2RM.train import train as train_TR2RM
from TrainEvalTest.DFDRUNet.train import train as train_DFDRUNet
from TrainEvalTest.GraphusionVAE.train import train as train_GraphusionVAE
from TrainEvalTest.Graphusion.train import train as train_Graphusion
import time

if __name__ == "__main__":
    # Scheduled task
    print("Waiting for start")
    # train_DFDRUNet()
    # train_cdvae()
    # train_DiT()
    # train_TR2RM()
    # train_SmallMap()
    # train_GraphusionVAE()
    train_Graphusion()

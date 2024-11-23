from TrainEvalTest.CrossDomainVAE.train import train as train_cdvae
from TrainEvalTest.DiT.train import train as train_DiT
from TrainEvalTest.SmallMapUNet.train import train as train_SmallMap
from TrainEvalTest.TR2RM.train import train as train_TR2RM
import time

if __name__ == "__main__":
    # Scheduled task
    print("Waiting for start")
    # train_cdvae()
    # train_DiT()
    train_TR2RM()
    train_SmallMap()

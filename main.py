from TrainEvalTest.CrossDomainVAE.train import train as train_cdvae
import time

if __name__ == "__main__":
    # Scheduled task
    print("Waiting for start")
    train_cdvae()

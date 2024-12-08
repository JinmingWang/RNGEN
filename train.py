from TrainEvalTest.RGVAE.train import train as train_rgvae
from TrainEvalTest.TRDiT.train import train as train_TRDiT
from TrainEvalTest.SmallMapUNet.train import train as train_SmallMap
from TrainEvalTest.TR2RM.train import train as train_TR2RM
from TrainEvalTest.DFDRUNet.train import train as train_DFDRUNet
from TrainEvalTest.GraphusionVAE.train import train as train_GraphusionVAE
from TrainEvalTest.Graphusion.train import train as train_Graphusion
from TrainEvalTest.NodeExtractor.train import train as train_NodeExtractor

dataset = "Tokyo_10k_sparse"

default_params = {
    "dataset_path": f"Dataset/{dataset}",
    "lr": 2e-4,
    "lr_reduce_factor": 0.5,
    "lr_reduce_patience": 30,
    "lr_reduce_min": 1e-7,
    "lr_reduce_threshold": 1e-5,
    "epochs": 1000,
    "B": 32,
    "mov_avg_len": 5,
    "log_interval": 10,
}

vae_params = {k: v for k, v in default_params.items()}
vae_params["lr"] = 1e-4
vae_params["kl_weight"] = 1e-6

diffusion_params = {k: v for k, v in default_params.items()}
diffusion_params["T"] = 500
diffusion_params["beta_min"] = 0.0001
diffusion_params["beta_max"] = 0.05
diffusion_params["eval_interval"] = 10


if __name__ == "__main__":
    print("Start Training DFDRUNet")    # ------------------------------------------- DFDRUNet
    train_DFDRUNet(
        title=dataset,
        **default_params,
        load_weights=None
    )

    print("Start Training TR2RM (AD-Linked Net)")   # ------------------------------------------- TR2RM
    train_TR2RM(
        title=dataset,
        **default_params,
        load_weights=None
    )

    print("Start Training SmallMap (UNet)")  # ------------------------------------------- SmallMapUNet
    train_SmallMap(
        title=dataset,
        **default_params,
        load_weights=None
    )

    print("Start Training NodeExtractor")   # ------------------------------------------- NodeExtractor
    train_NodeExtractor(
        title=dataset,
        **default_params,
        load_weights=None
    )

    print("Start Training RGVAE")   # ------------------------------------------- RGVAE
    rgvae_path = train_rgvae(
        title=dataset,
        **vae_params,
        load_weights=None
    )

    print("Start Training TRDiT")   # ------------------------------------------- TRDiT
    train_TRDiT(
        title=dataset,
        **diffusion_params,
        vae_path=rgvae_path,
        load_weights=None
    )

    print("Start Training GraphusionVAE")   # ------------------------------------------- GraphusionVAE
    graphusion_vae_path = train_GraphusionVAE(
        title=dataset,
        **vae_params,
        load_weights=None
    )

    print("Start Training Graphusion")  # ------------------------------------------- Graphusion
    train_Graphusion(
        title=dataset,
        **diffusion_params,
        vae_path=graphusion_vae_path,
        load_weights=None
    )

from TrainEvalTest.WGVAE.train import train as train_WGVAE
from TrainEvalTest.TWDiT.train import train as train_TWDiT
from TrainEvalTest.SmallMapUNet.train import train as train_SmallMap
from TrainEvalTest.TR2RM.train import train as train_TR2RM
from TrainEvalTest.DFDRUNet.train import train as train_DFDRUNet
from TrainEvalTest.GraphusionVAE.train import train as train_GraphusionVAE
from TrainEvalTest.Graphusion.train import train as train_Graphusion
from TrainEvalTest.NodeExtractor.train import train as train_NodeExtractor

from TrainEvalTest.WGVAE_MHSA.train import train as train_WGVAE_MHSA
from TrainEvalTest.TWDiT_MHSA.train import train as train_TWDiT_MHSA
from TrainEvalTest.TWDiT_NoVAE.train import train as train_TWDiT_NoVAE
from TrainEvalTest.TGTransformer.train import train as train_TGTransformer

from TrainEvalTest.WGVAE_MHSA.train_new import train as train_WGVAE_new

dataset = "Chengdu"

default_params = {
    "dataset_path": f"Dataset/{dataset}",
    "lr": 2e-4,
    "lr_reduce_factor": 0.5,
    "lr_reduce_patience": 20,
    "lr_reduce_min": 1e-6,
    "lr_reduce_threshold": 1e-7,
    "epochs": 500,
    "B": 32,
    "mov_avg_len": 5,
    "log_interval": 10,
}

### For fine-tunning
default_params["lr"] = 1e-4

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
        load_weights="Runs/DFDRUNet/250102_1118_Tokyo_Special/last.pth"
    )

    print("Start Training TR2RM (AD-Linked Net)")   # ------------------------------------------- TR2RM
    heatmap_model_path = train_TR2RM(
        title=dataset,
        **default_params,
        load_weights="Runs/TR2RM/250102_1126_Tokyo_Special/last.pth"
    )

    print("Start Training SmallMap (UNet)")  # ------------------------------------------- SmallMapUNet
    train_SmallMap(
        title=dataset,
        **default_params,
        load_weights="Runs/SmallMap/250102_1130_Tokyo_Special/last.pth"
    )

    # print("Start Training WGVAE")   # ------------------------------------------- RGVAE
    # wgvae_path = train_WGVAE(
    #     title=dataset,
    #     **vae_params,
    #     load_weights="Runs/WGVAE/241227_1045_Tokyo/last.pth"
    # )
    #
    # print("Start Training TWDiT")   # ------------------------------------------- TRDiT
    # train_TWDiT(
    #     title=dataset,
    #     **diffusion_params,
    #     vae_path=rgvae_path,
    #     load_weights="Runs/TWDiT/250102_1206_Tokyo_Special/last.pth"
    # )

    print("Start Training GraphusionVAE")   # ------------------------------------------- GraphusionVAE
    graphusion_vae_path = train_GraphusionVAE(
        title=dataset,
        **vae_params,
        load_weights="Runs/GraphusionVAE/250102_1218_Tokyo_Special/last.pth"
    )

    print("Start Training Graphusion")  # ------------------------------------------- Graphusion
    train_Graphusion(
        title=dataset,
        **diffusion_params,
        vae_path=graphusion_vae_path,
        load_weights="Runs/Graphusion/250102_1222_Tokyo_Special/last.pth"
    )

    # print("Start Training WGVAE_MHSA new")   # ------------------------------------------- RGVAE
    # wgvae_path = train_WGVAE_new(
    #     title=dataset + "new",
    #     **vae_params,
    #     load_weights=None
    # )

    # print("Start Training TWDiT_MHSA")   # ------------------------------------------- TRDiT
    # train_TWDiT_MHSA(
    #     title=dataset + "new",
    #     **diffusion_params,
    #     # vae_path=wgvae_path,
    #     vae_path="Runs/WGVAE_NEW/250310_0951_Tokyonew/last.pth",
    #     load_weights=None
    # )

    # print("Start Training TWDiT_NoVAE")  # ------------------------------------------- TRDiT
    # train_TWDiT_NoVAE(
    #     title=dataset,
    #     **diffusion_params,
    #     load_weights="Runs/TWDiT_NoVAE/250225_0108_Tokyo/last.pth"
    # )
    #
    # print("Start Training TGTransformer")
    # train_TGTransformer(
    #     title=dataset,
    #     **vae_params,
    #     load_weights="Runs/TGTransformer/250303_0112_Tokyo/last.pth"
    # )


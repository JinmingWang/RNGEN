from TrainEvalTest.TWDiT.test_on_metrics import test as test_TWDiT
from TrainEvalTest.TR2RM.test_on_metrics import test as test_TR2RM
from TrainEvalTest.DFDRUNet.test_on_metrics import test as test_DFDRUNet
from TrainEvalTest.Graphusion.test_on_metrics import test as test_Graphusion
from TrainEvalTest.SmallMapUNet.test_on_metrics import test as test_SmallMap
from TrainEvalTest.TWDiT_MHSA.test_on_metrics import test as test_TWDiT_MHSA
from TrainEvalTest.TWDiT_NoVAE.test_on_metrics import test as test_TWDiT_NoVAE
from TrainEvalTest.TGTransformer.test_on_metrics import test as test_TGTransformer
import os

weights = {
    "Tokyo": {
        "WGVAE": "Runs/WGVAE/241227_1045_Tokyo/last.pth",
        "TWDiT": "Runs/TWDiT/241228_0200_Tokyo/last.pth",
        "WGVAE_MHSA": "Runs/WGVAE_MHSA/250225_1110_Tokyo/last.pth",
        "TWDiT_MHSA": "Runs/TWDiT_MHSA/250225_2046_Tokyo/last.pth",
        "TWDiT_NoVAE": "Runs/TWDiT_NoVAE/250226_1517_Tokyo/last.pth",
        "Deduplicator": "Runs/Deduplicator/250228_0919_Tokyo/last.pth",
        "WGVAE_new": "Runs/WGVAE_NEW/250310_0951_Tokyonew/last.pth",
        "TGTransformer": "Runs/TGTransformer/250304_0110_Tokyo/last.pth"
    },
    "Shanghai": {
        "WGVAE": "Runs/WGVAE/241218_2207_Shanghai/last.pth",
        "TWDiT": "Runs/TWDiT/241219_0901_Shanghai/last.pth",
    },
    "LasVegas": {
        "WGVAE": "Runs/WGVAE/241223_0427_LasVegas/last.pth",
        "TWDiT": "Runs/TWDiT/241223_1525_LasVegas/last.pth",
    },
    # "UseCase": {}
}

def test_all():
    scheduled_tests = {("Tokyo", "Tokyo")}

    for weight_dataset, load_dataset in scheduled_tests:

        report_to = f"reports/{weight_dataset}_{load_dataset}_Ablation"

        if not os.path.exists(report_to):
            os.makedirs(report_to)

        # print(f"Start Testing TWDiT on {load_dataset}")
        # test_TWDiT(
        #     T=500,
        #     beta_min=0.0001,
        #     beta_max=0.05,
        #     data_path=f"Dataset/{load_dataset}",
        #     model_path=weights[weight_dataset]["TWDiT"],
        #     vae_path=weights[weight_dataset]["WGVAE"],
        #     report_to=report_to
        # )
        #
        # print(f"Start Testing TWDiT_MHSA on {load_dataset}")
        # test_TWDiT_MHSA(
        #     T=500,
        #     beta_min=0.0001,
        #     beta_max=0.05,
        #     data_path=f"Dataset/{load_dataset}",
        #     model_path="Runs/TWDiT_MHSA/250311_1025_Tokyonew/last.pth",
        #     vae_path=weights[weight_dataset]["WGVAE_new"],
        #     report_to=report_to
        # )

        # print(f"Start Testing TWDiT_NoVAE on {load_dataset}")
        # test_TWDiT_NoVAE(
        #     T=500,
        #     beta_min=0.0001,
        #     beta_max=0.05,
        #     data_path=f"Dataset/{load_dataset}",
        #     model_path=weights[weight_dataset]["TWDiT_NoVAE"],
        #     #deduplicator_path=weights[weight_dataset]["Deduplicator"],
        #     report_to=report_to
        # )

        print(f"Start Trasting TGTransformer on {load_dataset}")
        test_TGTransformer(
            dataset_path=f"Dataset/{load_dataset}",
            model_path=weights[weight_dataset]["TGTransformer"],
            report_to=report_to
        )


if __name__ == "__main__":
    test_all()



from TrainEvalTest.TRDiT.test_on_metrics import test as test_TRDiT
from TrainEvalTest.TR2RM.test_on_metrics import test as test_TR2RM
from TrainEvalTest.DFDRUNet.test_on_metrics import test as test_DFDRUNet
from TrainEvalTest.Graphusion.test_on_metrics import test as test_Graphusion
from TrainEvalTest.SmallMapUNet.test_on_metrics import test as test_SmallMap


weights = {
    "Tokyo": {
        "TR2RM": "Runs/TR2RM/241209_0940_Tokyo/last.pth",
        "DFDRUNet": "Runs/DFDRUNet/241208_2042_Tokyo/last.pth",
        "SmallMap": "Runs/SmallMap/241209_1422_Tokyo/last.pth",
        "NodeExtractor": "Runs/NodeExtractor/241209_1814_Tokyo/last.pth"
    },
    "Shanghai": {},
    "Paris": {},
    "LasVegas": {},
    "UseCase": {}
}


if __name__ == "__main__":
    for dataset_name in weights:
        print(f"Start Testing TR2RM on {dataset_name}")
        test_TR2RM(
            dataset_path=f"Dataset/{dataset_name}",
            model_path=weights[dataset_name]["TR2RM"],
            node_extractor_path=weights[dataset_name]["NodeExtractor"]
        )

        print(f"Start Testing DFDRUNet on {dataset_name}")
        test_DFDRUNet(
            dataset_path=f"Dataset/{dataset_name}",
            model_path=weights[dataset_name]["DFDRUNet"],
            node_extractor_path=weights[dataset_name]["NodeExtractor"]
        )

        print(f"Start Testing SmallMap on {dataset_name}")
        test_SmallMap(
            dataset_path=f"Dataset/{dataset_name}",
            model_path=weights[dataset_name]["SmallMap"],
            node_extractor_path=weights[dataset_name]["NodeExtractor"]
        )

        print(f"Start Testing Graphusion on {dataset_name}")
        test_Graphusion(
            dataset_path=f"Dataset/{dataset_name}",
            model_path=weights[dataset_name]["Graphusion"],
            vae_path=weights[dataset_name]["GraphusionVAE"]
        )

        print(f"Start Testing TRDiT on {dataset_name}")
        test_TRDiT(
            data_path=f"Dataset/{dataset_name}",
            model_path=weights[dataset_name]["TRDiT"],
            vae_path=weights[dataset_name]["RGVAE"]
        )


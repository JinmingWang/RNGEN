from TrainEvalTest.TRDiT.test_on_metrics import test as test_TRDiT
from TrainEvalTest.TR2RM.test_on_metrics import test as test_TR2RM
from TrainEvalTest.DFDRUNet.test_on_metrics import test as test_DFDRUNet
from TrainEvalTest.Graphusion.test_on_metrics import test as test_Graphusion
from TrainEvalTest.SmallMapUNet.test_on_metrics import test as test_SmallMap


weights = {
    "Tokyo": {},
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

        print(f"Start Testing SmallMapUNet on {dataset_name}")
        test_SmallMap(
            dataset_path=f"Dataset/{dataset_name}",
            model_path=weights[dataset_name]["SmallMapUNet"],
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


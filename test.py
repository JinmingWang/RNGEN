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
        "NodeExtractor": "Runs/NodeExtractor/241209_1814_Tokyo/last.pth",
        "RGVAE": "Runs/RGVAE/241212_0246_Tokyo/last.pth",
        "TRDiT": "Runs/TRDiT/241213_1116_Tokyo/last.pth",
        "GraphusionVAE": "Runs/GraphusionVAE/241212_2337_Tokyo/last.pth",
        "Graphusion": "Runs/Graphusion/241213_0709_Tokyo/last.pth"
    },
    "Shanghai": {
        "TR2RM": "Runs/TR2RM/241214_2302_Shanghai/last.pth",
        "DFDRUNet": "Runs/DFDRUNet/241214_1357_Shanghai/last.pth",
        "SmallMap": "Runs/SmallMap/241215_0222_Shanghai/last.pth",
        "NodeExtractor": "Runs/NodeExtractor/241215_1543_Shanghai/last.pth",
        "RGVAE": "Runs/RGVAE/241218_2207_Shanghai/last.pth",
        "TRDiT": "Runs/TRDiT/241219_0901_Shanghai/last.pth",
        "GraphusionVAE": "Runs/GraphusionVAE/241216_1428_Shanghai/last.pth",
        "Graphusion": "Runs/Graphusion/241216_1612_Shanghai/last.pth"
    },
    "LasVegas": {
        "TR2RM": "Runs/TR2RM/241221_0455_LasVegas/last.pth",
        "DFDRUNet": "Runs/DFDRUNet/241220_1949_LasVegas/last.pth",
        "SmallMap": "Runs/SmallMap/241221_0816_LasVegas/last.pth",
        "NodeExtractor": "Runs/NodeExtractor/241221_2037_LasVegas/last.pth",
        "RGVAE": "Runs/RGVAE/241223_0427_LasVegas/last.pth",
        "TRDiT": "Runs/TRDiT/241223_1525_LasVegas/last.pth",
        "GraphusionVAE": "Runs/GraphusionVAE/241224_0644_LasVegas/last.pth",
        "Graphusion": "Runs/Graphusion/241224_0801_LasVegas/last.pth"
    },
    # "UseCase": {}
}


if __name__ == "__main__":
    weight_dataset = "LasVegas"
    load_dataset = "LasVegas"

    print(f"Start Testing TR2RM on {load_dataset}")
    test_TR2RM(
        dataset_path=f"Dataset/{load_dataset}",
        model_path=weights[weight_dataset]["TR2RM"],
        node_extractor_path=weights[weight_dataset]["NodeExtractor"]
    )

    print(f"Start Testing DFDRUNet on {load_dataset}")
    test_DFDRUNet(
        dataset_path=f"Dataset/{load_dataset}",
        model_path=weights[weight_dataset]["DFDRUNet"],
        node_extractor_path=weights[weight_dataset]["NodeExtractor"]
    )

    print(f"Start Testing SmallMap on {load_dataset}")
    test_SmallMap(
        dataset_path=f"Dataset/{load_dataset}",
        model_path=weights[weight_dataset]["SmallMap"],
        node_extractor_path=weights[weight_dataset]["NodeExtractor"]
    )

    print(f"Start Testing Graphusion on {load_dataset}")
    test_Graphusion(
        dataset_path=f"Dataset/{load_dataset}",
        model_path=weights[weight_dataset]["Graphusion"],
        vae_path=weights[weight_dataset]["GraphusionVAE"]
    )

    print(f"Start Testing TRDiT on {load_dataset}")
    test_TRDiT(
        data_path=f"Dataset/{load_dataset}",
        model_path=weights[weight_dataset]["TRDiT"],
        vae_path=weights[weight_dataset]["RGVAE"]
    )




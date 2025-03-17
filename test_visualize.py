from TrainEvalTest.TWDiT.test_on_metrics import visualize as visualize_RN_Traj_Heatmap_Segs
from TrainEvalTest.TR2RM.test_on_metrics import visualize as visualize_PredHeatmap_PredSegs

if __name__ == "__main__":
    # This dataset only contains 4 samples
    # Each corresponds to one specific scenario
    data_path = "Dataset/Tokyo_Special"

    visualize_RN_Traj_Heatmap_Segs(
        T=500,
        beta_min=0.0001,
        beta_max=0.05,
        data_path=data_path,
        model_path="Runs/TRDiT/250102_1948_Tokyo_Special/last.pth",
        vae_path="Runs/RGVAE/250102_1850_Tokyo_Special/last.pth",
        report_to="./"
    )

    visualize_PredHeatmap_PredSegs(
        dataset_path=data_path,
        model_path="Runs/TR2RM/250102_1711_Tokyo_Special/last.pth",
        node_extractor_path="Runs/NodeExtractor/241209_1814_Tokyo/last.pth",
        report_to="./"
    )
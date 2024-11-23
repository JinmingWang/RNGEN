import pandas as pd
import matplotlib.pyplot as plt
from itertools import product

report = pd.read_csv("Report.csv")


titles = ["heatmap_accuracy", "heatmap_precision", "heatmap_recall", "heatmap_f1",
          "hungarian_mae", "hungarian_mse", "chamfer_mae", "chamfer_mse"]

for item in report.iterrows():
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.view(-1)
    for i, title in enumerate(titles):
        axes[i].plot(item[title +"_mean"])


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

file_names = list(filter(lambda name: name.startswith("Report"), os.listdir()))

model_names = list(map(lambda name: name[:-4].split("_")[-1], file_names))

reports = {model_name: pd.read_csv(file_name) for (model_name, file_name) in zip(model_names, file_names)}


colors = ["red", "blue", "green", "orange", "purple", "pink", "cyan"]


titles = ["heatmap_accuracy", "heatmap_precision", "heatmap_recall", "heatmap_f1",
          "hungarian_mae", "hungarian_mse", "chamfer_mae", "chamfer_mse"]

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for i, title in enumerate(titles):
    axes[i].set_title(title)
    # draw grid
    axes[i].grid(True)
    axes[i].set_xlabel("Models")
    axes[i].set_ylabel(title.replace("_", " "))

    model_data = [dataframe[title].to_numpy() for dataframe in reports.values()]
    parts = axes[i].violinplot(model_data, showmeans=False, showmedians=False, showextrema=False)
    for j, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[j])
        # pc.set_edgecolor('black')
        pc.set_alpha(0.3)

        quartile1, medians, quartile3 = np.percentile(model_data[j], [25, 50, 75])

        # Draw a small line at median
        axes[i].scatter(j+1, medians, color='black', s=100, zorder=3, marker="_")
        axes[i].vlines(j+1, quartile1, quartile3, color='black', linestyle='-', lw=5)
        axes[i].vlines(j+1, np.min(model_data[j]), np.max(model_data[j]), color='black', linestyle='-', lw=1)

        # Set x-axis labels
        axes[i].set_xticks(np.arange(1, len(model_names) + 1))
        axes[i].set_xticklabels(model_names)

plt.tight_layout()

plt.savefig("report.png", dpi=100)

plt.show()



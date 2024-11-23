import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

report = pd.read_csv("Report.csv")


colors = ["red", "blue", "green", "yellow", "orange", "purple", "pink", "cyan"]


titles = ["heatmap_accuracy", "heatmap_precision", "heatmap_recall", "heatmap_f1",
          "hungarian_mae", "hungarian_mse", "chamfer_mae", "chamfer_mse"]

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

model_names = report["name"].unique()

for i, title in enumerate(titles):
    axes[i].set_title(title)
    # violin plot

    model_data = [np.array(report[report["name"] == name][title].values) for name in model_names]
    parts = axes[i].violinplot(model_data, showmeans=True, showmedians=True, showextrema=False)
    for j, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[j])
        pc.set_edgecolor('black')
        pc.set_alpha(0.5)

plt.tight_layout()

plt.show()



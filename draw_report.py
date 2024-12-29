import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

model_names = ["TR2RM", "DFDRUNet", "SmallMap", "Graphusion", "GraphWalker"]

file_names = [f"reports/Tokyo_Tokyo/Report_{name}.csv" for name in model_names]

print(file_names)

model_names = list(map(lambda name: name[:-4].split("_")[-1], file_names))

reports = {model_name: pd.read_csv(file_name) for (model_name, file_name) in zip(model_names, file_names)}

colors = ["red", "blue", "green", "orange", "purple", "pink", "cyan"]

titles = ["hungarian_mae", "hungarian_mse", "chamfer_mae", "chamfer_mse", "diff_seg_len"]

plot_type = "box"
# plot_type = "violin"

fig, axes = plt.subplots(5, 1, figsize=(6, 12))
axes = axes.flatten()

report_data = []

for i, title in enumerate(titles):
    axes[i].set_title(title)
    # axes[i].set_xlabel("Models")
    axes[i].set_ylabel(title.replace("_", " "))

    model_data = [dataframe[title].to_numpy()[:500] for dataframe in reports.values()]
    if plot_type == "violin":
        axes[i].grid(True)
        parts = axes[i].violinplot(model_data, showmeans=False, showmedians=False, showextrema=False)
        for j, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[j])
            # pc.set_edgecolor('black')
            pc.set_alpha(0.3)

            quartile1, medians, quartile3 = np.percentile(model_data[j], [25, 50, 75])

            # Draw a small line at median
            axes[i].scatter(j + 1, medians, color='black', s=100, zorder=3, marker="_")
            axes[i].vlines(j + 1, quartile1, quartile3, color='black', linestyle='-', lw=5)
            axes[i].vlines(j + 1, np.min(model_data[j]), np.max(model_data[j]), color='black', linestyle='-', lw=1)

    elif plot_type == "box":
        parts = axes[i].boxplot(model_data, patch_artist=True, showmeans=False, showfliers=False, showbox=True,
                                showcaps=True)
        for j, pc in enumerate(parts['boxes']):
            pc.set_facecolor(colors[j])
            pc.set_edgecolor('black')
            pc.set_alpha(0.3)

            parts['medians'][j].set_color('black')


    report_data.append(np.mean(model_data, axis=1))

    # Set x-axis labels
    axes[i].set_xticks(np.arange(1, len(model_names) + 1))
    axes[i].set_xticklabels(model_names)

report_data = np.array(report_data).T
for row in report_data:
    print(" & ".join([f"{val:.3f}" for val in row]))

plt.tight_layout()

plt.savefig("report.png", dpi=200)

plt.show()

"""
0.705 & 0.816 & 0.833 & 0.878 & 0.531                                                                                                                                                                   
0.701 & 0.812 & 0.835 & 0.879 & 0.535                                                                                                                                                                   
0.699 & 0.808 & 0.829 & 0.874 & 0.532                                                                                                                                                                   
0.398 & 0.350 & 0.513 & 0.289 & 0.213                                                                                                                                                                   
0.304 & 0.262 & 0.318 & 0.202 & 0.178
"""

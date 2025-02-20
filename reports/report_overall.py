import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from itertools import product


datasets = ["Tokyo", "Shanghai", "LasVegas"]
dataset_titles = ["TYO", "SHA", "LV"]
model_names = ["TR2RM", "DFDRUNet", "SmallMap", "Graphusion", "GraphWalker"]
display_names = ["TR2RM", "DF-DRUNet", "SmallMap", "Graphusion", "GraphWalker"]
metrics = ["hungarian_mae", "hungarian_mse", "chamfer_mae", "chamfer_mse", "diff_seg_len"]
metric_titles = ["Hungarian MAE", "Hungarian MSE", "Chamfer MAE", "Chamfer MSE", "Segment Length"]

font_size = 12

# fig, axes = plt.subplots(3, 2, figsize=(15, 6))
# axes = axes.flatten()

report_data = []

for i, metric in enumerate(metrics):

    metric_heatmap = np.zeros((len(model_names), len(datasets) ** 2))

    for j, (train_set, test_set) in enumerate(product(datasets, datasets)):
        file_names = [f"reports/{train_set}_{test_set}/Report_{name}.csv" for name in model_names]

        reports = {model_name: pd.read_csv(file_name) for (model_name, file_name) in zip(model_names, file_names)}

        score_means = [dataframe[metric].to_numpy()[:500].mean() for dataframe in reports.values()]

        metric_heatmap[:, j] = score_means

    fig, ax = plt.subplots(figsize=(7, 2))
    # ax.set_title(metric_titles[i])
    cax = ax.matshow(metric_heatmap, cmap='viridis_r', aspect='auto')

    for (k, l), val in np.ndenumerate(metric_heatmap):
        # text_color = "white" if metric_heatmap[k, l] > 0.5 else "black"
        text_color="white"
        # best and second best bold
        if val == metric_heatmap[:, l].min():
            ax.text(l, k, f"{val:.3f}", ha='center', va='center', color="red", fontsize=font_size, fontweight='bold')
        elif val == np.partition(metric_heatmap[:, l], 1)[1]:
            ax.text(l, k, f"{val:.3f}", ha='center', va='center', color=text_color, fontsize=font_size, fontweight='bold')
        else:
            ax.text(l, k, f"{val:.3f}", ha='center', va='center', color=text_color, fontsize=font_size)

    ax.set_xticks(np.arange(len(datasets) ** 2))
    ax.set_yticks(np.arange(len(model_names)))

    ax.set_xticklabels([f"{train_set}\n{test_set}" for train_set, test_set in product(dataset_titles, dataset_titles)], fontsize=font_size)
    ax.set_yticklabels(model_names, fontsize=font_size)

    ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.set_xticks(np.arange(-.5, len(datasets) ** 2, 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(model_names), 1), minor=True)

    # annotate on the left of all x-ticks
    ax.text(-0.5, -2.4, "Trained On:", ha='right', va='center', fontsize=font_size)
    ax.text(-0.5, -1.4, "Tested On:", ha='right', va='center', fontsize=font_size)

    # set colorbar
    # cbar = fig.colorbar(cax, ax=ax, aspect=5, pad=0.02)
    # value_min = metric_heatmap.min()
    # value_max = metric_heatmap.max()
    # cbar.set_ticks(np.round(np.linspace(value_min, value_max, 4), 2))
    # ticklabs = cbar.ax.get_yticklabels()
    # cbar.ax.set_yticklabels(ticklabs, fontsize=font_size)


    plt.tight_layout()

    plt.savefig(f"reports/heatmap_{metric_titles[i]}.pdf", dpi=300)

    print(metric_titles[i])
    # print the table
    for r in range(len(model_names)):
        texts = []
        for c in range(len(datasets) ** 2):
            val = metric_heatmap[r, c]
            if val == metric_heatmap[:, c].min():
                texts.append("\\fst{" + f"{val:.3f}" + "}")
            elif val == np.partition(metric_heatmap[:, c], 1)[1]:
                texts.append("\\snd{" + f"{val:.3f}" + "}")
            else:
                texts.append(f"{val:.3f}")
        print("& " + display_names[r] + " & " + " & ".join(texts) + "\\\\")

    plt.show()

# Caption:
# A comprehensive cross-city evaluation of the 5 methods using 5 metrics. The x-axis represents the training and testing cities, and the y-axis represents the 5 methods. The color and the text of each cell represent the performance of the method trained on training city and tested on testing city. The best result is bolded in red, and the second-best result is bolded.

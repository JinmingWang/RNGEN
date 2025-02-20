import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

model_names = ["TR2RM", "DFDRUNet", "SmallMap", "Graphusion", "GraphWalker"]

# K datasets: cityname_1, cityname_2, ..., cityname_K
cityname = "Shanghai"
postfixes = ["_24traj", "_32traj", "_40traj", ""]
K = len(postfixes)  # Number of datasets
file_names = [
    [f"reports/{cityname}_{cityname}{postfixes[i]}/Report_{name}.csv" for i in range(K)]
    for name in model_names
]
postfixes = ["24", "32", "40", "48"]

print(file_names)

metrics = ["hungarian_mae", "hungarian_mse", "chamfer_mae", "chamfer_mse", "diff_seg_len"]
metric_titles = ["Hungarian MAE", "Hungarian MSE", "Chamfer MAE", "Chamfer MSE", "Wasserstein Distance of\nEdge Length Distribution"]

colors = ["red", "blue", "green", "orange", "purple"]
markers = ["D", "o", "s", "v", "^"]  # Different markers for each model

# Initialize dictionary to store aggregated data
aggregated_data = {model: {title: [] for title in metrics} for model in model_names}

# Load and aggregate data
for model_idx, model_name in enumerate(model_names):
    for k in range(K):
        report = pd.read_csv(file_names[model_idx][k])
        for title in metrics:
            aggregated_data[model_name][title].append(report[title].to_numpy())

# Prepare data for line plot
# fig, axes = plt.subplots(3, 2, figsize=(10, 10))
# axes = axes.flatten()

# To collect legend entries
handles = []
labels = []

font_size = 13
marker_size = 8

fig, axes = plt.subplots(1, 5, figsize=(12, 2))

for i, title in enumerate(metrics):
    # fig, ax = plt.subplots(figsize=(6, 4))
    ax = axes[i]
    ax.set_title(metric_titles[i], fontsize=font_size)
    ax.set_xlabel("Number of Trajectories", fontsize=font_size)
    ax.set_ylabel("Loss", fontsize=font_size, labelpad=0.5)
    ax.grid(True, which='both', axis='y', color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.grid(True, which='both', axis='x', color='gray', linestyle='-', linewidth=0.5, alpha=0.5)

    for model_idx, model_name in enumerate(model_names):
        # Stack data for the current metric
        metric_data = np.array(aggregated_data[model_name][title])  # Shape (K, N)
        mean_values = metric_data.mean(axis=1)  # Mean across samples
        std_values = metric_data.std(axis=1)  # Std across samples

        # Plot mean line, std shaded area, and markers
        x = np.arange(1, K + 1)
        line, = ax.plot(x, mean_values, label=model_name, color=colors[model_idx], marker=markers[model_idx],
                        markersize=marker_size, linewidth=1, alpha=0.6, markeredgewidth=0)
        # ax.fill_between(x, mean_values - std_values * 0.3, mean_values + std_values * 0.3, color=colors[model_idx], alpha=0.1)

        # Collect legend handles and labels
        if i == 0:  # Only collect once
            handles.append(line)
            labels.append(model_name)

    ax.set_xticks(np.arange(1, len(postfixes) + 1))
    ax.set_xticklabels(postfixes, fontsize=font_size)

    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 3, 4], prune=None))
    # ylabels = [item.get_text() for item in ax.get_yticklabels()]
    # ax.set_yticklabels([f"{float(label):.2f}" for label in ylabels], fontsize=font_size)

    # plt.tight_layout()
    # plt.savefig(f"reports/scarcity_{metric_titles[i]}.pdf", dpi=300)
    # plt.show()

axes[-1].legend(handles, labels, fontsize=font_size, bbox_to_anchor=(1.1, 0.6, 0.5, 0.5), markerscale=0.8, handlelength=1)

fig.subplots_adjust(left=0.1, right=0.5, top=2.7, bottom=0.1, wspace=0.5, hspace=0.5)

# Draw another plot just for legend
# fig, ax = plt.subplots(figsize=(7, 4))
# ax.axis("off")
# ax.legend(handles, labels, loc="center", fontsize=font_size)
plt.tight_layout(rect=(0, 0, 0.99, 1), pad=0)
plt.savefig("reports/scarcity.pdf", dpi=300)
plt.show()


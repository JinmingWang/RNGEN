import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

model_names = ["TR2RM", "DFDRUNet", "SmallMap", "Graphusion", "GraphWalker"]

# K datasets: cityname_1, cityname_2, ..., cityname_K
cityname = "LasVegas"
postfixes = ["half", "LasVegas", "double", "triple"]
K = len(postfixes)  # Number of datasets
file_names = [
    [f"reports/{cityname}_{postfixes[i]}/Report_{name}.csv" for i in range(K)]
    for name in model_names
]
postfixes[1] = "normal"

print(file_names)

titles = ["hungarian_mae", "hungarian_mse", "chamfer_mae", "chamfer_mse", "diff_seg_len"]

colors = ["red", "blue", "green", "orange", "purple"]
markers = ["o", "s", "D", "v", "^"]  # Different markers for each model

# Initialize dictionary to store aggregated data
aggregated_data = {model: {title: [] for title in titles} for model in model_names}

# Load and aggregate data
for model_idx, model_name in enumerate(model_names):
    for k in range(K):
        report = pd.read_csv(file_names[model_idx][k])
        for title in titles:
            aggregated_data[model_name][title].append(report[title].to_numpy()[:500])

# Prepare data for line plot
fig, axes = plt.subplots(3, 2, figsize=(10, 10))
axes = axes.flatten()

# To collect legend entries
handles = []
labels = []

for i, title in enumerate(titles):
    axes[i].set_title(title.replace("_", " ").capitalize())
    axes[i].set_ylabel("Value")
    axes[i].grid(True)

    for model_idx, model_name in enumerate(model_names):
        # Stack data for the current metric
        metric_data = np.array(aggregated_data[model_name][title])  # Shape (K, N)
        mean_values = metric_data.mean(axis=1)  # Mean across samples
        std_values = metric_data.std(axis=1)  # Std across samples

        # Plot mean line, std shaded area, and markers
        x = np.arange(1, K + 1)
        line, = axes[i].plot(x, mean_values, label=model_name, color=colors[model_idx], marker=markers[model_idx])
        axes[i].fill_between(x, mean_values - std_values * 0.3, mean_values + std_values * 0.3, color=colors[model_idx], alpha=0.2)

        # Collect legend handles and labels
        if i == 0:  # Only collect once
            handles.append(line)
            labels.append(model_name)

    axes[i].set_xticks(np.arange(1, len(postfixes) + 1))
    axes[i].set_xticklabels(postfixes)

# Add legend to the last (empty) subplot
axes[-1].axis("off")  # Turn off the last subplot axis
axes[-1].legend(handles, labels, loc="center", fontsize="medium", markerscale=1.2)

# Remove extra subplot if necessary
if len(titles) < len(axes):
    for j in range(len(titles), len(axes)):
        axes[j].axis("off")

plt.tight_layout()
plt.savefig("scalability_test.png", dpi=200)
plt.show()

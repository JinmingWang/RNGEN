import numpy as np

# Define the data from the table
data = {
    "Hungarian_MAE": [
        [0.381, 0.474, 0.480, 0.389, 0.427, 0.470, 0.375, 0.467, 0.471],  # TR2RM
        [0.400, 0.455, 0.479, 0.397, 0.409, 0.471, 0.389, 0.412, 0.449],  # DF-DRUNet
        [0.364, 0.508, 0.525, 0.364, 0.537, 0.550, 0.351, 0.554, 0.553],  # SmallMap
        [0.396, 0.552, 0.504, 0.410, 0.415, 0.434, 0.412, 0.426, 0.412],  # Graphusion
        [0.378, 0.411, 0.415, 0.349, 0.241, 0.286, 0.365, 0.251, 0.299],  # GraphWalker
    ],
    "Hungarian_MSE": [
        [0.334, 0.507, 0.489, 0.344, 0.452, 0.485, 0.316, 0.502, 0.486],  # TR2RM
        [0.369, 0.481, 0.501, 0.362, 0.428, 0.500, 0.346, 0.432, 0.479],  # DF-DRUNet
        [0.305, 0.542, 0.530, 0.301, 0.577, 0.564, 0.282, 0.595, 0.563],  # SmallMap
        [0.348, 0.613, 0.552, 0.371, 0.485, 0.471, 0.374, 0.482, 0.450],  # Graphusion
        [0.377, 0.456, 0.449, 0.329, 0.272, 0.312, 0.350, 0.284, 0.330],  # GraphWalker
    ],
    "Chamfer_MAE": [
        [0.453, 0.777, 0.751, 0.460, 0.785, 0.759, 0.468, 0.798, 0.745],  # TR2RM
        [0.448, 0.788, 0.732, 0.440, 0.793, 0.740, 0.452, 0.775, 0.721],  # DF-DRUNet
        [0.440, 0.800, 0.707, 0.440, 0.802, 0.702, 0.444, 0.804, 0.703],  # SmallMap
        [0.488, 0.897, 0.787, 0.520, 0.826, 0.728, 0.515, 0.808, 0.672],  # Graphusion
        [0.291, 0.521, 0.454, 0.332, 0.456, 0.420, 0.315, 0.445, 0.418],  # GraphWalker
    ],
    "Chamfer_MSE": [
        [0.232, 0.659, 0.570, 0.228, 0.689, 0.590, 0.236, 0.705, 0.579],  # TR2RM
        [0.227, 0.672, 0.563, 0.217, 0.707, 0.574, 0.232, 0.658, 0.543],  # DF-DRUNet
        [0.219, 0.669, 0.521, 0.223, 0.675, 0.530, 0.216, 0.671, 0.523],  # SmallMap
        [0.270, 0.818, 0.646, 0.291, 0.856, 0.614, 0.295, 0.829, 0.576],  # Graphusion
        [0.177, 0.456, 0.368, 0.183, 0.437, 0.357, 0.176, 0.427, 0.362],  # GraphWalker
    ],
    "Wasserstein_Distance_of_Edge_Length_Distributions": [
        [0.250, 0.848, 0.572, 0.293, 0.998, 0.746, 0.272, 0.942, 0.706],  # TR2RM
        [0.286, 0.846, 0.602, 0.294, 1.076, 0.740, 0.268, 0.969, 0.730],  # DF-DRUNet
        [0.227, 0.820, 0.509, 0.238, 0.777, 0.532, 0.221, 0.740, 0.519],  # SmallMap
        [0.206, 0.733, 0.567, 0.293, 0.747, 0.499, 0.330, 0.776, 0.491],  # Graphusion
        [0.173, 0.624, 0.453, 0.185, 0.626, 0.462, 0.182, 0.622, 0.452],  # GraphWalker
    ],
}

# Convert the dictionary to a NumPy array
hungarian_mae = np.array(data["Hungarian_MAE"])
hungarian_mse = np.array(data["Hungarian_MSE"])
chamfer_mae = np.array(data["Chamfer_MAE"])
chamfer_mse = np.array(data["Chamfer_MSE"])
wasserstein_distance = np.array(data["Wasserstein_Distance_of_Edge_Length_Distributions"])

# 1. Compute the average loss for each entry (5 rows × 5 metrics)
metrics = [hungarian_mae, hungarian_mse, chamfer_mae, chamfer_mse, wasserstein_distance]    # (5, 5, 9)
average_losses = np.mean(metrics, axis=2)  # Compute mean across the 9 columns
print("Average Losses (5x5):")
print(average_losses)

# 2. Compute the percentage improvement of the lowest error (excluding GraphWalker) compared to GraphWalker
graphwalker_errors = average_losses[-1]  # Last row corresponds to GraphWalker
lowest_non_graphwalker_errors = np.min(average_losses[:-1], axis=0)  # Lowest value excluding GraphWalker
percentage_improvement = (graphwalker_errors - lowest_non_graphwalker_errors) / graphwalker_errors * 100

print("\nPercentage Improvement for Each Metric:")
print(percentage_improvement)

avg_percent_inprove = np.mean(percentage_improvement)

print("\nAverage Percentage Improvement:", avg_percent_inprove)

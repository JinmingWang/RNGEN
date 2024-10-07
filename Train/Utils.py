import matplotlib.pyplot as plt
import torch
import numpy as np

def loadModels(path: str, *models: torch.nn.Module) -> torch.nn.Module:
    """
    Load models from a file
    :param path: The path to the file
    :param models: The models to load
    :return: The loaded models
    """
    state_dicts = torch.load(path)
    for model in models:
        model.load_state_dict(state_dicts[model.__class__.__name__])
    return models


def saveModels(path: str, *models: torch.nn.Module) -> None:
    """
    Save models to a file
    :param path: The path to the file
    :param models: The models to save
    :return: None
    """
    state_dicts = {}
    for model in models:
        state_dicts[model.__class__.__name__] = model.state_dict()
    torch.save(state_dicts, path)

class PlotManager:
    def __init__(self, cell_size, grid_rows, grid_cols):
        self.cell_size = cell_size
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.fig, self.axs = plt.subplots(grid_rows, grid_cols, figsize=(grid_cols * cell_size, grid_rows * cell_size))
        self.axs = np.atleast_2d(self.axs)  # Ensure axs is 2D, even if it's a single row/col

    def plotSegments(self, segs, row, col, title):
        """
        Plot line segments given a tensor of shape (N, 2, 2), where N is the number of segments
        and each segment is defined by two points in 2D.
        """
        ax = self.axs[row, col]
        ax.clear()  # Clear previous content
        ax.set_title(title, fontsize=14, color='darkblue')

        # Extract the points for each line segment
        for seg in segs:
            x = seg[:, 0].cpu().numpy()  # X coordinates
            y = seg[:, 1].cpu().numpy()  # Y coordinates
            ax.plot(x, y, marker='.', linestyle='-', color='#63B2EE', markersize=10, markerfacecolor='#76DA91',
                    lw=2)

        ax.set_xlim([-3, 3])
        ax.set_ylim([-3, 3])
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)

    def plotTrajs(self, trajs, row, col, title):
        """
        Plot trajectories given a tensor of shape (N, L, 2), where N is the number of trajectories,
        and each trajectory consists of L points in 2D.
        """
        ax = self.axs[row, col]
        ax.clear()  # Clear previous content
        ax.set_title(title, fontsize=14, color='darkgreen')

        # Extract the points for each trajectory
        for traj in trajs:
            x = traj[:, 0].cpu().detach().numpy()  # X coordinates
            y = traj[:, 1].cpu().detach().numpy()  # Y coordinates
            ax.plot(x, y, marker='.', linestyle='-', color='#F8CB7F', markersize=5, alpha=0.3,
                    markerfacecolor='red', lw=1)

        ax.set_xlim([-3, 3])
        ax.set_ylim([-3, 3])
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)

    def plotHeatmap(self, heatmap, row, col, title):
        """
        Plot a heatmap given a tensor of shape (1, H, W) or (H, W), where H is the height and W is the width.
        A colorbar is added, and the origin is set to 'lower' by default.
        """
        ax = self.axs[row, col]
        ax.clear()  # Clear previous content
        ax.set_title(title, fontsize=14, color='darkred')

        if heatmap.ndim == 3:  # Shape (1, H, W)
            heatmap = heatmap.squeeze(0)

        heatmap_np = heatmap.cpu().numpy()
        cax = ax.imshow(heatmap_np, origin='lower', cmap='coolwarm')
        self.fig.colorbar(cax, ax=ax)

        ax.axis('off')

    def plotNodesWithAdjMat(self, nodes, adj_mat, row, col, title):
        """
        Plot nodes and their connections based on the adjacency matrix.

        nodes: Tensor of shape (N, 3), where each row is (x, y, is_valid_node)
        adj_mat: Tensor of shape (N, N), adjacency matrix representing connections between nodes
        row, col: Grid position to plot
        title: Title of the plot
        """
        ax = self.axs[row, col]
        ax.clear()  # Clear previous content
        ax.set_title(title, fontsize=14, color='darkblue')

        # Extract valid nodes (where is_valid_node == 1)
        valid_mask = nodes[:, 2].bool()
        valid_nodes = nodes[valid_mask]

        ax.scatter(valid_nodes[:, 0].cpu().numpy(), valid_nodes[:, 1].cpu().numpy(),
                   color='#76DA91', s=20, edgecolors='#63B2EE')

        # Plot connections based on adjacency matrix
        adj_np = adj_mat.cpu().numpy()
        num_nodes = len(valid_nodes)

        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if adj_np[i, j] == 1:
                    p1 = valid_nodes[i, :2].cpu().numpy()
                    p2 = valid_nodes[j, :2].cpu().numpy()
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], linestyle='-', color='#63B2EE', lw=2)

        ax.set_xlim([-3, 3])
        ax.set_ylim([-3, 3])
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)

    def show(self):
        plt.tight_layout()
        plt.show()

    def save(self, path: str):
        plt.savefig(path, dpi=200)

    def getFigure(self):
        """Return the figure object, which can be used for TensorBoard SummaryWriter."""
        return self.fig


class MovingAvg():
    def __init__(self, window_size: int):
        self.window_size = window_size
        self.values = []

    def update(self, value):
        self.values.append(float(value))
        if len(self.values) > self.window_size:
            self.values.pop(0)

    def get(self) -> float:
        return np.mean(self.values).item()

    def __len__(self):
        return len(self.values)
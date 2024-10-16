import matplotlib.pyplot as plt
import torch
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.live import Live
from threading import Thread
import time

from typing import List, Tuple, Dict
Tensor = torch.Tensor

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


def setPaddingToZero(lengths: torch.Tensor, sequences: List[Tensor]=None, matrices: List[Tensor]=None):
    """
    Set the padding part of tensors to zero
    :param lengths: the lengths of valid part
    :param tensors: either sequential tensor of shape (B, L, C) or adj_mat tensor of shape (B, L, L)
    :return:
    """
    if sequences is None:
        sequences = []
    if matrices is None:
        matrices = []

    B = lengths.shape[0]

    for i in range(len(sequences)):
        for b in range(B):
            sequences[i][b, lengths[b]:] = 0

    for i in range(len(matrices)):
        for b in range(B):
            matrices[i][b, lengths[b]:, :] = 0
            matrices[i][b, :, lengths[b]:] = 0


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
            x = seg[:, 0].cpu().detach().numpy()  # X coordinates
            y = seg[:, 1].cpu().detach().numpy()  # Y coordinates
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

        valid_nodes = nodes

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
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], linestyle='-', color='#63B2EE', lw=1)

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


class ProgressManager:
    def __init__(self, items_per_epoch: int, epochs: int, show_recent: int, refresh_interval: int = 1, custom_fields: List[str] = None):
        """
        :param dataloader: the data loader
        :param epochs: the total number of epochs
        :param show_recent: how many recent epochs to display
        :param description: the texts showing in front of progress bar
        """
        self.epochs = epochs
        self.display_recent = show_recent
        self.steps_per_epoch = items_per_epoch
        self.refresh_interval = refresh_interval
        self.custom_fields = [] if custom_fields is None else custom_fields

        # Initialize tracking
        self.overall_progress = 0
        self.start_time = time.time()  # Start tracking time
        self.console = Console(width=120)
        self.live = None

        self.current_epoch = 1
        self.current_step = 1

        # Initialize progress data for recent epochs
        self.memory = [
            ({"epoch": epoch, "completed": 0, "t_start": 0.0, "t_end": 0.0} | dict(zip(self.custom_fields, [0]*len(custom_fields))))
            for epoch in range(1, epochs + 1)
        ]

    def __enter__(self):
        """Enter the context: start displaying the progress."""
        self.console.print("[bold green]Starting Training...[/bold green]")
        self.live = Live(self.render_progress_table(1), refresh_per_second=1, console=self.console)
        self.live.__enter__()
        self.live_thread = Thread(target=self.live_update)
        self.thread_stop = False
        self.start_time = time.time()
        self.live_thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context: stop the progress and display the completion message."""
        self.console.print("[bold green]Training Completed![/bold green]")
        self.thread_stop = True
        self.live.__exit__(exc_type, exc_val, exc_tb)

    def update(self, current_epoch: int, current_step: int, **custom_values):
        """Update the progress of the current epoch and overall progress."""
        # Update overall progress
        self.overall_progress += 1

        # Update the specific epoch progress
        self.memory[current_epoch]["completed"] = current_step + 1
        for k in self.custom_fields:
            self.memory[current_epoch][k] = custom_values[k]

        self.current_epoch = current_epoch + 1
        self.current_step = current_step + 1

        if self.memory[current_epoch]["t_start"] == 0:
            # we are starting a new epoch, record the start time
            self.memory[current_epoch]["t_start"] = time.time()
            if current_epoch >= 1:
                # update the end time of the previous epoch
                self.memory[current_epoch - 1]["t_end"] = time.time()


    def format_time(self, seconds: float) -> str:
        """Format time in seconds into hh:mm:ss."""
        hrs, rem = divmod(seconds, 3600)
        mins, secs = divmod(rem, 60)
        return f"{int(hrs):02}:{int(mins):02}:{int(secs):02}"

    def render_progress_table(self, current_epoch: int) -> Table:
        """Create a table to display overall and recent epoch progress."""
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Desc", width=10)
        table.add_column("Percent", width=7)
        table.add_column("Progress", width=10)
        table.add_column("Elapsed", width=8)
        table.add_column("Remaining", width=9)
        for k in self.custom_fields:
            table.add_column(k, width=12)

        # Calculate time details
        elapsed_time_total = time.time() - self.start_time
        total_steps = self.epochs * self.steps_per_epoch
        remaining_time_total = (total_steps - self.overall_progress) * (
                    elapsed_time_total / self.overall_progress) if self.overall_progress > 0 else 0

        # Overall progress row
        overall_percentage = self.overall_progress / total_steps * 100
        table.add_row(
            "[#00aaff]Overall[#00aaff]",
            f"[#00aaff]{overall_percentage:.2f}%[/#00aaff]",
            f"[#00aaff]{self.overall_progress}/{total_steps}[/#00aaff]",
            f"[#00aaff]{self.format_time(elapsed_time_total)}[/#00aaff]",
            f"[#00aaff]{self.format_time(remaining_time_total)}[/#00aaff]",
            "",
        )

        # Display the recent epochs
        for i in range(max(0, current_epoch - self.display_recent), current_epoch):
            epoch_data = self.memory[i]
            epoch_percentage = epoch_data["completed"] / self.steps_per_epoch * 100
            if epoch_data["t_end"] == 0:
                # epoch is not completed yet
                elapsed_time_epoch = time.time() - epoch_data["t_start"]
                remaining_time_epoch = (self.steps_per_epoch - epoch_data["completed"]) * (
                            elapsed_time_epoch / epoch_data["completed"]) if epoch_data["completed"] > 0 else 0
                complete_color = "green" if remaining_time_epoch == 0 else "#ffff00"
            else:
                # epoch is completed
                elapsed_time_epoch = epoch_data["t_end"] - epoch_data["t_start"]
                remaining_time_epoch = 0
                complete_color = "green"
            table.add_row(
                f"[{complete_color}]Epoch {epoch_data['epoch']}[/{complete_color}]",
                f"[{complete_color}]{epoch_percentage:.2f}%[/{complete_color}]",
                f"[{complete_color}]{epoch_data['completed']}/{self.steps_per_epoch}[/{complete_color}]",
                f"[{complete_color}]{self.format_time(elapsed_time_epoch)}[/{complete_color}]",
                f"[{complete_color}]{self.format_time(remaining_time_epoch)}[/{complete_color}]",
                *[f"{epoch_data[k]:.5e}" for k in self.custom_fields]
            )

        return table

    def live_update(self):
        while not self.thread_stop:
            self.live.update(self.render_progress_table(self.current_epoch))
            time.sleep(self.refresh_interval)




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



if __name__ == "__main__":

    t1 = torch.randn(1, 3, 2)
    t2 = torch.randn(1, 4, 2)
    print(t1)
    print(t2)

    setPaddingToZero(torch.tensor([2,]), [t1, t2], None)
    print(t1)
    print(t2)

from TrainEvalTest.Utils import *
from datetime import datetime
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

import os

from Dataset import DEVICE, RoadNetworkDataset
from Models import UNet2D, AD_Linked_Net, NodeExtractor


def train(
        title: str = "initial",
        dataset_path: str = "Dataset/Tokyo_10k_sparse",
        kl_weight: float = 1e-6,
        lr: float = 1e-4,
        lr_reduce_factor: float = 0.5,
        lr_reduce_patience: int = 30,
        lr_reduce_min: float = 1e-7,
        lr_reduce_threshold: float = 1e-5,
        epochs: int = 1000,
        B: int = 32,
        mov_avg_len: int = 6,
        log_interval: int = 10,
        heatmap_model_path: str = "Runs/TR2RM/241124_1849_sparse/last.pth",
        load_weights: str = None
):
    log_dir = f"./Runs/NodeExtractor/{datetime.now().strftime('%Y%m%d_%H%M')[2:]}_{title}/"
    # Dataset & DataLoader
    dataset = RoadNetworkDataset(folder_path=dataset_path,
                                 batch_size=B,
                                 drop_last=True,
                                 set_name="train",
                                 permute_seq=False,
                                 enable_aug=False,
                                 img_H=256,
                                 img_W=256
                                 )

    heatmap_model = AD_Linked_Net(d_in=4, H=256, W=256).to(DEVICE)
    loadModels(heatmap_model_path, ADLinkedNet=heatmap_model)
    heatmap_model.eval()

    node_model = NodeExtractor().to(DEVICE)

    if load_weights is not None:
        loadModels(load_weights, node_model=node_model)

    loss_func = torch.nn.BCELoss()

    # Optimizer & Scheduler
    optimizer = AdamW(node_model.parameters(), lr=lr)
    lr_scheduler = ReduceLROnPlateau(optimizer, factor=lr_reduce_factor, patience=lr_reduce_patience,
                                     min_lr=lr_reduce_min, threshold=lr_reduce_threshold)

    # Prepare Logging
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    mov_avg_loss = MovingAvg(mov_avg_len * len(dataset))
    global_step = 0
    best_loss = float("inf")
    plot_manager = PlotManager(4, 1, 5)

    with ProgressManager(len(dataset), epochs, 5, 2, ["Loss", "lr"]) as progress:
        for e in range(epochs):
            total_loss = 0
            for i, batch in enumerate(dataset):
                batch: Dict[str, torch.Tensor]

                batch |= RoadNetworkDataset.getNodeHeatmaps(batch)

                optimizer.zero_grad()

                with torch.no_grad():
                    pred_heatmap = heatmap_model(torch.cat([batch["heatmap"], batch["image"]], dim=1))

                pred_nodes = node_model(pred_heatmap)

                loss = loss_func(pred_nodes, batch["node_heatmap"])

                # Backpropagation
                loss.backward()
                optimizer.step()

                total_loss += loss
                global_step += 1
                mov_avg_loss.update(loss)

                # Progress update
                progress.update(e, i,
                                Loss=mov_avg_loss.get(),
                                lr=optimizer.param_groups[0]['lr'])

                # Logging
                if global_step % log_interval == 0:
                    writer.add_scalar("Loss", mov_avg_loss.get(), global_step)
                    writer.add_scalar("Learning Rate", optimizer.param_groups[0]["lr"], global_step)

            # Plot reconstructed segments and graphs
            plot_manager.plotHeatmap(batch["heatmap"][0], 0, 0, "Input")
            plot_manager.plotHeatmap(batch["target_heatmaps"][0], 0, 1, "Target")
            plot_manager.plotHeatmap(pred_heatmap[0], 0, 2, "Pred Heatmap")
            plot_manager.plotHeatmap(pred_nodes[0], 0, 3, "Pred Nodes")
            plot_manager.plotHeatmap(batch["node_heatmap"][0], 0, 4, "Target Nodes")

            writer.add_figure("Figures", plot_manager.getFigure(), global_step)

            # Save models
            saveModels(os.path.join(log_dir, "last.pth"), node_model = node_model)
            if total_loss < best_loss:
                best_loss = total_loss
                saveModels(os.path.join(log_dir, "best.pth"), node_model = node_model)

            # Step scheduler
            lr_scheduler.step(total_loss)

            if optimizer.param_groups[0]["lr"] <= lr_reduce_min:
                # Stop training if learning rate is too low
                break

    return os.path.join(log_dir, "last.pth")

if __name__ == "__main__":
    train()

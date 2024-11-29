from TrainEvalTest.GlobalConfigs import *
from TrainEvalTest.NodeExtractor.configs import *
from TrainEvalTest.Utils import *

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

import os

from Dataset import DEVICE, RoadNetworkDataset
from Models import UNet2D, AD_Linked_Net, NodeExtractor


def train():
    # Dataset & DataLoader
    dataset = RoadNetworkDataset("Dataset/Tokyo_10k_sparse",
                                 batch_size=B,
                                 drop_last=True,
                                 set_name="train",
                                 permute_seq=False,
                                 enable_aug=False,
                                 img_H=256,
                                 img_W=256
                                 )

    heatmap_model = AD_Linked_Net(d_in=4, H=256, W=256).to(DEVICE)
    loadModels("Runs/TR2RM/241124_1849_sparse/last.pth", ADLinkedNet=heatmap_model)
    heatmap_model.eval()

    node_model = NodeExtractor().to(DEVICE)

    loss_func = torch.nn.BCELoss()

    # Optimizer & Scheduler
    optimizer = AdamW(node_model.parameters(), lr=LR)
    lr_scheduler = ReduceLROnPlateau(optimizer, factor=LR_REDUCE_FACTOR, patience=LR_REDUCE_PATIENCE,
                                     min_lr=LR_REDUCE_MIN, threshold=LR_REDUCE_THRESHOLD)

    # Prepare Logging
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR, exist_ok=True)
    writer = SummaryWriter(log_dir=LOG_DIR)
    mov_avg_loss = MovingAvg(MOV_AVG_LEN * len(dataset))
    global_step = 0
    best_loss = float("inf")
    plot_manager = PlotManager(4, 1, 5)

    with ProgressManager(len(dataset), EPOCHS, 5, 2, ["Loss", "lr"]) as progress:
        for e in range(EPOCHS):
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
                if global_step % LOG_INTERVAL == 0:
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
            saveModels(os.path.join(LOG_DIR, "last.pth"), node_model = node_model)
            if total_loss < best_loss:
                best_loss = total_loss
                saveModels(os.path.join(LOG_DIR, "best.pth"), node_model = node_model)

            # Step scheduler
            lr_scheduler.step(total_loss)


if __name__ == "__main__":
    train()

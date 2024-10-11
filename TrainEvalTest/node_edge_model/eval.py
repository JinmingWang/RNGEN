import matplotlib.pyplot as plt

from TrainEvalTest.GlobalConfigs import *
from TrainEvalTest.node_edge_model.configs import *
from TrainEvalTest.Utils import *
from Models import HungarianLoss_SeqMat, DiffusionNet, Encoder
from Diffusion import DDPM

import torch


def eval(batch: Dict[str, Tensor], encoder: Encoder, diffusion_net: DiffusionNet, ddpm: DDPM) -> Tuple[plt.Figure, Tensor]:
    """
    Evaluate the model on the given batch
    :param batch: The batch to evaluate
    :param encoder: The encoder model
    :param diffusion_net: The diffusion network model
    :param ddpm: The diffusion scheduler
    :return: The figure and loss
    """
    encoder.eval()
    diffusion_net.eval()

    with torch.no_grad():
        traj_enc = encoder(batch["trajs"])
        node_noise = torch.randn_like(batch["nodes"])
        adj_mat_noise = torch.randn_like(batch["adj_mats"])
        # setPaddingToZero(batch["n_nodes"], [node_noise], [adj_mat_noise])

        def pred_func(noisy_contents: List[Tensor], t: Tensor) -> List[Tensor]:
            # setPaddingToZero(batch["n_nodes"], [noisy_contents[0]], [noisy_contents[1]])
            node_noise_pred, adj_mat_noise_pred = diffusion_net(*noisy_contents, traj_enc, t)
            # setPaddingToZero(batch["n_nodes"], [node_noise_pred], [adj_mat_noise_pred])
            return [node_noise_pred, adj_mat_noise_pred]

        nodes, adj_mat = ddpm.diffusionBackward([node_noise, adj_mat_noise], pred_func)

    # Remove padding nodes
    valid_mask = nodes[:, :, 2] >= 0.5
    adj_mat = (adj_mat > 0.5).to(torch.int32)

    # Get valid nodes and adjacency matrices
    valid_nodes = []
    valid_adj_mat = []
    for b in range(batch["nodes"].shape[0]):
        valid_nodes.append(nodes[b, :, :2][valid_mask[b]])
        valid_adj_mat.append(adj_mat[b][valid_mask[b]][:, valid_mask[b]])

    node_loss, edge_loss = HungarianLoss_SeqMat()(nodes, batch["nodes"], adj_mat, batch["adj_mats"])

    plot_manager = PlotManager(5, 2, 2)

    # Plot reconstructed graph
    plot_manager.plotNodesWithAdjMat(batch["nodes"][0], batch["adj_mats"][0], 0, 0, "Ground Truth")
    plot_manager.plotNodesWithAdjMat(valid_nodes[0], valid_adj_mat[0], 0, 1, "Reconstructed")
    plot_manager.plotTrajs(batch["trajs"][0], 1, 0, "Trajectories")

    encoder.train()
    diffusion_net.train()

    return plot_manager.getFigure(), node_loss + edge_loss

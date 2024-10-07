import matplotlib.pyplot as plt

from TrainEvalTest.GlobalConfigs import *
from TrainEvalTest.node_edge_model.configs import *
from TrainEvalTest.Utils import *
from Models import HungarianLoss, DiffusionNet, Encoder
from Diffusion import DDPM

import torch


def eval(batch: Dict[str, Tensor], encoder: Encoder, diffusion_net: DiffusionNet, ddpm: DDPM) -> Tuple[plt.Figure, Tensor]:
    encoder.eval()
    diffusion_net.eval()

    with torch.no_grad():
        traj_enc = encoder(batch["trajs"])
        node_noise = torch.randn_like(batch["nodes"])
        adj_mat_noise = torch.randn_like(batch["adj_mats"])
        setPaddingToZero(batch["n_nodes"], [node_noise], [adj_mat_noise])

        def pred_func(noisy_contents: List[Tensor], t: Tensor) -> List[Tensor]:
            setPaddingToZero(batch["n_nodes"], [noisy_contents[0]], [noisy_contents[1]])
            node_noise_pred, adj_mat_noise_pred = diffusion_net(*noisy_contents, traj_enc, t)
            setPaddingToZero(batch["n_nodes"], [node_noise_pred], [adj_mat_noise_pred])
            return [node_noise_pred, adj_mat_noise_pred]

        nodes, adj_mat = ddpm.diffusionBackward([node_noise, adj_mat_noise], pred_func)
        adj_mat = (adj_mat > 0.5).to(torch.int32)

    node_loss = HungarianLoss()(nodes, batch["nodes"])

    plot_manager = PlotManager(5, 2, 2)

    plot_manager.plotNodesWithAdjMat(batch["nodes"][0], batch["adj_mats"][0], 0, 0, "Original Graph")
    plot_manager.plotNodesWithAdjMat(nodes[0], torch.zeros_like(adj_mat[0]), 0, 1, "Reconstructed Graph")
    plot_manager.plotTrajs(batch["trajs"][0], 1, 0, "Trajectories")

    encoder.train()
    diffusion_net.train()

    return plot_manager.getFigure(), node_loss

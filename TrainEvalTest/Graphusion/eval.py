from Dataset import RoadNetworkDataset
from TrainEvalTest.GraphusionVAE.train import nodesAdjMatToSegs
from TrainEvalTest.Utils import *
from Models import Graphusion, GraphusionVAE
from Diffusion import DDIM

from typing import Callable

import torch


def pred_func(noisy_contents: List[Tensor], t: Tensor, model: torch.nn.Module, trajs: Tensor):
    pred = model(*noisy_contents, trajs, t)
    return [pred]


def getEvalFunction(dataset_path: str, vae: GraphusionVAE) -> Callable:
    """
    Evaluate the model on the given batch
    :param vae: The VAE model
    :return: The figure and loss
    """
    test_set = RoadNetworkDataset(dataset_path,
                                 batch_size=10,
                                 drop_last=True,
                                 set_name="test",
                                 enable_aug=True,
                                 img_H=16,
                                 img_W=16,
                                 need_nodes=True
                                 )

    batch = test_set[0:10]

    with torch.no_grad():
        latent, _ = vae.encode(batch["nodes"], batch["edges"], batch["adj_mat"])

    latent_noise = torch.randn_like(latent)

    plot_manager = PlotManager(4, 1, 5)

    loss_func = torch.nn.MSELoss()

    def eval(graphusion, ddim: DDIM) -> Tuple[plt.Figure, Tensor]:

        graphusion.eval()

        with torch.no_grad():
            latent_pred = ddim.diffusionBackward([latent_noise], pred_func, mode="eps", model=graphusion, trajs=batch["trajs"])[0]
            f_nodes, f_edges, pred_adj_mat, pred_degrees = vae.decode(latent_pred)

        loss = (loss_func(f_nodes, batch["nodes"]) +
                loss_func(f_edges, batch["edges"]) +
                loss_func(pred_adj_mat, batch["adj_mat"]) +
                loss_func(pred_degrees, batch["degrees"]))

        pred_segs = nodesAdjMatToSegs(f_nodes[0:1], pred_adj_mat[0:1], f_edges[0:1], threshold=0.5)

        plot_manager.plotNodesWithAdjMat(batch["nodes"][0], batch["adj_mat"][0], 0, 0, "Nodes")
        plot_manager.plotSegments(batch["segs"][0], 0, 1, "Segs", color="blue")
        plot_manager.plotNodesWithAdjMat(f_nodes[0], pred_adj_mat[0], 0, 2, "Reconstructed Nodes")
        plot_manager.plotTrajs(batch["trajs"][0], 0, 3, "Trajectories")
        plot_manager.plotSegments(pred_segs[0], 0, 4, "Pred Segments", color="red")

        graphusion.train()

        return plot_manager.getFigure(), loss.item()

    return eval

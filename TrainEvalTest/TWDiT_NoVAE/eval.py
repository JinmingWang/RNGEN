from Dataset import RoadNetworkDataset
from TrainEvalTest.Utils import *
from Diffusion import DDIM

from typing import Callable

import torch


def pred_func(noisy_contents: List[Tensor], t: Tensor, model: torch.nn.Module, trajs: Tensor):
    pred = model(*noisy_contents, trajs, t)
    return [pred]


def getEvalFunction(dataset_path: str) -> Callable:
    """
    Evaluate the model on the given batch
    :param vae: The VAE model
    :return: The figure and loss
    """
    test_set = RoadNetworkDataset(dataset_path,
                                 batch_size=10,
                                 drop_last=True,
                                 set_name="test",
                                 enable_aug=False,
                                 img_H=16,
                                 img_W=16
                                 )

    batch = test_set[0:10]

    noise = torch.randn_like(batch["routes"].flatten(3))

    plot_manager = PlotManager(4, 1, 4)

    def eval(DiT, ddim: DDIM) -> Tuple[plt.Figure, Tensor]:

        DiT.eval()

        with torch.no_grad():
            routes_pred = ddim.diffusionBackward([noise], pred_func, mode="eps", model=DiT, trajs=batch["trajs"])[0]

        loss = torch.nn.functional.mse_loss(routes_pred, batch["routes"].flatten(3))

        plot_manager.plotSegments(batch["routes"][0], 0, 0, "Routes", color="red")
        plot_manager.plotSegments(batch["segs"][0], 0, 1, "Segs", color="blue")
        plot_manager.plotSegments(routes_pred[0].detach(), 0, 2, "Pred Routes")
        plot_manager.plotTrajs(batch["trajs"][0], 0, 3, "Trajectories")

        DiT.train()

        return plot_manager.getFigure(), loss.item()

    return eval

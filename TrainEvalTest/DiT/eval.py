from Dataset import LaDeCachedDataset
from TrainEvalTest.GlobalConfigs import *
from TrainEvalTest.DiT.configs import *
from TrainEvalTest.Utils import *
from Models import HungarianLoss, HungarianMode, SegmentsModel
from Diffusion import DDIM

from typing import Callable

import torch

def getEvalFunction() -> Callable:
    """
    Evaluate the model on the given batch
    :param encoder: The encoder model
    :param diffusion_net: The diffusion network model
    :param ddim: The diffusion scheduler
    :return: The figure and loss
    """
    test_set = LaDeCachedDataset(DATA_DIR, max_trajs=N_TRAJS, set_name="test")
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=B, collate_fn=LaDeCachedDataset.collate_fn, shuffle=False)

    batch = next(iter(test_loader))

    noise = torch.randn(B, N_TRAJS, L_PATH, 2).to(batch["paths"].device)

    def eval(models: Dict[str, torch.nn.Module], ddim: DDIM) -> Tuple[plt.Figure, Tensor]:

        is_training = dict()
        for name in models:
            is_training[name] = models[name].training
            models[name].eval()

        with torch.no_grad():

            def pred_func(noisy_contents: List[Tensor], t: Tensor):
                pred = models["DiT"](*noisy_contents, batch["trajs"], t)
                return [pred]

            pred_paths = ddim.diffusionBackward([noise], pred_func, mode="eps")[0]

        loss = torch.nn.functional.mse_loss(pred_paths, batch["paths"])

        plot_manager = PlotManager(5, 1, 4)
        plot_manager.plotSegments(batch["segs"][0], 0, 0, "Segs")
        plot_manager.plotTrajs(batch["trajs"][0], 0, 1, "Trajectories")
        plot_manager.plotTrajs(batch["paths"][0], 0, 2, "Paths")
        plot_manager.plotTrajs(pred_paths[0], 0, 3, "Pred Paths")

        for name in models:
            if is_training[name]:
                models[name].train()

        return plot_manager.getFigure(), loss

    return eval

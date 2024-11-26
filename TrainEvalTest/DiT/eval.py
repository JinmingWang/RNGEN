from Dataset import RoadNetworkDataset
from TrainEvalTest.GlobalConfigs import *
from TrainEvalTest.DiT.configs import *
from TrainEvalTest.Utils import *
from Models import RoutesDiT, CrossDomainVAE
from Diffusion import DDIM

from typing import Callable

import torch

def getEvalFunction(vae: CrossDomainVAE) -> Callable:
    """
    Evaluate the model on the given batch
    :param vae: The VAE model
    :return: The figure and loss
    """
    test_set = RoadNetworkDataset("Dataset/Tokyo_10k_sparse",
                                 batch_size=B,
                                 drop_last=True,
                                 set_name="train",
                                 enable_aug=True,
                                 img_H=16,
                                 img_W=16
                                 )

    batch = test_set[0:B]

    with torch.no_grad():
        latent, _ = vae.encode(batch["routes"])

    latent_noise = torch.randn_like(latent)

    def eval(models: Dict[str, torch.nn.Module], ddim: DDIM) -> Tuple[plt.Figure, Tensor]:

        is_training = dict()
        for name in models:
            is_training[name] = models[name].training
            models[name].eval()

        with torch.no_grad():
            def pred_func(noisy_contents: List[Tensor], t: Tensor):
                pred = models["DiT"](*noisy_contents, batch["trajs"], t)
                return [pred]

            latent_pred = ddim.diffusionBackward([latent_noise], pred_func, mode="eps")[0]

            duplicate_segs, cluster_mat, cluster_means, coi_means = vae.decode(latent_pred)

        loss = torch.nn.functional.mse_loss(duplicate_segs, batch["routes"].flatten(1, 2))

        plot_manager = PlotManager(4, 1, 5)
        plot_manager.plotSegments(batch["routes"][0], 0, 0, "Routes", color="red")
        plot_manager.plotSegments(batch["segs"][0], 0, 1, "Segs", color="blue")
        plot_manager.plotSegments(coi_means[0], 0, 2, "Pred Segs", color="green")
        plot_manager.plotSegments(duplicate_segs[0], 0, 3, "Pred Duplicate Segs")
        plot_manager.plotTrajs(batch["trajs"][0], 0, 4, "Trajectories")

        for name in models:
            if is_training[name]:
                models[name].train()

        return plot_manager.getFigure(), loss

    return eval

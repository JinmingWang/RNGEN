from Dataset import LaDeCachedDataset
from TrainEvalTest.GlobalConfigs import *
from TrainEvalTest.DiT.configs import *
from TrainEvalTest.Utils import *
from Models import PathsDiT, CrossDomainVAE
from Diffusion import DDIM

from typing import Callable

import torch

def getEvalFunction(vae: CrossDomainVAE) -> Callable:
    """
    Evaluate the model on the given batch
    :param vae: The VAE model
    :return: The figure and loss
    """
    test_set = LaDeCachedDataset(DATA_DIR, max_trajs=N_TRAJS, set_name="test")
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=B, collate_fn=LaDeCachedDataset.collate_fn, shuffle=False)

    batch = next(iter(test_loader))

    with torch.no_grad():
        latent, _ = vae.encode(batch["paths"])

    latent_noise = torch.randn_like(latent)

    batch["duplicate_segs"] = torch.cat([batch["paths"][..., 1:, :], batch["paths"][..., :-1, :]],
                                        dim=-1)  # (B, N, L-1, 4)
    batch["duplicate_segs"] = batch["duplicate_segs"].view(B, -1, 4).contiguous()  # (B, N_segs, 4)
    filter_mask = ~torch.any(batch["duplicate_segs"] == 0, dim=2, keepdim=True)
    batch["duplicate_segs"] = batch["duplicate_segs"] * filter_mask

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

        loss = torch.nn.functional.mse_loss(duplicate_segs, batch["duplicate_segs"])

        plot_manager = PlotManager(5, 1, 5)
        plot_manager.plotSegments(batch["segs"][0], 0, 0, "Segs")
        plot_manager.plotTrajs(batch["trajs"][0], 0, 1, "Trajectories")
        plot_manager.plotTrajs(batch["paths"][0], 0, 2, "Paths")
        plot_manager.plotSegments(duplicate_segs[0], 0, 3, "Pred Dup Segs")
        plot_manager.plotSegments(coi_means[0], 0, 4, "Pred Segs")

        for name in models:
            if is_training[name]:
                models[name].train()

        return plot_manager.getFigure(), loss

    return eval

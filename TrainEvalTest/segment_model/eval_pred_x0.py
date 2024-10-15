from Dataset import LaDeCachedDataset
from TrainEvalTest.Utils import *
from Models import HungarianLoss_Sequential, SegmentsModel, Encoder
from Diffusion import DDIM

import torch

def eval(batch: Dict[str, Tensor], encoder: Encoder, diffusion_net: SegmentsModel, ddim: DDIM) -> Tuple[plt.Figure, Tensor]:
    """
    Evaluate the model on the given batch
    :param batch: The batch to evaluate
    :param encoder: The encoder model
    :param diffusion_net: The diffusion network model
    :param ddim: The diffusion scheduler
    :return: The figure and loss
    """
    encoder.eval()
    diffusion_net.eval()

    with torch.no_grad():
        traj_enc = encoder(batch["trajs"])
        noise = torch.randn_like(batch["segs"])

        def pred_func(noisy_contents: List[Tensor], t: Tensor) -> Tuple[List[Tensor], List[Tensor]]:
            x0_pred, _ = diffusion_net(*noisy_contents, traj_enc, t)
            noise_pred = torch.randn_like(x0_pred)
            return [x0_pred], [noise_pred]

        segs = ddim.diffusionBackward([noise], pred_func)[0]

    # Remove padding nodes
    valid_mask = segs[:, :, -1] >= 0.5

    pred_segs_xyxy = LaDeCachedDataset.xydl2xyxy(segs)
    segs_xyxy = LaDeCachedDataset.xydl2xyxy(batch["segs"])

    # Get valid nodes and adjacency matrices
    valid_segs = []
    for b in range(batch["segs"].shape[0]):
        valid_segs.append(pred_segs_xyxy[b, :, :-1][valid_mask[b]])

    loss = HungarianLoss_Sequential()(pred_segs_xyxy, segs_xyxy)

    plot_manager = PlotManager(5, 2, 2)

    # Plot reconstructed graph
    plot_manager.plotSegments(batch["graphs"][0], 0, 0, "Ground Truth")
    plot_manager.plotSegments(valid_segs[0].reshape(-1, 2, 2), 0, 1, "Reconstructed")
    plot_manager.plotTrajs(batch["trajs"][0], 1, 0, "Trajectories")

    encoder.train()
    diffusion_net.train()

    return plot_manager.getFigure(), loss

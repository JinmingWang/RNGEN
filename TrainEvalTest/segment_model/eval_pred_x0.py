from Dataset import LaDeCachedDataset
from TrainEvalTest.Utils import *
from Models import HungarianLoss, HungarianMode, SegmentsModel, TrajEncoder
from Diffusion import DDIM

import torch

def eval(batch: Dict[str, Tensor], models: Dict[str, torch.nn.Module], ddim: DDIM) -> Tuple[plt.Figure, Tensor]:
    """
    Evaluate the model on the given batch
    :param batch: The batch to evaluate
    :param encoder: The encoder model
    :param diffusion_net: The diffusion network model
    :param ddim: The diffusion scheduler
    :return: The figure and loss
    """
    is_training = dict()
    for name in models:
        is_training[name] = models[name].training
        models[name].eval()

    with torch.no_grad():
        traj_enc = models["traj_encoder"](batch["trajs"])
        noise = torch.randn_like(batch["graph_enc"])

        def pred_func(noisy_contents: List[Tensor], t: Tensor) -> Tuple[List[Tensor], List[Tensor]]:
            x0_pred = models["DiT"](*noisy_contents, traj_enc, t)
            return [x0_pred], [torch.randn_like(x0_pred)]

        pred_graph_enc = ddim.diffusionBackward([noise], pred_func)[0]

        pred_segs, pred_joints = models["graph_decoder"](pred_graph_enc)

    pred_segs_jointed = matchJoints(pred_segs[0], pred_joints[0])

    loss = HungarianLoss(HungarianMode.Seq)(pred_segs, batch["segs"])

    joints = LaDeCachedDataset.getJointsFromSegments(batch["segs"][0:1])["joints"]

    plot_manager = PlotManager(5, 2, 3)
    plot_manager.plotSegments(batch["segs"][0], 0, 0, "Segs")
    plot_manager.plotSegments(pred_segs[0], 0, 1, f"Pred segs (Loss: {loss.item():.3e})")
    plot_manager.plotSegments(pred_segs_jointed, 0, 2, "Pred segs jointed")
    plot_manager.plotTrajs(batch["trajs"][0], 1, 0, "Trajectories")
    plot_manager.plotHeatmap(joints[0], 1, 1, "Joints")
    plot_manager.plotHeatmap(pred_joints[0], 1, 2, "Pred joints")

    for name in models:
        if is_training[name]:
            models[name].train()

    return plot_manager.getFigure(), loss

import torch.nn as nn
import torch.nn.functional as func


class BiasLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        mask = pred >= target
        return func.mse_loss(pred[mask], target[mask]) + func.mse_loss(pred[~mask], target[~mask]) * 2
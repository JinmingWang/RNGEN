import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from typing import Literal

class HungarianLoss(nn.Module):
    def __init__(self, base_loss: Literal['mse', 'l1'] = 'mse'):
        """
        The difference between this and HungarianLoss_Sequential is that this one is for double sequences.
        The cdist is computed using the first sequence, and both sequence use this cdist to find the best matches.
        :param base_loss:
        """
        super(HungarianLoss, self).__init__()
        self.base_loss = nn.MSELoss() if base_loss == 'mse' else nn.L1Loss()

    def forward(self, pred_seq, target_seq, second_pred, second_target):
        """
        :param pred_seq:
        :param target_seq:
        :param second_pred:
        :param second_target:
        :return:
        """

        B, N, D = pred_seq.shape  # Batch size, number of nodes, feature dimension
        losses = []

        for b in range(B):
            # Step 1: Compute pairwise distance matrix (L2 distance) between tokens in the first sequence
            cost_matrix = torch.cdist(pred_seq[b], target_seq[b], p=2).cpu().detach().numpy()  # (N, N)

            # Step 2: Apply Hungarian algorithm to find best matches
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # Step 3: Compute node feature loss for matched pairs
            loss_first_seq = self.base_loss(pred_seq[b][row_ind], target_seq[b][col_ind])
            loss_second_seq = self.base_loss(second_pred[b][row_ind], second_target[b][col_ind])
            losses.append(loss_first_seq + loss_second_seq)

        return torch.stack(losses).sum()
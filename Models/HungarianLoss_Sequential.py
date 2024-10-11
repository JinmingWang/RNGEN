import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from typing import Literal

class HungarianLoss(nn.Module):
    def __init__(self, base_loss: Literal['mse', 'l1'] = 'mse'):
        """
        Hungarian loss for token-level sequence matching

        For two sequences of tokens with length N, we cannot guarantee the order of tokens.

        For example, if we have two sequences of tokens:
        pred = [A, B, C]
        target = [B, A, C]

        Then these two sequences should be considered as the same sequence.

        To represent a graph, we have to use a sequence of nodes / edges / line segments.

        However, there is no good way to order the nodes / edges / line segments.

        So we give up the ordering, and match them first using the Hungarian algorithm.

        Then we compute the loss for matched pairs.

        :param base_loss: Base loss function to compute loss for matched pairs
        """
        super(HungarianLoss, self).__init__()
        self.base_loss = nn.MSELoss() if base_loss == 'mse' else nn.L1Loss()

    def forward(self, pred_nodes, target_nodes, pred_adj, target_adj):
        B, N, D = pred_nodes.shape  # Batch size, number of nodes, feature dimension
        losses = []

        for b in range(B):
            # Step 1: Compute pairwise distance matrix (L2 distance) between nodes
            cost_matrix = torch.cdist(pred_nodes[b], target_nodes[b], p=2).cpu().detach().numpy()  # (N, N)

            # Step 2: Apply Hungarian algorithm to find best node matches
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # Step 3: Compute node feature loss for matched pairs
            losses.append(self.base_loss(pred_nodes[b][row_ind], target_nodes[b][col_ind]))

        return torch.stack(losses).sum()
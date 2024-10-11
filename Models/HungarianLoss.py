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
        edge_losses = []

        for b in range(B):
            # Step 1: Compute pairwise distance matrix (L2 distance) between nodes
            cost_matrix = torch.cdist(pred_nodes[b], target_nodes[b], p=2).cpu().detach().numpy()  # (N, N)

            # Step 2: Apply Hungarian algorithm to find best node matches
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # Step 3: Compute node feature loss for matched pairs
            node_loss = self.base_loss(pred_nodes[b][row_ind], target_nodes[b][col_ind])
            losses.append(node_loss)

            if pred_adj is not None and target_adj is not None:
                # Step 4: Reorder the adjacency matrix according to the node matching
                pred_adj_matched = pred_adj[b][row_ind][:, row_ind]  # (N, N)
                target_adj_matched = target_adj[b][col_ind][:, col_ind]  # (N, N)

                # Step 5: Compute edge loss for the matched adjacency matrices
                edge_loss = self.base_loss(pred_adj_matched, target_adj_matched)
                edge_losses.append(edge_loss)

        # Combine node and edge losses
        node_loss = torch.stack(losses).sum()
        edge_loss = torch.stack(edge_losses).sum()  # Combine edge and node losses

        return node_loss, edge_loss
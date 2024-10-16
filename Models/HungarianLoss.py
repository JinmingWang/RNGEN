import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from typing import Literal, List
from enum import Enum

class HungarianMode(Enum):
    Seq = 0
    DoubleSeq = 1
    SeqMat = 2


class HungarianLoss(nn.Module):
    def __init__(self, mode: HungarianMode, base_loss: Literal['mse', 'l1'] = 'mse', feature_weight: List[float] = None):
        """
        The difference between this and HungarianLoss_Sequential is that this one is for double sequences.
        The cdist is computed using the first sequence, and both sequence use this cdist to find the best matches.
        :param base_loss:
        """
        super(HungarianLoss, self).__init__()
        self.base_loss = nn.MSELoss() if base_loss == 'mse' else nn.L1Loss()
        self.mode = mode

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if feature_weight is None:
            self.f_w = torch.ones(1, 1, device=device, dtype=torch.float32)
        else:
            self.f_w = torch.tensor(feature_weight, device=device, dtype=torch.float32).view(1, -1)

    def forward_SeqMat(self, pred_nodes, target_nodes, pred_adj, target_adj):
        B, N, D = pred_nodes.shape  # Batch size, number of nodes, feature dimension
        losses = []
        edge_losses = []

        for b in range(B):
            # Step 1: Compute pairwise distance matrix (L2 distance) between nodes
            cost_matrix = torch.cdist(pred_nodes[b] * self.f_w,
                                      target_nodes[b] * self.f_w, p=2).cpu().detach().numpy()  # (N, N)

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

    def forward_DoubleSeq(self, pred_seq, target_seq, second_pred, second_target):
        """
        :param pred_seq:
        :param target_seq:
        :param second_pred:
        :param second_target:
        :param channel_dist_weight:
        :return:
        """

        B, N, D = pred_seq.shape  # Batch size, number of nodes, feature dimension
        losses = []

        for b in range(B):
            # Step 1: Compute pairwise distance matrix (L2 distance) between tokens in the first sequence
            cost_matrix = torch.cdist(pred_seq[b] * self.f_w,
                                      target_seq[b] * self.f_w, p=2).cpu().detach().numpy()  # (N, N)

            # Step 2: Apply Hungarian algorithm to find best matches
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # Step 3: Compute node feature loss for matched pairs
            loss_first_seq = self.base_loss(pred_seq[b][row_ind], target_seq[b][col_ind])
            loss_second_seq = self.base_loss(second_pred[b][row_ind], second_target[b][col_ind])
            losses.append(loss_first_seq + loss_second_seq)

        return torch.stack(losses).sum()

    def forward_Seq(self, pred_nodes, target_nodes):
        B, N, D = pred_nodes.shape  # Batch size, number of nodes, feature dimension
        losses = []

        for b in range(B):
            # Step 1: Compute pairwise distance matrix (L2 distance) between nodes
            cost_matrix = torch.cdist(pred_nodes[b] * self.f_w,
                                      target_nodes[b] * self.f_w, p=2).cpu().detach().numpy()  # (N, N)

            # Step 2: Apply Hungarian algorithm to find best node matches
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            # Step 3: Compute node feature loss for matched pairs
            losses.append(self.base_loss(pred_nodes[b][row_ind], target_nodes[b][col_ind]))

        return torch.stack(losses).sum()

    def forward(self, *args, **kwargs):
        match (self.mode):
            case HungarianMode.Seq:
                return self.forward_Seq(*args, **kwargs)
            case HungarianMode.SeqMat:
                return self.forward_SeqMat(*args, **kwargs)
            case HungarianMode.DoubleSeq:
                return self.forward_DoubleSeq(*args, **kwargs)
            case _:
                print("Invalid mode")
                return None
from time import time
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

def compute_loss(pred, target, penalty_weight=1.0):
    M, D = pred.shape
    N, _ = target.shape

    start = time()

    # Step 1: Compute pairwise distance matrix (L2 distance)
    cost_matrix = torch.cdist(pred, target, p=2)

    # Step 2: Apply Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix.cpu().detach().numpy())

    # Step 3: Compute loss for matched pairs
    matched_loss = F.mse_loss(pred[row_ind], target[col_ind], reduction='sum')

    # Step 4: Compute penalty for unmatched items
    unmatched_pred = set(range(M)) - set(row_ind)
    unmatched_target = set(range(N)) - set(col_ind)

    # for each unmatched item, their squared L2 distance is added to the penalty
    penalty = 0
    if unmatched_pred:
        penalty += torch.sum(pred[list(unmatched_pred)] ** 2)
    if unmatched_target:
        penalty += torch.sum(target[list(unmatched_target)] ** 2)

    penalty = penalty_weight * penalty

    # Final loss
    total_loss = matched_loss + penalty

    # Prepare matchings list
    matchings = list(zip(row_ind, col_ind))

    end = time()
    print(f"Time taken: {end - start:.6f} seconds")

    return total_loss, matchings


# Visualization with Matplotlib
def visualize_matchings(pred, target, matchings):
    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()

    plt.figure(figsize=(8, 8))

    # Plot the pred and target points
    plt.scatter(pred_np[:, 0], pred_np[:, 1], c='red', label='Pred', marker='x')
    plt.scatter(target_np[:, 0], target_np[:, 1], c='blue', label='Target', marker='o')

    # Draw lines for the matchings
    for id_src, id_dst in matchings:
        # dotted lines for matched pairs
        plt.plot([pred_np[id_src, 0], target_np[id_dst, 0]],
                 [pred_np[id_src, 1], target_np[id_dst, 1]], c='gray', linestyle='dotted')

    plt.legend()
    plt.title('Pred and Target Points with Hungarian Matchings')
    plt.show()


# Example usage
def test_case():
    # Test case with D=2
    N = 32
    M = 32
    pred = torch.randn(M, 2)
    target = torch.linspace(-2, 2, N * 2).view(-1, 2) + torch.randn(N, 2) * 0.1

    # Compute loss and get matchings
    loss, matchings = compute_loss(pred, target, penalty_weight=1.0)

    print(f"Computed Loss: {loss.item()}")
    print(f"Matchings: {matchings}")

    # Visualize the matchings
    visualize_matchings(pred, target, matchings)


# Run the test case
test_case()
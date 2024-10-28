import torch

from .Basics import *

class PathEncoder(nn.Module):
    def __init__(self, N_trajs: int, L_traj: int, L_path: int, d_encode: int, n_layers: int):
        super().__init__()

        self.N_trajs = N_trajs
        self.L_traj = L_traj
        self.L_path = L_path
        self.n_layers = n_layers
        self.d_encode = d_encode

        # (BN, C=8, L=64)

        self.stages = nn.Sequential(
            Rearrange("B N L C", "(B N) C L"),
            nn.Conv1d(2, 32, 3, 1, 1),
            Swish(),
            nn.Conv1d(32, 64, 3, 1, 1),
            Swish(),
            nn.Conv1d(64, 64, 3, 1, 1),
            Rearrange("(B N) C L", "B (N L) C", N=N_trajs),
            *[AttentionBlock(64, 64, 256, 64, 4) for _ in range(n_layers)],
            nn.Linear(64, d_encode)
        )

    def forward(self, x):
        return self.stages(x)

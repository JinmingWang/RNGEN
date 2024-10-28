import torch

from .Basics import *

class TrajEncoder(nn.Module):
    def __init__(self, N_trajs: int, L_traj: int, L_path: int, d_encode: int, n_layers: int):
        super().__init__()

        self.N_trajs = N_trajs
        self.L_traj = L_traj
        self.L_path = L_path
        self.n_layers = n_layers
        self.d_encode = d_encode

        # input: (B, N, L, C=2)
        self.resnet = nn.Sequential(
            Rearrange("B N L C", "(B N) C L"),
            Conv1dBnAct(2, 16, 3, 1, 1),
            Conv1dBnAct(16, 64, 3, 1, 1),

            Res1D(64, 128, 64),
            Res1D(64, 128, 64),
            Res1D(64, 128, 64),
            Conv1dBnAct(64, 128, 3, 1, 1),
            nn.MaxPool1d(2),

            Res1D(128, 256, 128),
            Res1D(128, 256, 128),
            Res1D(128, 256, 128),
            Conv1dBnAct(128, 256, 3, 1, 1),
            nn.MaxPool1d(2),

            nn.Conv1d(256, 128, 3, 1, 1),
            # (BN, C=8, L=32)
        )

        # (BN, C=8, L=64)

        self.stages = nn.Sequential(
            Rearrange("(B N) C L", "B (N L) C", N=N_trajs),
            *[AttentionBlock(128, 64, 256, 128, 8) for _ in range(n_layers)],
        )

        self.head = nn.Linear(128, d_encode)

    def forward(self, x):
        x = self.resnet(x)
        x = self.stages(x)
        return self.head(x)

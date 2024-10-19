from .Basics import *

# This encoder splits the trajectories into sub-trajectories

class TrajEncoder(nn.Module):
    def __init__(self, N_trajs: int, L_traj: int, D_encode: int):
        super().__init__()

        self.D_token = L_traj * 2
        self.D_subtoken = D_encode

        # input: (B, N, L, C=2)
        self.layers = nn.Sequential(
            Rearrange("B N L C", "(B N) C L"),
            nn.Conv1d(2, 32, 3, 1, 1),
            Swish(),

            Res1D(32, 128, 32),
            Res1D(32, 128, 32),
            nn.Conv1d(32, 64, 3, 2, 1),     # (BN, 64, 32)
            Swish(),

            Res1D(64, 256, 64),
            Res1D(64, 256, 64),
            nn.Conv1d(64, 128, 3, 2, 1),  # (BN, 128, 16)

            Rearrange("(B N) C L", "B (N L) C", N=N_trajs),
            *[AttentionBlock(d_in=128, d_out=128, d_head=128, n_heads=8, d_expand=512) for _ in range(8)],
            nn.Linear(128, D_encode),
        )

    def forward(self, x):
        return self.layers(x)

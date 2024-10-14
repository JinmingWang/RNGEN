from .Basics import *

# This encoder splits the trajectories into sub-trajectories

class Encoder(nn.Module):
    def __init__(self, N_trajs: int, L_traj: int, L_subtraj: int, D_encode: int):
        super().__init__()

        self.D_token = L_traj * 2
        self.L_subtraj = L_subtraj
        self.D_subtoken = 4 * L_subtraj

        # input: (B, N, L, C=2)
        self.layers = nn.Sequential(
            Rearrange("B N L C", "(B N) C L"), # (B*N, C, L)
            Conv1dBnAct(2, 16, 5, 1, 2),  # (B*N, C=8, L)
            Conv1dBnAct(16, 32, 5, 1, 2),  # (B*N, C=16, L)
            Conv1dBnAct(32, 4, 5, 1, 2),  # (B*N, C=32, L)
            Rearrange("(B N) C L", "B N (C L)", N=N_trajs),  # (B, N, 32*L)
            # nn.Flatten(2),  # (B, N, L*C)
            nn.Conv1d(N_trajs, N_trajs*self.D_subtoken, kernel_size=self.D_subtoken, stride=self.D_subtoken, groups=N_trajs),
            # (B, N*D_subtoken, L_subtoken)
            Rearrange("B (N D) S", "B (N S) D", N=N_trajs, D=self.D_subtoken),  # (B, N*L_subtoken, D_subtoken)
            *[AttentionBlock(d_in=self.D_subtoken, d_out=self.D_subtoken, d_head=self.D_subtoken, n_heads=8,
                             d_expand=256) for _ in range(8)],
            nn.Linear(self.D_subtoken, D_encode),
        )

        self.projs = nn.Sequential(

            nn.Linear(self.D_subtoken, D_encode),
        )

    def forward(self, x):
        B, N, L, C = x.shape
        return self.layers(x)

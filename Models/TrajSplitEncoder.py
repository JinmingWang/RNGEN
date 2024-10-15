from .Basics import *

# This encoder splits the trajectories into sub-trajectories

class Encoder(nn.Module):
    def __init__(self, N_trajs: int, L_traj: int, D_encode: int):
        super().__init__()

        self.D_token = L_traj * 2
        self.D_subtoken = D_encode

        # input: (B, N, L, C=2)
        self.layers = nn.Sequential(
            Rearrange("B N L C", "(B N) C L"),
            Res1D(2, 64, 32),
            Res1D(32, 64, 2),
            Rearrange("(B N) C L", "B N (L C)", N=N_trajs),
            # Rearrange("B N L C", "B N (L C)"),
            nn.Conv1d(N_trajs, N_trajs*self.D_subtoken, kernel_size=self.D_subtoken, stride=self.D_subtoken//2, groups=N_trajs),
            # (B, N*D_subtoken, L_subtoken)
            Rearrange("B (N D) S", "B (N S) D", N=N_trajs, D=self.D_subtoken),
            *[AttentionBlock(d_in=D_encode, d_out=D_encode, d_head=D_encode, n_heads=12, d_expand=256) for _ in
              range(10)]
        )

    def forward(self, x):
        return self.layers(x)

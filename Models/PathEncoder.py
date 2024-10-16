from .Basics import *

# This encoder splits the trajectories into sub-trajectories
class Stage(nn.Sequential):
    def __init__(self, N_trajs: int, L_traj: int, L_segment: int, D_token: int, downsample: bool = False):
        self.N_trajs = N_trajs
        self.L_traj = L_traj
        self.L_segment = L_segment
        self.D_token = D_token
        super().__init__(
            Rearrange("(B N) C L", "B N (L C)", N=N_trajs),
            # The design: the idea is that each head will focus on a segment of the trajectory
            AttentionBlock(d_in=L_traj * D_token, d_out=L_traj * D_token, d_head=L_segment * D_token,
                           n_heads=L_traj // L_segment, d_expand=256),
            AttentionBlock(d_in=L_traj * D_token, d_out=L_traj * D_token, d_head=L_segment * D_token,
                           n_heads=L_traj // L_segment, d_expand=256),

            Rearrange("B N (L C)", "(B N) C L", L=L_traj),
            Res1D(D_token, D_token * 4, D_token),
            Res1D(D_token, D_token * 4, D_token),

            nn.Conv1d(D_token, D_token * (1 + int(downsample)), 3, 1 + int(downsample), 1),
        )


class PathEncoder(nn.Module):
    def __init__(self, N_trajs: int, L_traj: int, L_path: int):
        super().__init__()

        self.N_trajs = N_trajs
        self.L_traj = L_traj
        self.L_path = L_path

        # input: (B, N, L, C=2)
        self.s0 = nn.Sequential(
            Rearrange("B N L C", "(B N) C L"),
            Res1D(2, 64, 16),
            Res1D(16, 64, 16),
            nn.Conv1d(16, 16, 3, 2, 1),
            # (BN, C=8, L=32)
        )

        # (BN, C=8, L=64)

        self.s1 = Stage(N_trajs, L_traj // 2, 4, 16, True)     # (BN, C=16, L=16)

        self.s2 = Stage(N_trajs, L_traj // 4, 4, 32)     # (BN, C=16, L=16)

        self.head = nn.Sequential(
            Rearrange("(B N) C L", "B N (L C)", N=N_trajs),     # (B, N, LC=512)
            AttentionBlock(d_in=L_traj * 8, d_out=256, d_head=64, n_heads=8, d_expand=256),
            AttentionBlock(d_in=256, d_out=64, d_head=64, n_heads=8, d_expand=256),
            AttentionBlock(d_in=64, d_out=L_path * 2, d_head=64, n_heads=8, d_expand=256),
            nn.Linear(L_path * 2, L_path * 2),
            Rearrange("B N (L C)", "B N L C", L=L_path),
        )

    def forward(self, x):
        x = self.s0(x)
        x = self.s1(x)
        x = self.s2(x)
        x = self.head(x)
        return x

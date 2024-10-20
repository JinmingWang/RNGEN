from .Basics import *

# This encoder splits the trajectories into sub-trajectories
class Stage(nn.Sequential):
    def __init__(self, N_trajs: int, L_traj: int, L_segment: int, D_token: int, D_out: int, downsample: bool = False):
        self.N_trajs = N_trajs
        self.L_traj = L_traj
        self.L_segment = L_segment
        self.D_token = D_token
        self.D_out = D_out
        super().__init__(
            Rearrange("(B N) C L", "B N (L C)", N=N_trajs),
            # The design: the idea is that each head will focus on a segment of the trajectory
            AttentionBlock(d_in=L_traj * D_token, d_out=L_traj * D_token, d_head=L_segment * D_token,
                           n_heads=L_traj // L_segment, d_expand=256),
            AttentionBlock(d_in=L_traj * D_token, d_out=L_traj * D_token, d_head=L_segment * D_token,
                           n_heads=L_traj // L_segment, d_expand=256),

            Rearrange("B N (L C)", "(B N) C L", L=L_traj),
            Res1D(D_token, 128, D_token),
            Res1D(D_token, 128, D_out),

            nn.Conv1d(D_out, D_out, 3, 1 + int(downsample), 1),
        )


class PathEncoder(nn.Module):
    def __init__(self, N_trajs: int, L_traj: int, L_path: int, d_encode: int, get_path: bool = False):
        super().__init__()

        self.N_trajs = N_trajs
        self.L_traj = L_traj
        self.L_path = L_path
        self.get_path = get_path
        self.d_encode = d_encode

        # input: (B, N, L, C=2)
        self.s0 = nn.Sequential(
            Rearrange("B N L C", "(B N) C L"),
            Res1D(2, 128, 32),
            Res1D(32, 128, 64),
            nn.Conv1d(64, 16, 3, 2, 1),
            # (BN, C=8, L=32)
        )

        # (BN, C=8, L=64)

        self.stages = nn.Sequential(
            Stage(N_trajs, L_traj // 2, 4, 16, 16),
            Stage(N_trajs, L_traj // 2, 4, 16, 16),
            Stage(N_trajs, L_traj // 2, 4, 16, 16),
            Stage(N_trajs, L_traj // 2, 4, 16, d_encode, downsample=True),
        )

        if self.get_path:
            # L*C = L_traj // 4 * 32 = L_traj * 8
            # if L_Traj = 64, then L*C = 512
            self.head = nn.Sequential(
                Rearrange("(B N) C L", "B N (L C)", N=N_trajs),  # (B, N, LC=512)
                nn.Linear(L_traj * 8, 128),
                AttentionBlock(d_in=128, d_out=64, d_head=64, n_heads=8, d_expand=256),
                AttentionBlock(d_in=64, d_out=L_path * 2, d_head=64, n_heads=8, d_expand=256),
                nn.Linear(L_path * 2, L_path * 2),
                Rearrange("B N (L C)", "B N L C", L=L_path),
            )
        else:
            self.head = nn.Sequential(
                Rearrange("(B N) C L", "B (N L) C", N=N_trajs),
            )

    def forward(self, x):
        x = self.s0(x)
        x = self.stages(x)
        return self.head(x)

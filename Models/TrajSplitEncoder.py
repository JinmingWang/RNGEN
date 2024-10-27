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


class TrajSplitEncoder(nn.Module):
    def __init__(self, N_trajs: int, L_traj: int, L_path: int, d_encode: int):
        super().__init__()

        self.N_trajs = N_trajs
        self.L_traj = L_traj
        self.L_path = L_path
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
            nn.Conv1d(256, 256, 3, 1, 1),
            # (BN, C=8, L=32)
        )

        # (BN, C=8, L=64)

        self.attentions = nn.Sequential(
            Rearrange("(B N) C L", "B (N L) C", N=N_trajs),
            AttentionBlock(256, 64, 512, 256, 8),
            AttentionBlock(256, 64, 512, 256, 8),
            AttentionBlock(256, 64, 512, 256, 8),
            AttentionBlock(256, 64, 512, 256, 8),
            AttentionBlock(256, 64, 512, 256, 8),
            AttentionBlock(256, 64, 512, 256, 8),
            AttentionBlock(256, 64, 512, 256, 8),
            AttentionBlock(256, 64, 512, 256, 8),

        )

        self.head = nn.Linear(256, d_encode)

    def forward(self, x):
        x = self.resnet(x)
        x = self.attentions(x)
        return self.head(x)

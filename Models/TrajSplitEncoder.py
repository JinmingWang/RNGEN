from .Basics import *

# This encoder splits the trajectories into sub-trajectories

class Encoder(nn.Module):
    def __init__(self, N_trajs: int, L_traj: int, D_encode: int):
        super().__init__()

        self.D_token = L_traj * 2
        self.D_subtoken = D_encode

        # input: (B, N, L, C=2)
        self.splitter = nn.Sequential(
            nn.Flatten(2),  # (B, N, L*C)
            nn.Conv1d(N_trajs, N_trajs*self.D_subtoken, kernel_size=self.D_subtoken, stride=self.D_subtoken//2, groups=N_trajs),
            # (B, N*D_subtoken, L_subtoken)
        )

        D_hidden = D_encode * 2
        self.projs = nn.Sequential(
            nn.Linear(D_encode, D_hidden),
            *[AttentionBlock(d_in=D_hidden, d_out=D_hidden, d_head=D_hidden, n_heads=8, d_expand=256) for _ in range(8)],
            nn.Linear(D_hidden, D_encode)
        )

    def forward(self, x):
        B, N, L, C = x.shape
        x = rearrange(self.splitter(x), "B (N D) S -> B (N S) D", N=N, D=self.D_subtoken)
        # (B, N*L_subtoken, D_subtoken) == (B, N_subtrajs, D_subtraj)
        return self.projs(x)
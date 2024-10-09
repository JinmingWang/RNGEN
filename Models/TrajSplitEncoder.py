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

        self.proj_1 = AttentionBlock(d_in=D_encode, d_out=D_encode, d_head=D_encode, n_heads=4, d_expand=D_encode * 4)
        self.proj_2 = AttentionBlock(d_in=D_encode, d_out=D_encode, d_head=D_encode, n_heads=4, d_expand=D_encode * 4)


    def forward(self, x):
        B, N, L, C = x.shape
        x = rearrange(self.splitter(x), "B (N D) S -> B (N S) D", N=N, D=self.D_subtoken)
        x = self.proj_1(x)
        x = self.proj_2(x)
        return x
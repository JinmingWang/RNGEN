from .Basics import *

class Encoder(nn.Module):
    def __init__(self, N_trajs: int, traj_len: int, encode_c: int):
        super().__init__()

        self.layer = nn.Identity() if traj_len * 2 == encode_c else nn.Linear(traj_len * 2, encode_c)


    def forward(self, x):
        B, N, L, C = x.shape
        return self.layer(rearrange(x, "B N L C -> B N (L C)"))
import torch

from .Basics import *


class ConvSplitAttn(nn.Module):
    def __init__(self, n_trajs: int, d_in: int, d_expand: int, d_out: int, n_splits: int, n_heads: int, dropout: float=0.1):
        super(ConvSplitAttn, self).__init__()

        # in shape: (BN, C, L)
        # Q shape: (B*H, N, head_c)
        # K shape: (B*H, N, head_c)
        # V shape: (B*H, N, in_c)
        self.N = n_trajs
        self.S = n_splits
        self.H = n_heads
        self.d_split = d_in // n_splits
        self.d_in = d_in
        self.dropout = nn.Dropout(dropout)

        self.d_q = d_in * n_heads
        self.d_k = d_in * n_heads
        self.d_v = d_in * n_heads
        d_qkv = self.d_q + self.d_k + self.d_v

        self.qkv_proj = nn.Sequential(
            nn.GroupNorm(8, d_in),
            nn.Conv1d(d_in, d_qkv, 3, 1, 1),
        )

        self.merge_head_proj = nn.Conv1d(d_in * n_heads, d_in, 3, 1, 1)

        torch.nn.init.zeros_(self.merge_head_proj.weight)
        torch.nn.init.zeros_(self.merge_head_proj.bias)

        self.ff = nn.Sequential(
            nn.GroupNorm(8, d_in),
            nn.Conv1d(d_in, d_expand, 3, 1, 1),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv1d(d_expand, d_out, 3, 1, 1),
        )

        torch.nn.init.zeros_(self.ff[-1].weight)
        torch.nn.init.zeros_(self.ff[-1].bias)

        self.shortcut = nn.Conv1d(d_in, d_out, 1) if d_in != d_out else nn.Identity()

    def forward(self, x):

        qkv = self.qkv_proj(x)
        # This looks complicated
        # The input x is of shape (BN, C, L), here batch size B and number of trajectories N are merged
        # After qkv_proj, qkv is of shape (BN, 3HC, L), here 3 is because we have q, k, v, and H is the number of heads
        # The last dimension, we split L to (L, S), the new L is the number of splits, and S is the length of each split
        qkv = rearrange(qkv, '(B N) (T H C) (L S) -> T (B H) (N S) (C L)', N=self.N, T=3, H=self.H, S=self.S)
        q, k, v = torch.unbind(qkv, dim=0)  # Each of shape (BH, NS, CL)

        # attn shape: (B*H, N, N)
        attn = 1 / torch.cdist(q, k, p=2)    # why not use softmax? because one token can attend to multiple tokens
        attn = self.dropout(attn)

        # out shape: (B, N, H*in_c)
        out = rearrange(torch.bmm(attn, v), '(B H) (N S) (C L) -> (B N) (H C) (L S)', H=self.H, N=self.N, S=self.S, C=self.d_in)
        x = x + self.merge_head_proj(out)

        return self.ff(x) + self.shortcut(x)


# This encoder splits the trajectories into sub-trajectories
class Stage(nn.Sequential):
    def __init__(self, N_trajs: int, D_token: int, n_splits: int, n_heads:int, downsample: bool = False):
        self.N_trajs = N_trajs
        self.D_token = D_token
        self.n_splits = n_splits
        self.n_heads = n_heads

        super().__init__(
            ConvSplitAttn(N_trajs, D_token, D_token * 4, D_token, n_splits, n_heads),
            ConvSplitAttn(N_trajs, D_token, D_token * 4, D_token, n_splits, n_heads),
            ConvSplitAttn(N_trajs, D_token, D_token * 4, D_token, n_splits, n_heads),
            nn.Conv1d(D_token, D_token * (1 + int(downsample)), 3, 1 + int(downsample), 1)
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
            Res1D(2, 64, 32),
            Res1D(32, 128, 64),
            # (BN, C=32, L=64)
        )

        self.s1 = Stage(N_trajs, 64, 16, 4, True)     # (BN, C=128, L=32)

        self.s2 = Stage(N_trajs, 128, 8, 4, True)     # (BN, C=256, L=16)

        self.head = nn.Sequential(
            Rearrange("(B N) C L", "B N (L C)", N=N_trajs),     # (B, N, LC=512)
            nn.Linear(L_traj * 64, 1024),
            Swish(),
            nn.Linear(1024, 512),
            AttentionBlock(d_in=512, d_out=128, d_head=64, n_heads=8, d_expand=256),
            AttentionBlock(d_in=128, d_out=32, d_head=64, n_heads=8, d_expand=256),
            AttentionBlock(d_in=32, d_out=L_path * 2, d_head=64, n_heads=8, d_expand=256),
            nn.Linear(L_path * 2, L_path * 2),
            Rearrange("B N (L C)", "B N L C", L=L_path),
        )

    def forward(self, x):
        x = self.s0(x)
        x = self.s1(x)
        x = self.s2(x)
        print(1, x.shape)
        x = self.head(x)
        return x

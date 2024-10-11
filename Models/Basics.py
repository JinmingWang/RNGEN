import torch
import torch.nn as nn
import torch.nn.functional as func
from einops import rearrange
from typing import List, Literal, Tuple

Tensor = torch.Tensor

class AttentionBlock(nn.Module):
    def __init__(self, d_in: int, d_head: int, d_expand: int, d_out: int, n_heads: int, dropout: float=0.1):
        super(AttentionBlock, self).__init__()

        # in shape: (B, N, in_c)
        # Q shape: (B*H, N, head_c)
        # K shape: (B*H, N, head_c)
        # V shape: (B*H, N, in_c)
        self.H = n_heads
        self.d_head = d_head
        self.d_in = d_in
        self.scale = d_head ** -0.5
        self.dropout = nn.Dropout(dropout)

        d_qkv = d_head * n_heads * 2 + d_in * n_heads

        self.qkv_proj = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_qkv),
        )

        self.merge_head_proj = nn.Linear(d_in * self.H, d_in)

        torch.nn.init.zeros_(self.merge_head_proj.weight)
        torch.nn.init.zeros_(self.merge_head_proj.bias)

        self.ff = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_expand),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_expand, d_out),
        )

        torch.nn.init.zeros_(self.ff[-1].weight)
        torch.nn.init.zeros_(self.ff[-1].bias)

        self.shortcut = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()

    def forward(self, x):
        B, N, in_c = x.shape

        q, k, v = self.qkv_proj(x).split([self.d_head * self.H, self.d_head * self.H, in_c * self.H], dim=-1)
        q = rearrange(q, 'B N (H C) -> (B H) N C', H=self.H)    # (B*H, N, head_c)
        kt = rearrange(k, 'B N (H C) -> (B H) C N', H=self.H)   # (B*H, head_c, N)
        v = rearrange(v, 'B N (H C) -> (B H) N C', H=self.H)    # (B*H, N, in_c)

        # attn shape: (B*H, N, N)
        attn = torch.bmm(q, kt) * self.scale
        attn = func.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # out shape: (B, N, H*in_c)
        out = rearrange(torch.bmm(attn, v), '(B H) N C -> B N (H C)', H=self.H, C=in_c)
        x = x + self.merge_head_proj(out)

        return self.ff(x) + self.shortcut(x)


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_in: int, d_context: int, d_head: int, d_expand: int, d_out: int, n_heads: int, dropout: float=0.1):
        super(CrossAttentionBlock, self).__init__()

        # in shape: (B, N, in_c)
        # Q shape: (B*H, N, head_c)
        # K shape: (B*H, N, head_c)
        # V shape: (B*H, N, in_c)
        self.H = n_heads
        self.d_head = d_head
        self.d_in = d_in
        self.scale = d_head ** -0.5
        self.dropout = nn.Dropout(dropout)

        self.q_proj = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_head * n_heads),
        )

        self.kv_proj = nn.Sequential(
            nn.LayerNorm(d_context),
            nn.Linear(d_context, (d_head + d_in) * n_heads),
        )

        self.merge_head_proj = nn.Linear(d_in * self.H, d_in)

        self.ff = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_expand),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_expand, d_out),
        )

        self.shortcut = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()

    def forward(self, x, context):
        B, N, in_c = x.shape

        q = rearrange(self.q_proj(x), 'B N (H C) -> (B H) N C', H=self.H)    # (B*H, N, head_c)

        k, v = self.kv_proj(context).split([self.d_head * self.H, in_c * self.H], dim=-1)
        kt = rearrange(k, 'B N (H C) -> (B H) C N', H=self.H)   # (B*H, head_c, N)
        v = rearrange(v, 'B N (H C) -> (B H) N C', H=self.H)    # (B*H, N, in_c)

        # attn shape: (B*H, N, N)
        attn = torch.bmm(q, kt) * self.scale
        attn = func.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # out shape: (B, N, H*in_c)
        out = rearrange(torch.bmm(attn, v), '(B H) N C -> B N (H C)', H=self.H, C=in_c)
        x = x + self.merge_head_proj(out)

        return self.ff(x) + self.shortcut(x)


class AttentionWithTime(nn.Module):
    def __init__(self, d_in: int, d_head: int, d_expand: int, d_out: int, d_time: int, n_heads: int, dropout: float=0.1):
        super(AttentionWithTime, self).__init__()

        # in shape: (B, N, in_c)
        # Q shape: (B*H, N, head_c)
        # K shape: (B*H, N, head_c)
        # V shape: (B*H, N, in_c)
        self.H = n_heads
        self.d_head = d_head
        self.d_in = d_in
        self.scale = d_head ** -0.5
        self.d_time = d_time
        self.dropout = nn.Dropout(dropout)

        d_qkv = d_head * n_heads * 2 + d_in * n_heads

        self.qkv_proj = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_qkv),
        )

        self.merge_head_proj = nn.Linear(d_in * self.H, d_in)

        self.time_proj = nn.Linear(d_time, d_time)

        self.ff = nn.Sequential(
            nn.LayerNorm(d_in + d_time),
            nn.Linear(d_in + d_time, d_expand),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_expand, d_out),
        )

        self.shortcut = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()

    def forward(self, x, t):
        B, N, in_c = x.shape

        q, k, v = self.qkv_proj(x).split([self.d_head * self.H, self.d_head * self.H, in_c * self.H], dim=-1)
        q = rearrange(q, 'B N (H C) -> (B H) N C', H=self.H)    # (B*H, N, head_c)
        kt = rearrange(k, 'B N (H C) -> (B H) C N', H=self.H)   # (B*H, head_c, N)
        v = rearrange(v, 'B N (H C) -> (B H) N C', H=self.H)    # (B*H, N, in_c)

        # attn shape: (B*H, N, N)
        attn = torch.bmm(q, kt) * self.scale
        attn = func.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # out shape: (B, N, H*in_c)
        out = rearrange(torch.bmm(attn, v), '(B H) N C -> B N (H C)', H=self.H, C=in_c)
        x = x + self.merge_head_proj(out)

        t = self.time_proj(t).view(B, 1, self.d_time).expand(-1, N, -1)     # (B, N, time_c)
        return self.shortcut(x) + self.ff(torch.cat([x, t], dim=-1))


class Transpose(nn.Module):
    def __init__(self, dim0: int, dim1: int):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)


class Res2D(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(d_in, d_in * 2, 3, 1, 1),
            nn.BatchNorm2d(d_in * 2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(d_in * 2, d_in * 2, 3, 1, 1),
            nn.BatchNorm2d(d_in * 2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(d_in * 2, d_out, 3, 1, 1)
        )

        self.shortcut = nn.Identity() if d_in == d_out else nn.Conv2d(d_in, d_out, 1, 1, 0)

    def forward(self, x):
        return self.shortcut(x) + self.layers(x)


class Res1D(nn.Module):
    def __init__(self, d_in: int, d_mid: int, d_out: int):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv1d(d_in, d_mid, 3, 1, 1),
            nn.BatchNorm1d(d_mid),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(d_mid, d_mid, 3, 1, 1),
            nn.BatchNorm1d(d_mid),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(d_mid, d_out, 3, 1, 1)
        )

        self.shortcut = nn.Identity() if d_in == d_out else nn.Conv1d(d_in, d_out, 1, 1, 0)

    def forward(self, x):
        return self.shortcut(x) + self.layers(x)
class Conv2dInAct(nn.Sequential):
    def __init__(self, d_in: int, d_out: int, k: int, s: 1, p: 0):
        super().__init__(
            nn.Conv2d(d_in, d_out, k, s, p),
            nn.InstanceNorm2d(d_out),
            nn.LeakyReLU(inplace=True)
        )


class Conv1dBnAct(nn.Sequential):
    def __init__(self, d_in: int, d_out: int, k: int, s: 1, p: 0):
        super().__init__(
            nn.Conv1d(d_in, d_out, k, s, p),
            nn.BatchNorm1d(d_out),
            nn.LeakyReLU(inplace=True)
        )


class SequentialWithAdditionalInputs(nn.Sequential):
    def forward(self, x, *args):
        for module in self:
            x = module(x, *args)
        return x
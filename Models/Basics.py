import torch
import torch.nn as nn
import torch.nn.functional as func
from einops import rearrange
from typing import List, Literal, Tuple

Tensor = torch.Tensor


class Swish(nn.Module):
    def forward(self, x):
        return x * func.sigmoid(x)


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

        self.d_q = d_head * n_heads
        self.d_k = d_head * n_heads
        self.d_v = d_in * n_heads
        d_qkv = self.d_q + self.d_k + self.d_v

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
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(d_expand, d_out),
        )

        torch.nn.init.zeros_(self.ff[-1].weight)
        torch.nn.init.zeros_(self.ff[-1].bias)

        self.shortcut = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()

    def forward(self, x, mask=None):
        B, N, in_c = x.shape

        q, k, v = self.qkv_proj(x).split([self.d_q, self.d_k, self.d_v], dim=-1)
        q = rearrange(q, 'B N (H C) -> (B H) N C', H=self.H)  # (B*H, N, head_c)
        kt = rearrange(k, 'B N (H C) -> (B H) C N', H=self.H)  # (B*H, head_c, N)
        v = rearrange(v, 'B N (H C) -> (B H) N C', H=self.H)  # (B*H, N, in_c)

        # attn shape: (B*H, N, N)
        attn = q @ kt * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask, float('-inf'))
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # out shape: (B, N, H*in_c)
        out = rearrange(torch.bmm(attn, v), '(B H) N C -> B N (H C)', H=self.H, C=in_c)
        x = x + self.merge_head_proj(out)

        return self.ff(x) + self.shortcut(x)


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_in: int, d_context: int, d_head: int, d_expand: int, d_out: int, d_time: int, n_heads: int, dropout: float=0.1):
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
        self.d_time = d_time

        self.q_proj = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_head * n_heads),
        )

        self.kv_proj = nn.Sequential(
            nn.LayerNorm(d_context),
            nn.Linear(d_context, (d_head + d_in) * n_heads),
        )

        self.merge_head_proj = nn.Linear(d_in * self.H, d_in)
        torch.nn.init.zeros_(self.merge_head_proj.weight)
        torch.nn.init.zeros_(self.merge_head_proj.bias)

        self.time_proj = nn.Sequential(
            nn.Linear(d_time, d_time),
            Swish(),
            nn.Linear(d_time, d_in)
        )

        self.ff = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_expand),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(d_expand, d_out),
        )

        torch.nn.init.zeros_(self.ff[-1].weight)
        torch.nn.init.zeros_(self.ff[-1].bias)

        self.shortcut = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()

    def forward(self, x, context, t):
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

        t = self.time_proj(t).view(B, 1, self.d_in)
        return self.ff(x + t) + self.shortcut(x)


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

        self.d_q = d_head * n_heads
        self.d_k = d_head *n_heads
        self.d_v = d_in * n_heads
        d_qkv = self.d_q * 2 + self.d_k * 2 + self.d_v

        self.qkv_proj = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_qkv),
        )

        self.score_lambda = nn.Parameter(torch.tensor(0.5))

        self.merge_head_proj = nn.Linear(d_in * self.H, d_in)
        torch.nn.init.zeros_(self.merge_head_proj.weight)
        torch.nn.init.zeros_(self.merge_head_proj.bias)

        self.time_proj = nn.Sequential(
            nn.Linear(d_time, d_time),
            Swish(),
            nn.Linear(d_time, d_in)
        )

        self.ff = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_expand),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(d_expand, d_out),
        )
        torch.nn.init.zeros_(self.ff[-1].weight)
        torch.nn.init.zeros_(self.ff[-1].bias)

        self.shortcut = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()

    def forward(self, x, t):
        B, N, in_c = x.shape

        q1, q2, k1, k2, v = self.qkv_proj(x).split([self.d_q, self.d_q, self.d_k, self.d_k, self.d_v], dim=-1)
        q1 = rearrange(q1, 'B N (H C) -> (B H) N C', H=self.H)    # (B*H, N, head_c)
        k1t = rearrange(k1, 'B N (H C) -> (B H) C N', H=self.H)   # (B*H, head_c, N)
        q2 = rearrange(q2, 'B N (H C) -> (B H) N C', H=self.H)  # (B*H, N, head_c)
        k2t = rearrange(k2, 'B N (H C) -> (B H) C N', H=self.H)  # (B*H, head_c, N)
        v = rearrange(v, 'B N (H C) -> (B H) N C', H=self.H)    # (B*H, N, in_c)

        # attn shape: (B*H, N, N)
        score_main = torch.softmax(q1 @ k1t * self.scale, dim=-1)
        score_res = torch.softmax(q2 @ k2t * self.scale, dim=-1)
        attn = score_main - self.score_lambda * score_res
        attn = self.dropout(attn)

        # out shape: (B, N, H*in_c)
        out = rearrange(torch.bmm(attn, v), '(B H) N C -> B N (H C)', H=self.H, C=in_c)
        x = x + self.merge_head_proj(out)

        t = self.time_proj(t).view(B, 1, self.d_in)
        return self.shortcut(x) + self.ff(x + t)


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
            Swish(),
            nn.Conv2d(d_in * 2, d_in * 2, 3, 1, 1),
            nn.BatchNorm2d(d_in * 2),
            Swish(),
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
            nn.GroupNorm(8, d_mid),
            Swish(),
            nn.Conv1d(d_mid, d_mid, 3, 1, 1),
            nn.GroupNorm(8, d_mid),
            Swish(),
            nn.Conv1d(d_mid, d_out, 3, 1, 1)
        )

        torch.nn.init.zeros_(self.layers[-1].weight)
        torch.nn.init.zeros_(self.layers[-1].bias)

        self.shortcut = nn.Identity() if d_in == d_out else nn.Conv1d(d_in, d_out, 1, 1, 0)

    def forward(self, x):
        return self.shortcut(x) + self.layers(x)
class Conv2dInAct(nn.Sequential):
    def __init__(self, d_in: int, d_out: int, k: int, s: 1, p: 0):
        super().__init__(
            nn.Conv2d(d_in, d_out, k, s, p),
            nn.InstanceNorm2d(d_out),
            Swish()
        )


class Conv1dBnAct(nn.Sequential):
    def __init__(self, d_in: int, d_out: int, k: int, s: 1, p: 0):
        super().__init__(
            nn.Conv1d(d_in, d_out, k, s, p),
            nn.BatchNorm1d(d_out),
            Swish()
        )


class SequentialWithAdditionalInputs(nn.Sequential):
    def forward(self, x, *args):
        for module in self:
            x = module(x, *args)
        return x


class Rearrange(nn.Module):
    def __init__(self, from_shape: str, to_shape: str, **kwargs):
        super().__init__()
        self.from_shape = from_shape
        self.to_shape = to_shape
        self.kwargs = kwargs

    def forward(self, x):
        return rearrange(x, self.from_shape + ' -> ' + self.to_shape, **self.kwargs)
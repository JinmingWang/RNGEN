import torch
import torch.nn as nn
import torch.nn.functional as func
from einops import rearrange
from jaxtyping import Float, Bool
from typing import List, Literal, Tuple
from inspect import getfullargspec
import math

from .Basics import PositionalEncoding, Swish, FeatureNorm

Tensor = torch.Tensor

class AttentionBlockOld(nn.Module):
    def __init__(self, d_in: int, d_head: int, d_expand: int, d_out: int, n_heads: int,
                 dropout: float=0.0, score: Literal["prod", "dist"] = "dist", d_time: int = 0):
        super(AttentionBlockOld, self).__init__()

        # in shape: (B, N, in_c)
        # Q shape: (B*H, N, head_c)
        # K shape: (B*H, N, head_c)
        # V shape: (B*H, N, in_c)
        self.H = n_heads
        self.d_head = d_head
        self.d_in = d_in
        self.d_expand = d_expand
        self.score = score
        self.d_time = d_time
        self.scale = nn.Parameter(torch.tensor(d_head if score == "prod" else 1.414, dtype=torch.float32))
        self.dropout = nn.Dropout(dropout)

        self.d_q = d_head * n_heads
        self.d_k = d_head * n_heads
        self.d_v = d_in * n_heads
        d_qkv = self.d_q + self.d_k + self.d_v

        self.qkv_proj = nn.Sequential(
            PositionalEncoding(d_in),
            FeatureNorm(d_in),
            Swish(),
            nn.Linear(d_in, d_qkv),
        )

        self.merge_head_proj = nn.Linear(d_in * self.H, d_in)
        torch.nn.init.zeros_(self.merge_head_proj.weight)
        torch.nn.init.zeros_(self.merge_head_proj.bias)

        if d_time == 0:
            self.ff = nn.Sequential(
                FeatureNorm(d_in),
                Swish(),
                nn.Linear(d_in, d_expand),
                Swish(),
                nn.Dropout(dropout),
                nn.Linear(d_expand, d_out),
            )
            torch.nn.init.zeros_(self.ff[-1].weight)
            torch.nn.init.zeros_(self.ff[-1].bias)
        else:
            self.time_proj = nn.Sequential(
                nn.Linear(d_time, d_time),
                Swish(),
                nn.Linear(d_time, d_expand)
            )
            # self.ff1 = nn.Sequential(self.norm(d_in), Swish(), nn.Linear(d_in, d_expand))
            self.ff1 = nn.Sequential(FeatureNorm(d_in), Swish(), nn.Linear(d_in, d_expand))
            self.ff2 = nn.Sequential(Swish(), nn.Dropout(dropout), nn.Linear(d_expand, d_out))
            torch.nn.init.zeros_(self.ff2[-1].weight)
            torch.nn.init.zeros_(self.ff2[-1].bias)

        self.shortcut = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()

    def forward(self, x, t=None):
        B, N, in_c = x.shape

        q, k, v = self.qkv_proj(x).split([self.d_q, self.d_k, self.d_v], dim=-1)
        q = rearrange(q, 'B N (H C) -> (B H) N C', H=self.H)  # (B*H, N, head_c)
        k = rearrange(k, 'B N (H C) -> (B H) N C', H=self.H)  # (B*H, head_c, N)
        v = rearrange(v, 'B N (H C) -> (B H) N C', H=self.H)  # (B*H, N, in_c)

        if self.score == "prod":
            attn = torch.softmax(q @ k.transpose(-1, -2) * (self.scale ** -0.5), dim=-1)
        else:
            attn = torch.softmax(- torch.cdist(q, k) ** 2 / self.scale ** 2, dim=-1)
            # attn = torch.softmax(torch.exp(- torch.cdist(q, k) ** 2 / self.scale ** 2), dim=-1)
        attn = self.dropout(attn)

        # out shape: (B, N, H*in_c)
        out = rearrange(torch.bmm(attn, v), '(B H) N C -> B N (H C)', H=self.H, C=in_c)
        x = x + self.merge_head_proj(out)

        if self.d_time == 0:
            return self.ff(x) + self.shortcut(x)
        else:
            return self.ff2(self.ff1(x) + self.time_proj(t)) + self.shortcut(x)
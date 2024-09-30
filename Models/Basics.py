import torch
import torch.nn as nn
import torch.nn.functional as func
from einops import rearrange

class AttentionBlock(nn.Module):
    def __init__(self, in_c: int, head_c: int, expand_c: int, out_c: int, num_heads: int, dropout: float=0.1):
        super(AttentionBlock, self).__init__()

        # in shape: (B, N, in_c)
        # Q shape: (B*H, N, head_c)
        # K shape: (B*H, N, head_c)
        # V shape: (B*H, N, in_c)
        self.H = num_heads
        self.head_c = head_c
        self.in_c = in_c
        self.scale = head_c ** -0.5
        self.dropout = nn.Dropout(dropout)

        qkv_dim = head_c * num_heads * 2 + in_c * num_heads

        self.qkv_proj = nn.Sequential(
            nn.LayerNorm(in_c),
            nn.Linear(in_c, qkv_dim),
        )

        self.merge_head_proj = nn.Linear(in_c * self.H, in_c)

        self.ff = nn.Sequential(
            nn.LayerNorm(in_c),
            nn.Linear(in_c, expand_c),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expand_c, out_c),
        )

        self.shortcut = nn.Linear(in_c, out_c) if in_c != out_c else nn.Identity()

    def forward(self, x):
        B, N, in_c = x.shape

        q, k, v = self.qkv_proj(x).split([self.head_c * self.H, self.head_c * self.H, in_c * self.H], dim=-1)
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
    def __init__(self, in_c: int, context_c: int, head_c: int, expand_c: int, out_c: int, num_heads: int, dropout: float=0.1):
        super(CrossAttentionBlock, self).__init__()

        # in shape: (B, N, in_c)
        # Q shape: (B*H, N, head_c)
        # K shape: (B*H, N, head_c)
        # V shape: (B*H, N, in_c)
        self.H = num_heads
        self.head_c = head_c
        self.in_c = in_c
        self.scale = head_c ** -0.5
        self.dropout = nn.Dropout(dropout)

        self.q_proj = nn.Sequential(
            nn.LayerNorm(in_c),
            nn.Linear(in_c, head_c * num_heads),
        )

        self.kv_proj = nn.Sequential(
            nn.LayerNorm(context_c),
            nn.Linear(context_c, (head_c + in_c) * num_heads),
        )

        self.merge_head_proj = nn.Linear(in_c * self.H, in_c)

        self.ff = nn.Sequential(
            nn.LayerNorm(in_c),
            nn.Linear(in_c, expand_c),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expand_c, out_c),
        )

        self.shortcut = nn.Linear(in_c, out_c) if in_c != out_c else nn.Identity()

    def forward(self, x, context):
        B, N, in_c = x.shape

        q = rearrange(self.q_proj(x), 'B N (H C) -> (B H) N C', H=self.H)    # (B*H, N, head_c)

        k, v = self.kv_proj(context).split([self.head_c * self.H, in_c * self.H], dim=-1)
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


class NegLogEuclidianAttn(nn.Module):
    def __init__(self, in_c: int, context_c: int, head_c: int, expand_c: int, out_c: int, num_heads: int, dropout: float=0.1):
        super(NegLogEuclidianAttn, self).__init__()

        # in shape: (B, N, in_c)
        # Q shape: (B*H, N, head_c)
        # K shape: (B*H, N, head_c)
        # V shape: (B*H, N, in_c)
        self.H = num_heads
        self.head_c = head_c
        self.in_c = in_c
        self.scale = head_c ** -0.5
        self.dropout = nn.Dropout(dropout)

        self.q_proj = nn.Sequential(
            nn.LayerNorm(in_c),
            nn.Linear(in_c, head_c * num_heads),
        )

        self.kv_proj = nn.Sequential(
            nn.LayerNorm(context_c),
            nn.Linear(context_c, (head_c + in_c) * num_heads),
        )

        self.merge_head_proj = nn.Linear(in_c * self.H, in_c)

        self.ff = nn.Sequential(
            nn.LayerNorm(in_c),
            nn.Linear(in_c, expand_c),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expand_c, out_c),
        )

        self.shortcut = nn.Linear(in_c, out_c) if in_c != out_c else nn.Identity()

    def forward(self, x, context):
        B, N, in_c = x.shape

        q = rearrange(self.q_proj(x), 'B N (H C) -> (B H) N C', H=self.H)    # (B*H, N, head_c)

        k, v = self.kv_proj(context).split([self.head_c * self.H, in_c * self.H], dim=-1)
        k = rearrange(k, 'B N (H C) -> (B H) N C', H=self.H)   # (B*H, head_c, N)
        v = rearrange(v, 'B N (H C) -> (B H) N C', H=self.H)    # (B*H, N, in_c)

        # attn shape: (B*H, N, N)
        score = -torch.log(torch.cdist(q, k))

        # out shape: (B, N, H*in_c)
        out = rearrange(torch.bmm(score, v), '(B H) N C -> B N (H C)', H=self.H, C=in_c)
        x = x + self.merge_head_proj(out)

        return self.ff(x) + self.shortcut(x)

class Transpose(nn.Module):
    def __init__(self, dim0: int, dim1: int):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)


class Res(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_c, in_c * 2, 3, 1, 1),
            nn.InstanceNorm2d(in_c * 2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_c * 2, in_c * 2, 3, 1, 1),
            nn.InstanceNorm2d(in_c * 2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_c * 2, out_c, 3, 1, 1)
        )

        self.shortcut = nn.Identity() if in_c == out_c else nn.Conv2d(in_c, out_c, 1, 1, 0)

    def forward(self, x):
        return self.shortcut(x) + self.layers(x)

class Conv2dInAct(nn.Sequential):
    def __init__(self, in_c: int, out_c: int, k: int, s: 1, p: 0):
        super().__init__(
            nn.Conv2d(in_c, out_c, k, s, p),
            nn.InstanceNorm2d(out_c),
            nn.LeakyReLU(inplace=True)
        )
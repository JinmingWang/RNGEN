import torch
import torch.nn as nn
import torch.nn.functional as func
from einops import rearrange
from jaxtyping import Float, Bool
from typing import List, Literal, Tuple
from inspect import getfullargspec

Tensor = torch.Tensor


class Swish(nn.Module):
    def forward(self, x):
        return x * func.sigmoid(x)


class FeatureNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.eps = eps
        # Learnable parameters for scaling and shifting
        self.gamma = nn.Parameter(torch.ones(1, 1, num_features))
        self.beta = nn.Parameter(torch.zeros(1, 1, num_features))

    def forward(self, x):
        # x is of shape (B, L, D)
        mean = x.mean(dim=1, keepdim=True)  # Mean along the token dimension (L)
        var = x.var(dim=1, keepdim=True, unbiased=False)  # Variance along L
        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        # Apply learnable scale (gamma) and shift (beta)
        return self.gamma * x_norm + self.beta


class AttentionBlock(nn.Module):
    def __init__(self, l_in: int, d_in: int, d_head: int, d_expand: int, d_out: int, n_heads: int,
                 dropout: float=0.0, mask: bool=False):
        super(AttentionBlock, self).__init__()

        # in shape: (B, N, in_c)
        # Q shape: (B*H, N, head_c)
        # K shape: (B*H, N, head_c)
        # V shape: (B*H, N, in_c)
        self.H = n_heads
        self.d_head = d_head
        self.d_in = d_in
        self.l_in = l_in
        # self.scale = nn.Parameter(torch.tensor(d_head, dtype=torch.float32))
        self.scale = nn.Parameter(torch.tensor(1.414))
        self.dropout = nn.Dropout(dropout)
        self.mask = mask

        self.d_q = d_head * n_heads
        self.d_k = d_head * n_heads
        self.d_v = d_in * n_heads
        d_qkv = self.d_q + self.d_k + self.d_v

        self.norm = FeatureNorm(d_in)

        self.pos_enc = nn.Parameter(torch.randn(1, l_in, d_in))

        self.qkv_proj = nn.Sequential(
            Swish(),
            nn.Linear(d_in, d_qkv),
        )

        self.merge_head_proj = nn.Linear(d_in * self.H, d_in)

        torch.nn.init.zeros_(self.merge_head_proj.weight)
        torch.nn.init.zeros_(self.merge_head_proj.bias)

        self.ff = nn.Sequential(
            FeatureNorm(d_in),
            nn.Linear(d_in, d_expand),
            FeatureNorm(d_expand),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(d_expand, d_out),
        )

        torch.nn.init.zeros_(self.ff[-1].weight)
        torch.nn.init.zeros_(self.ff[-1].bias)

        self.shortcut = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()

    def forward(self, x):
        B, N, in_c = x.shape

        q, k, v = self.qkv_proj(self.norm(x) + self.pos_enc).split([self.d_q, self.d_k, self.d_v], dim=-1)
        q = rearrange(q, 'B N (H C) -> (B H) N C', H=self.H)  # (B*H, N, head_c)
        k = rearrange(k, 'B N (H C) -> (B H) N C', H=self.H)  # (B*H, head_c, N)
        v = rearrange(v, 'B N (H C) -> (B H) N C', H=self.H)  # (B*H, N, in_c)

        # attn shape: (B*H, N, N)
        # score = q @ kt * (self.scale ** -0.5)
        score = torch.exp(- torch.cdist(q, k) ** 2 / self.scale ** 2)
        # if self.mask:
        #     upper_mask = torch.triu(torch.ones(1, N, N, device=x.device, dtype=torch.bool), diagonal=1)
        #     score = score.masked_fill(upper_mask, float('-inf'))
        # score = torch.softmax(score, dim=-1)
        score = self.dropout(score)

        # out shape: (B, N, H*in_c)
        out = rearrange(torch.bmm(score, v), '(B H) N C -> B N (H C)', H=self.H, C=in_c)
        x = x + self.merge_head_proj(out)

        return self.ff(x) + self.shortcut(x)


class CrossAttnBlock(nn.Module):
    def __init__(self, d_in: int, d_context: int, d_head: int, d_expand: int,
                 d_out: int, n_heads: int, dropout: float=0.1):
        super(CrossAttnBlock, self).__init__()

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
            Swish(),
            nn.Linear(d_in, d_head * n_heads),
        )

        self.kv_proj = nn.Sequential(
            nn.LayerNorm(d_context),
            Swish(),
            nn.Linear(d_context, (d_head + d_in) * n_heads),
        )

        self.merge_head_proj = nn.Linear(d_in * self.H, d_in)
        torch.nn.init.zeros_(self.merge_head_proj.weight)
        torch.nn.init.zeros_(self.merge_head_proj.bias)

        self.ff = nn.Sequential(
            nn.LayerNorm(d_in),
            Swish(),
            nn.Linear(d_in, d_expand),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(d_expand, d_out),
        )

        torch.nn.init.zeros_(self.ff[-1].weight)
        torch.nn.init.zeros_(self.ff[-1].bias)

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

class CrossAttentionBlockWithTime(nn.Module):
    def __init__(self, l_in: int, d_in: int, d_context: int, d_head: int, d_expand: int,
                 d_out: int, d_time: int, n_heads: int, dropout: float=0.1):
        super(CrossAttentionBlockWithTime, self).__init__()

        # in shape: (B, N, in_c)
        # Q shape: (B*H, N, head_c)
        # K shape: (B*H, N, head_c)
        # V shape: (B*H, N, in_c)
        self.l_in = l_in
        self.H = n_heads
        self.d_head = d_head
        self.d_in = d_in
        self.d_expand = d_expand
        self.scale = d_head ** -0.5
        self.dropout = nn.Dropout(dropout)
        self.d_time = d_time

        self.ln = nn.LayerNorm(d_in)

        self.pos_enc = nn.Parameter(torch.randn(1, l_in, d_in))

        self.q_proj = nn.Sequential(
            Swish(),
            nn.Linear(d_in, d_head * n_heads),
        )

        self.kv_proj = nn.Sequential(
            nn.Linear(d_context, d_context),
            nn.LayerNorm(d_context),
            Swish(),
            nn.Linear(d_context, (d_head + d_in) * n_heads),
        )

        mask = 1 - torch.triu(torch.ones(l_in, l_in))
        self.register_buffer("mask", mask.unsqueeze(0).to(torch.bool))

        self.merge_head_proj = nn.Linear(d_in * self.H, d_in)
        torch.nn.init.zeros_(self.merge_head_proj.weight)
        torch.nn.init.zeros_(self.merge_head_proj.bias)

        self.time_proj = nn.Sequential(
            nn.Linear(d_time, d_time),
            Swish(),
            nn.Linear(d_time, d_in + d_expand),
            nn.Unflatten(-1, (1, -1))
        )

        self.ff1 = nn.Sequential(
            nn.LayerNorm(d_in),
            Swish(),
            nn.Linear(d_in, d_expand),
        )

        self.ff2 = nn.Sequential(
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(d_expand, d_out),
        )

        torch.nn.init.zeros_(self.ff2[-1].weight)
        torch.nn.init.zeros_(self.ff2[-1].bias)

        self.shortcut = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()

    def forward(self, x, context, t):
        B, N, in_c = x.shape

        q = rearrange(self.q_proj(self.ln(x) + self.pos_enc), 'B N (H C) -> (B H) N C', H=self.H)    # (B*H, N, head_c)

        k, v = self.kv_proj(context).split([self.d_head * self.H, in_c * self.H], dim=-1)
        kt = rearrange(k, 'B N (H C) -> (B H) C N', H=self.H)   # (B*H, head_c, N)
        v = rearrange(v, 'B N (H C) -> (B H) N C', H=self.H)    # (B*H, N, in_c)

        # attn shape: (B*H, N, N)
        attn = torch.bmm(q, kt) * self.scale
        attn.masked_fill_(self.mask, -torch.inf)
        attn = func.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # out shape: (B, N, H*in_c)
        out = rearrange(torch.bmm(attn, v), '(B H) N C -> B N (H C)', H=self.H, C=in_c)
        x = x + self.merge_head_proj(out)

        t_shift, t_scale = self.time_proj(t).split([self.d_in, self.d_expand], dim=-1)
        return self.ff2(self.ff1(x + t_shift) * torch.sigmoid(t_scale)) + self.shortcut(x)


class AttentionWithTime(nn.Module):
    def __init__(self, l_in: int, d_in: int, d_head: int, d_expand: int, d_out: int, d_time: int, n_heads: int, dropout: float=0.0):
        super(AttentionWithTime, self).__init__()

        # in shape: (B, N, in_c)
        # Q shape: (B*H, N, head_c)
        # K shape: (B*H, N, head_c)
        # V shape: (B*H, N, in_c)
        self.l_in = l_in
        self.H = n_heads
        self.d_head = d_head
        self.d_in = d_in
        self.d_expand = d_expand
        #self.scale = nn.Parameter(torch.tensor(d_head, dtype=torch.float32))
        self.scale = nn.Parameter(torch.tensor(1.414))
        self.d_time = d_time
        self.dropout = nn.Dropout(dropout)

        self.d_q = d_head * n_heads
        self.d_k = d_head *n_heads
        self.d_v = d_in * n_heads
        d_qkv = self.d_q + self.d_k + self.d_v

        # self.norm = nn.LayerNorm(d_in)
        self.norm = FeatureNorm(d_in)

        self.pos_enc = nn.Parameter(torch.randn(1, l_in, d_in))

        self.qkv_proj = nn.Sequential(
            Swish(),
            nn.Linear(d_in, d_qkv),
        )

        self.merge_head_proj = nn.Linear(d_in * self.H, d_in)
        torch.nn.init.zeros_(self.merge_head_proj.weight)
        torch.nn.init.zeros_(self.merge_head_proj.bias)

        self.time_proj = nn.Linear(d_time, d_in + d_expand)

        self.ff1 = nn.Sequential(
            FeatureNorm(d_in),
            Swish(),
            nn.Linear(d_in, d_expand),
        )

        self.ff2 = nn.Sequential(
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(d_expand, d_out),
        )

        torch.nn.init.zeros_(self.ff2[-1].weight)
        torch.nn.init.zeros_(self.ff2[-1].bias)

        self.shortcut = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()

    def forward(self, x, t):
        B, N, in_c = x.shape

        q, k, v = self.qkv_proj(self.norm(x) + self.pos_enc).split([self.d_q, self.d_k, self.d_v], dim=-1)
        q = rearrange(q, 'B N (H C) -> (B H) N C', H=self.H)    # (B*H, N, head_c)
        k = rearrange(k, 'B N (H C) -> (B H) N C', H=self.H)   # (B*H, head_c, N)
        v = rearrange(v, 'B N (H C) -> (B H) N C', H=self.H)    # (B*H, N, in_c)

        # attn shape: (B*H, N, N)
        # attn = torch.softmax(q @ kt * (self.scale  ** -0.5), dim=-1)
        attn = torch.exp(- torch.cdist(q, k) ** 2 / self.scale ** 2)
        attn = self.dropout(attn)

        # out shape: (B, N, H*in_c)
        out = rearrange(attn @ v, '(B H) N C -> B N (H C)', H=self.H, C=in_c)
        x = x + self.merge_head_proj(out)

        t_shift, t_scale = self.time_proj(t).split([self.d_in, self.d_expand], dim=-1)
        return self.ff2(self.ff1(x + t_shift) * torch.sigmoid(t_scale)) + self.shortcut(x)


class Transpose(nn.Module):
    def __init__(self, dim0: int, dim1: int):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)


class Res2D(nn.Module):
    def __init__(self, d_in: int, d_mid: int, d_out: int):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(d_in, d_mid, 3, 1, 1),
            nn.GroupNorm(8, d_mid),
            Swish(),
            nn.Conv2d(d_mid, d_mid, 3, 1, 1),
            nn.GroupNorm(8, d_mid),
            Swish(),
            nn.Conv2d(d_mid, d_out, 3, 1, 1)
        )

        torch.nn.init.zeros_(self.layers[-1].weight)
        torch.nn.init.zeros_(self.layers[-1].bias)

        self.shortcut = nn.Identity() if d_in == d_out else nn.Conv2d(d_in, d_out, 1, 1, 0)

    def forward(self, x):
        return self.shortcut(x) + self.layers(x)


class Res1D(nn.Module):
    def __init__(self, d_in: int, d_mid: int, d_out: int):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv1d(d_in, d_mid, 1, 1, 0),
            nn.GroupNorm(8, d_mid),
            Swish(),
            nn.Conv1d(d_mid, d_mid, 3, 1, 1),
            nn.GroupNorm(8, d_mid),
            Swish(),
            nn.Conv1d(d_mid, d_out, 1, 1, 0)
        )

        torch.nn.init.zeros_(self.layers[-1].weight)
        torch.nn.init.zeros_(self.layers[-1].bias)

        self.shortcut = nn.Identity() if d_in == d_out else nn.Conv1d(d_in, d_out, 1, 1, 0)

    def forward(self, x):
        return self.shortcut(x) + self.layers(x)

class Conv2dNormAct(nn.Sequential):
    def __init__(self, d_in: int, d_out: int, k: int, s: 1, p: 0):
        super().__init__(
            nn.Conv2d(d_in, d_out, k, s, p),
            nn.GroupNorm(8, d_out),
            Swish()
        )


class Conv1dNormAct(nn.Sequential):
    def __init__(self, d_in: int, d_out: int, k: int, s: 1, p: 0):
        super().__init__(
            nn.Conv1d(d_in, d_out, k, s, p),
            nn.GroupNorm(8, d_out),
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


def cacheArgumentsUponInit(original_init):
    # set an attribute with the same name and value as whatever is passed to __init__
    def __init__(self, *args, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        # get names of args
        argspec = getfullargspec(original_init)
        for i, arg in enumerate(argspec.args):
            if i < len(args):
                setattr(self, arg, args[i])

        original_init(self, *args, **kwargs)

    return __init__

def xyxy2xydl(xyxy: Tensor) -> Tensor:
    """
    Convert a segment (x1, y1, x2, y2) to a segment (x_center, y_center, direction, length)
    Why? Because a segment in x1y1x2y2 format can also be represented in x2y2x1y1 format,
    this non-uniqueness is problematic. Imagine a segment (x1, y1, x2, y2), if the model predicts
    (x2, y2, x1, y1) instead, it should be considered correct. Or, if two given segments are (x1, y1, x2, y2) and
    (x2, y2, x1, y1), the attention mechanism should be able to match them as if they are the same segment.
    :param xyxy: The segments (B, N, 5), where 5 is (x1, y1, x2, y2, is_valid)
    :return: The segments in (B, N, 5) format
    """
    # Unbind the input tensor to get x1, y1, x2, y2
    x1, y1, x2, y2 = torch.unbind(xyxy, dim=-1)

    # Compute center coordinates
    x_c, y_c = (x1 + x2) / 2, (y1 + y2) / 2

    # Compute the length of each segment
    lengths = torch.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # Compute the direction and ensure it's in the range [0, pi)
    directions = torch.atan2(y2 - y1, x2 - x1)
    directions = torch.remainder(directions, torch.pi)

    # Return the concatenated result
    return torch.stack([x_c, y_c, directions, lengths], dim=-1)

def xydl2xyxy(xydl: Tensor) -> Tensor:
    """
    Convert a segment (x_center, y_center, direction, length) to a segment (x1, y1, x2, y2)
    :param xydl: The segments (B, N, 5), where 5 is (x_center, y_center, direction, length, is_valid)
    :return: The segments in (B, N, 5) format
    """
    # Unbind the input tensor to get x_c, y_c, directions, lengths
    x_c, y_c, directions, lengths = torch.unbind(xydl, dim=-1)

    # Compute half-length components
    half_dx = (lengths / 2) * torch.cos(directions)
    half_dy = (lengths / 2) * torch.sin(directions)

    # Compute x1, y1, x2, y2 based on the center and direction
    x1, y1 = x_c - half_dx, y_c - half_dy
    x2, y2 = x_c + half_dx, y_c + half_dy

    # Return the concatenated result
    return torch.stack([x1, y1, x2, y2], dim=-1)
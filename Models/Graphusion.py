from .Basics import *


class GraphusionSelfAttn(nn.Module):
    def __init__(self,
                 d_in: int,
                 d_head: int,
                 d_expand: int,
                 d_out: int,
                 n_heads: int,
                 dropout: float = 0.1,
                 score: Literal["prod", "dist"] = "dist",
                 d_time: int = 0):
        super(GraphusionSelfAttn, self).__init__()

        self.H = n_heads
        self.d_head = d_head
        self.d_expand = d_expand
        self.d_in = d_in
        self.score = score
        self.d_time = d_time
        self.scale = nn.Parameter(torch.tensor(d_head if score == "prod" else 1.414, dtype=torch.float32))
        self.dropout = nn.Dropout(dropout)

        self.d_q = d_head * n_heads
        self.d_k = d_head * n_heads
        self.d_v = d_in * n_heads
        d_qkv = self.d_q + self.d_k + self.d_v

        self.qkv_proj = nn.Sequential(
            nn.LayerNorm(d_in),
            Swish(),
            nn.Linear(d_in, d_qkv)
        )

        self.merge_head_proj = nn.Linear(d_in * self.H, d_in)
        torch.nn.init.zeros_(self.merge_head_proj.weight)
        torch.nn.init.zeros_(self.merge_head_proj.bias)

        if d_time == 0:
            self.ff = nn.Sequential(
                nn.LayerNorm(d_in), Swish(),
                nn.Linear(d_in, d_expand),
                Swish(), nn.Dropout(dropout),
                nn.Linear(d_expand, d_out),
            )
            torch.nn.init.zeros_(self.ff[-1].weight)
            torch.nn.init.zeros_(self.ff[-1].bias)
        else:
            self.time_proj = nn.Sequential(
                nn.Linear(d_time, d_time),
                Swish(),
                nn.Linear(d_time, d_in + d_expand)
            )
            self.ff1 = nn.Sequential(nn.LayerNorm(d_in), Swish(), nn.Linear(d_in, d_expand))
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

        # attn shape: (B*H, N, N)
        if self.score == "prod":
            attn = torch.softmax(q @ k.transpose(-1, -2) * (self.scale ** -0.5), dim=-1)
        else:
            attn = torch.exp(- torch.cdist(q, k) ** 2 / self.scale ** 2)
        attn = self.dropout(attn)

        # out shape: (B, N, H*in_c)
        out = rearrange(torch.bmm(attn, v), '(B H) N C -> B N (H C)', H=self.H, C=in_c)
        x = x + self.merge_head_proj(out)

        if self.d_time == 0:
            return self.ff(x) + self.shortcut(x)
        else:
            t_shift, t_scale = self.time_proj(t).split([self.d_in, self.d_expand], dim=-1)
            return self.ff2(self.ff1(x + t_shift) * torch.sigmoid(t_scale)) + self.shortcut(x)


class Block(nn.Module):
    def __init__(self,
                 d_in: int,
                 d_out: int,
                 d_ctx: int,
                 d_time: int,
                 n_heads: int = 8,
                 dropout: float = 0.0,
                 ):
        super().__init__()

        self.d_in = d_in
        self.d_out = d_out
        self.d_ctx = d_ctx
        self.n_heads = n_heads
        self.dropout = dropout

        self.time_proj = nn.Sequential(
            nn.Linear(d_time, d_in),
            Swish()
        )

        self.ca = CrossAttnBlock(
            d_in=d_in,
            d_context=d_ctx,
            d_head=64,
            d_expand=d_in*2,
            d_out=d_in*2,
            n_heads=n_heads,
            dropout=dropout,
            score="prod",
            d_time=0
        )


        self.sa = GraphusionSelfAttn(
            d_in=d_in*2,
            d_head=64,
            d_expand=d_in * 2,
            d_out=d_out,
            d_time=d_in,
            n_heads=self.n_heads,
            dropout=self.dropout,
            score="prod"
        )

        self.out_proj = nn.Sequential(
            Swish(),
            nn.Linear(d_in, d_out)
        )

        self.shortcut = nn.Identity() if d_in == d_out else nn.Linear(d_in, d_out)

    def forward(self, x, context, t):
        residual = self.shortcut(x)
        x = self.ca(x, context)
        x = self.sa(x, self.time_proj(t))
        return self.out_proj(x) + residual


class Graphusion(nn.Module):
    def __init__(self, D_in: int, L_enc: int, N_trajs: int, L_traj: int, d_context: int, n_layers: int, T: int):
        super().__init__()
        self.D_in = D_in
        self.L_enc = L_enc
        self.N_trajs = N_trajs
        self.L_traj = L_traj
        self.d_context = d_context
        self.n_layers = n_layers
        self.T = T

        self.time_embed = nn.Sequential(
            nn.Embedding(T, 128),
            nn.Linear(128, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 256),
            nn.LeakyReLU(inplace=True),
            nn.Unflatten(-1, (1, -1))
        )

        # Input: (B, N, L, 2)

        self.x_proj = nn.Sequential(
            nn.Linear(D_in, 128),
            GraphusionSelfAttn(128, 64, 256, 256, 8, score="prod"),
            GraphusionSelfAttn(256, 64, 256, 256, 8, score="prod"),
        )

        self.traj_proj = nn.Sequential(
            # Trajectory inner feature extraction
            # traj -> feature sequence
            Rearrange("B N L D", "(B N) D L"),  # (BN, 2, L')
            nn.Conv1d(d_context, 128, 3, 2, 1), Swish(),
            *[Res1D(128, 256, 128) for _ in range(4)],
            nn.Conv1d(128, 256, 3, 2, 1),

            # Attention among all traj tokens
            Rearrange("(B N) D L", "B N (L D)", N=N_trajs),
            nn.Linear(L_traj // 4 * 256, 256),
            GraphusionSelfAttn(256, 64, 512, 256, 4, score="prod"),
            GraphusionSelfAttn(256, 64, 512, 256, 4, score="prod"),
        )

        self.stages = SequentialWithAdditionalInputs(*[
            Block(256, 256, 256, 256, 8)
            for _ in range(n_layers)
        ])

        # (B, N, L, 128)
        self.head = nn.Sequential(
            nn.Linear(256, 128),
            Swish(),
            nn.Linear(128, D_in)
        )

    def forward(self, x, context, t):
        """
        :param segments: (B, N_segs, C_seg)
        :param traj_encoding: (B, N_traj=32, traj_encoding_c=128)
        :return:
        """

        t = self.time_embed(t)
        x = self.x_proj(x)
        context = self.traj_proj(context)

        x = self.stages(x, context, t)

        return self.head(x)

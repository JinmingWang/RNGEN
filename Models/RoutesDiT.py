from .Basics import *

class Block(nn.Module):
    def __init__(self,
                 n_paths: int,
                 l_path: int,
                 d_in: int,
                 d_out: int,
                 d_context: int,
                 d_time: int,
                 n_heads: int = 8,
                 dropout: float = 0.0,
                 ):
        super().__init__()

        self.d_in = d_in
        self.d_out = d_out
        self.d_context = d_context
        self.n_heads = n_heads
        self.dropout = dropout

        self.time_proj = nn.Sequential(
            nn.Linear(d_time, d_in),
            Swish()
        )

        d_mid = d_in + d_context
        self.res = nn.Sequential(
            Rearrange("B (N L) D", "(B N) D L", N=n_paths),
            Res1D(d_mid, d_mid*2, d_mid),
            Res1D(d_mid, d_mid*2, d_mid),
            nn.Conv1d(d_mid, d_out, 3, 1, 1),
            Rearrange("(B N) D L", "B (N L) D", N=n_paths)
        )

        self.attn = AttentionWithTime(
            l_in=n_paths * l_path,
            d_in=d_out,
            d_head=d_out // 4,
            d_expand=d_out * 2,
            d_out=d_out,
            d_time=d_in,
            n_heads=self.n_heads,
            dropout=self.dropout
        )

    def forward(self, x, context, t):
        # x: (B, N, L, D)
        # context: (B, N, L, D')
        x = self.res(torch.cat([x, context], dim=-1))   # (B, N, L, D)
        x = self.attn(x, self.time_proj(t))
        return x


class RoutesDiT(nn.Module):
    @cacheArgumentsUponInit
    def __init__(self, D_in: int, N_routes: int, L_route: int, L_traj: int, d_context: int, n_layers: int, T: int):
        super().__init__()

        self.time_embed = nn.Sequential(
            nn.Embedding(T, 128),
            nn.Linear(128, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(inplace=True),
            nn.Unflatten(-1, (1, -1))
        )

        # Input: (B, N, L, 2)

        self.x_proj = nn.Sequential(
            Rearrange("B N L D", "(B N) D L"), # (BN, 2, L)
            nn.Conv1d(D_in, 32, 3, 1, 1),
            Swish(),
            nn.Conv1d(32, 128, 3, 1, 1),
            Rearrange("(B N) D L", "B N L D", N=N_routes)
        )

        self.pos_route = nn.Parameter(torch.randn(1, 1, L_route, 128))

        self.context_proj = nn.Sequential(
            Rearrange("B N L D", "(B N) D L"),  # (BN, 2, L')
            nn.Conv1d(d_context, 32, 3, 2, 1),
            Swish(),

            *[Res1D(32, 64, 32) for _ in range(3)],
            Conv1dNormAct(32, 64, 3, 1, 1),

            *[Res1D(64, 128, 64) for _ in range(3)],
            Conv1dNormAct(64, 128, 3, 1, 1),

            *[Res1D(128, 256, 128) for _ in range(3)],
            Conv1dNormAct(128, 256, 3, 1, 1),

            Rearrange("(B N) D L", "B (N L) D", N=N_routes),

            AttentionBlock(L_traj//2 * N_routes, 256, 64, 512, 256, 4),
            AttentionBlock(L_traj//2 * N_routes, 256, 64, 512, 256, 4),

            Rearrange("B (N L) D", "B N L D", N=N_routes)
        )

        self.stages = SequentialWithAdditionalInputs(
            *[Block(N_routes, L_route, 128, 128, 256, 512, 8) for _ in range(n_layers)]
        )

        # (B, N, L, 128)
        self.head = nn.Sequential(
            Rearrange("B (N L) D",  "B N L D", N=N_routes),
            nn.Linear(128, 64),
            Swish(),
            nn.Linear(64, D_in)
        )

    def forward(self, x, context, t):
        """
        :param segments: (B, N_segs, C_seg)
        :param traj_encoding: (B, N_traj=32, traj_encoding_c=128)
        :return:
        """
        B = x.shape[0]

        t = self.time_embed(t)
        x = self.x_proj(x) + self.pos_route
        context = self.context_proj(context)

        x = rearrange(x, "B N L D -> B (N L) D")
        context = rearrange(context, "B N L D -> B (N L) D")

        x = self.stages(x, context, t)

        return self.head(x)

from .Basics import *

class Block(nn.Module):
    def __init__(self,
                 l_enc: int,
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

        self.cross_attn = CrossAttnBlock(
            d_in=d_in,
            d_context=d_context,
            d_head=64,
            d_expand=d_in * 2,
            d_out=d_in,
            n_heads=self.n_heads,
            dropout=self.dropout
        )

        self.attn = AttentionWithTime(
            l_in=l_enc,
            d_in=d_in,
            d_head=64,
            d_expand=d_in * 2,
            d_out=d_in,
            d_time=d_in,
            n_heads=self.n_heads,
            dropout=self.dropout
        )

        self.out_proj = nn.Sequential(
            Swish(),
            nn.Linear(d_in, d_out)
        )

        self.shortcut = nn.Identity() if d_in == d_out else nn.Linear(d_in, d_out)

    def forward(self, x, context, t):
        residual = self.shortcut(x)
        x = self.cross_attn(x, context)
        x = self.attn(x, self.time_proj(t))
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
            AttentionBlock(L_enc, 128, 64, 256, 256, 8),
            AttentionBlock(L_enc, 256, 64, 256, 256, 8),
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
            AttentionBlock(N_trajs, 256, 64, 512, 256, 4),
            AttentionBlock(N_trajs, 256, 64, 512, 256, 4),
        )

        self.stages = SequentialWithAdditionalInputs(*[
            Block(L_enc, 256, 256, 256, 256, 8)
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

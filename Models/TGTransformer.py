from .Basics import *
from .WGVAE_MHSA import MultiHeadSelfRelationMatrix, DRAC

class TGTransformer(nn.Module):
    def __init__(self, N_routes: int, L_route: int, N_interp: int, threshold: float = 0.5):
        super().__init__()
        self.N_routes = N_routes
        self.L_route = L_route
        self.N_interp = N_interp
        self.threshold = threshold

        self.N_segs = N_routes * L_route


        attn_params = {
            "d_in": 384, "d_head": 32, "d_expand": 512, "d_out": 384,
            "n_heads": 12, "dropout": 0.1, "score": "prod"
        }

        self.traj_proj = nn.Sequential(
            # Trajectory inner feature extraction
            # traj -> feature sequence
            Rearrange("B N L D", "(B N) D L"),  # (BN, 2, L')
            nn.Conv1d(2, 128, 3, 2, 1), Swish(),
            *[SERes1D(128, 256, 128) for _ in range(4)],
            nn.Conv1d(128, 384, 3, 1, 1),

            # Attention among all traj tokens
            Rearrange("(B N) D L", "B (N L) D", N=N_routes),
            *[AttentionBlock(384, 64, 512, 384, 4, score="prod") for _ in range(8)],
        )

        self.segs_head = nn.Sequential(
            AttentionBlock(**attn_params),
            nn.Linear(384, 128),
            Swish(),
            nn.Linear(128, N_interp * 2)
        )

        self.cluster_head = nn.Sequential(
            AttentionBlock(**attn_params),
            MultiHeadSelfRelationMatrix(384, 32),  # (B, L, L)
            nn.Sigmoid()
        )

        self.drac = DRAC(self.N_segs, threshold)

    def forward(self, trajs):
        x = self.traj_proj(trajs)

        duplicate_segs = self.segs_head(x)  # (B, N_routes*L_route, N_interp * 2)
        cluster_mat = self.cluster_head(x)  # (B, N_routes*L_route, N_routes*L_route)

        # cluster_means: (B, N_routes*L_route, N_interp * 2)
        # coi_means: (B, ? , N_interp*2)
        cluster_means, coi_means = self.drac(duplicate_segs, cluster_mat)

        duplicate_segs = duplicate_segs.unflatten(-1, (-1, 2))
        cluster_means = cluster_means.unflatten(-1, (-1, 2))
        coi_means = [coi_mean.unflatten(-1, (-1, 2)) for coi_mean in coi_means]

        return duplicate_segs, cluster_mat, cluster_means, coi_means


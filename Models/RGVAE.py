from .Basics import *


class MultiHeadSelfRelationMatrix(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, n_heads: int):
        super().__init__()

        self.d_in = d_in
        self.n_heads = n_heads

        self.edge_proj = nn.Sequential(
            nn.Linear(d_in, d_in),
            Swish(),
            nn.Linear(d_in, n_heads * d_hidden),
            Rearrange("B N (H C)", "(B H) N C", H=n_heads),
        )

        self.adj_mat_proj = nn.Sequential(
            nn.Unflatten(0, (-1, n_heads)),
            nn.Conv2d(n_heads, 1, 1, 1, 0),
            nn.Flatten(1, 2)
        )

    def forward(self, x):
        x = self.edge_proj(x)
        mat = torch.cdist(x, x, p=2)    # (B, H, N, N)
        return self.adj_mat_proj(mat)   # (B, N_nodes, N_nodes)


class DRAC(nn.Module):
    def __init__(self, threshold: float=0.5):
        super().__init__()
        """
        DRAC: Duplicate Removal and Cluster Averaging for Cross-Domain Trajectory Prediction
        """
        self.threshold = threshold


    def forward(self, seqs: Float[Tensor, "B L D"], cluster_mat: Float[Tensor, "B L L"]):
        B, L, D = seqs.shape

        # cluster_mat[i, j] > threshold means token i and token j are in the same cluster
        threshold_mask = (cluster_mat < self.threshold)

        # Step 1: Cluster Averaging, compute average for each row
        # Filtered_cluster_mat here is the weights for each token in the cluster
        # We use threshold because we do not want tokens that are not in this cluster to affect the average
        # Which are low-confidence tokens
        # We use -inf because later we will use softmax to get the weights
        # We use softmax because we are computing weighted average and the sum of the weights should be 1
        filtered_cluster_mat = torch.masked_fill(cluster_mat, threshold_mask, -torch.inf) # (B, L, L)
        # TODO: Check if this is correct
        # segment can be {p1, ... pn} or {pn, ... p1}
        cluster_means = torch.softmax(filtered_cluster_mat, dim=-1) @ seqs

        cluster_means[torch.isnan(cluster_means)] = 0

        # Step 2: Duplicate Removal, only keep the first token in the cluster
        # lower[i, j] = 1 means token i and j are in the same cluster, and i is after j
        lower = torch.tril(~threshold_mask, diagonal=-1)  # (B, L, L)
        is_first_token = torch.sum(lower, dim=2, keepdim=True) == 0  # (B, L, 1)
        cluster_means = cluster_means * is_first_token.float()

        # Step 3: Zero removing, lots of elements are zeroed out during step 2
        # Now we want to remove them
        # That is, we only want Cluster of Interest (COI) means
        # Index selecting only is_first_token == 1
        coi_means = []
        for b in range(B):
            coi_means.append(cluster_means[b, is_first_token[b, :, 0]])

        return cluster_means, coi_means


class RGVAE(nn.Module):
    def __init__(self, N_routes: int, L_route: int, N_interp: int, threshold: float=0.5):
        super().__init__()
        self.N_routes = N_routes
        self.L_route = L_route
        self.N_interp = N_interp
        self.threshold = threshold

        self.N_segs = N_routes * L_route

        # Input (B, N_trajs, L_route, N_interp, 2)
        self.encoder = nn.Sequential(
            Rearrange("B N_routes L_route N_interp D", "(B N_routes) (N_interp D) L_route"),

            nn.Conv1d(2 * N_interp, 64, 3, 1, 1),
            Swish(),

            *[SERes1D(64, 128, 64) for _ in range(3)],
            Conv1dNormAct(64, 128, 1, 1, 0),

            *[SERes1D(128, 256, 128) for _ in range(3)],
            nn.Conv1d(128, 256, 1, 1, 0),

            Rearrange("(B N_routes) D L_route", "B (N_routes L_route) D", N_routes=N_routes),

            AttentionBlock(256, 64, 512, 256, 8),
            AttentionBlock(256, 64, 512, 256, 8),

            Rearrange("B (N_routes L_route) D", "B N_routes L_route D", N_routes=N_routes),
        )

        self.mu_head = nn.Linear(256, 2 * N_interp)
        self.logvar_head = nn.Linear(256, 2 * N_interp)

        attn_params = {
            "d_in": 384, "d_head": 64, "d_expand": 768, "d_out": 384,
            "n_heads": 8
        }

        self.decoder_shared = nn.Sequential(
            Rearrange("B N_routes L_route D", "(B N_routes) D L_route"),

            nn.Conv1d(N_interp * 2, 64, 1, 1, 0),
            *[SERes1D(64, 128, 64) for _ in range(3)],

            nn.Conv1d(64, 128, 1, 1, 0),
            *[SERes1D(128, 256, 128) for _ in range(3)],

            nn.Conv1d(128, 384, 1, 1, 0),

            Rearrange("(B N_routes) D L_route", "B (N_routes L_route) D", N_routes=N_routes),
            *[AttentionBlock(**attn_params) for _ in range(6)],
        )

        self.segs_head = nn.Sequential(
            AttentionBlock(**attn_params),
            nn.Linear(384, 128),
            Swish(),
            nn.Linear(128, N_interp * 2)
        )

        self.cluster_head = nn.Sequential(
            AttentionBlock(**attn_params),
            MultiHeadSelfRelationMatrix(384, 64, 16), # (B, L, L)
            nn.Sigmoid()
        )

        self.drac = DRAC(threshold)


    def encode(self, paths):
        # paths: (B, N, L, 2)
        x = self.encoder(paths)
        z_mean = self.mu_head(x)
        z_logvar = self.logvar_head(x)

        return z_mean, z_logvar

    def decode(self, z):
        # z: (B, N_routes, L_route, N_interp * 2)
        B, N_routes, L_route, _ = z.shape

        x = self.decoder_shared(z)  # (B, N_routes*L_route, 256)
        duplicate_segs = self.segs_head(x)  # (B, N_routes*L_route, N_interp * 2)
        cluster_mat = self.cluster_head(x)  # (B, N_routes*L_route, N_routes*L_route)

        # cluster_means: (B, N_routes*L_route, N_interp * 2)
        # coi_means: (B, ? , N_interp*2)
        cluster_means, coi_means = self.drac(duplicate_segs, cluster_mat)

        duplicate_segs = duplicate_segs.unflatten(-1, (-1, 2))
        cluster_means = cluster_means.unflatten(-1, (-1, 2))
        coi_means = [coi_mean.unflatten(-1, (-1, 2)) for coi_mean in coi_means]

        return duplicate_segs, cluster_mat, cluster_means, coi_means

    @staticmethod
    def reparameterize(z_mean, z_logvar):
        # Sample epsilon from standard normal distribution
        epsilon = torch.randn_like(z_mean)
        # Compute the latent vector
        z = z_mean + torch.exp(0.5 * z_logvar) * epsilon
        return z

    def forward(self, paths):
        z_mean, z_logvar = self.encode(paths)
        z = self.reparameterize(z_mean, z_logvar)
        return z_mean, z_logvar, *self.decode(z)

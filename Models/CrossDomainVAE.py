from .Basics import *


class MultiHeadSelfRelationMatrix(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, n_heads: int):
        super().__init__()

        self.d_in = d_in
        self.n_heads = n_heads

        self.edge_proj = nn.Sequential(
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
        # We convert xyxy2xydl because xyxy can be (x1, y1, x2, y2) or (x2, y2, x1, y1)
        # If we compute the average between this kind of opposite line segments, the result will be wrong
        # So we convert it to xydl, which is (x_center, y_center, direction, length)
        xydl_seqs = xyxy2xydl(seqs)
        # Compute weighted average and convert back to xyxy
        cluster_means = torch.softmax(filtered_cluster_mat, dim=-1) @ xydl_seqs
        cluster_means = xydl2xyxy(cluster_means)

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


class CrossDomainVAE(nn.Module):
    @cacheArgumentsUponInit
    def __init__(self, N_paths: int, L_path: int, D_enc: int, threshold: float=0.5):
        super().__init__()

        self.N_nodes = N_paths * L_path
        self.N_segs = N_paths * (L_path - 1)

        # Input (B, N, L, D)
        self.encoder = nn.Sequential(
            Rearrange("B N L D", "(B N) D L"),  # (BN, 2, L)
            nn.Conv1d(2, 32, 3, 1, 1),
            Swish(),

            Conv1dBnAct(32, 64, 3, 1, 1),
            Res1D(64, 128, 64),
            Res1D(64, 128, 64),
            Res1D(64, 128, 64),
            Res1D(64, 128, 64),

            Rearrange("(B N) D L", "B N L D", N=N_paths),
        )

        self.mu_head = nn.Linear(64, D_enc)
        self.logvar_head = nn.Linear(64, D_enc)

        self.decoder_shared = nn.Sequential(
            Rearrange("B L D", "B D L"),

            nn.Conv1d(D_enc * 2, 64, 1, 1, 0),
            Res1D(64, 128, 64),
            Res1D(64, 128, 64),
            Res1D(64, 128, 64),

            nn.Conv1d(64, 128, 1, 1, 0),
            Res1D(128, 256, 128),
            Res1D(128, 256, 128),
            Res1D(128, 256, 128),

            Rearrange("B D L", "B L D"),
            AttentionBlock(self.N_segs, 128, 64, 256, 256, 8, 0.0),
        )   # (B, D=128, L)

        self.segs_head = nn.Sequential(
            AttentionBlock(self.N_segs, 256, 64, 512, 256, 8, 0.0),
            nn.Linear(256, 4),
        )

        self.cluster_head = nn.Sequential(
            AttentionBlock(self.N_segs, 256, 64, 512, 256, 8, 0.0),
            AttentionBlock(self.N_segs, 256, 64, 512, 256, 8, 0.0),
            AttentionBlock(self.N_segs, 256, 64, 512, 256, 8, 0.0),
            AttentionBlock(self.N_segs, 256, 64, 512, 256, 8, 0.0),
            nn.Linear(256, 128),
            Swish(),
            MultiHeadSelfRelationMatrix(128, 128, 16), # (B, L, L)
            nn.Sigmoid()
        )

        self.drac = DRAC(threshold)


    def encode(self, paths):
        # paths: (B, N, L, 2)
        x = self.encoder(paths)
        return self.mu_head(x), self.logvar_head(x)

    def decode(self, z):
        # encodings: (B, N, L, D)
        # paths: (B, N, L, 2)

        B, N, L, D = z.shape
        encoding_pairs = torch.cat([z[..., 1:, :], z[..., :-1, :]], dim=-1)  # (B, N, L-1, 4)
        encoding_pairs = encoding_pairs.view(B, -1, D * 2).contiguous()  # (B, N_segs, 4)

        x = self.decoder_shared(encoding_pairs)
        duplicate_segs = self.segs_head(x)
        cluster_mat = self.cluster_head(x)

        cluster_means, coi_means = self.drac(duplicate_segs, cluster_mat)

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

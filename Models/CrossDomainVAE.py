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
        self.threshold = threshold


    def forward(self, seqs: Tensor, cluster_mat: Tensor):
        # x: (B, L, D)
        B, L, D = seqs.shape
        # cluster_mat[i, j] = 1 if x[i] and x[j] are in the same cluster

        cluster_mat = (cluster_mat > self.threshold).float()

        # With this mask, a token can only see its previous tokens
        pre_mask = torch.tril(torch.ones(L, L), diagonal=-1).to(seqs.device)

        # How to do cluster averaging?
        # First, every token will be the average of the entire cluster
        cluster_size = cluster_mat.sum(dim=2, keepdim=True)  # (B, L, 1)
        xydl_seqs = xyxy2xydl(seqs)
        cluster_means = cluster_mat @ xydl_seqs / cluster_size  # (B, L, D)
        cluster_means = xydl2xyxy(cluster_means)

        # Then we need to check whether token i is the first token in the cluster
        # cluster_mat * pre_mask will give the tokens in the cluster that are before token i
        # If the sum if 0, then no token is before token i in the cluster
        is_first_token = torch.sum(cluster_mat * pre_mask, dim=2, keepdim=True) == 0    # (B, L, 1)

        return cluster_means * is_first_token.float()


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

        unique_segs = self.drac(duplicate_segs, cluster_mat)

        return duplicate_segs, cluster_mat, unique_segs

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
        duplicate_segs, cluster_mat, unique_segs = self.decode(z)
        return z_mean, z_logvar, duplicate_segs, cluster_mat, unique_segs

from .Basics import *

class MultiHeadSelfRelationMatrix(nn.Module):
    def __init__(self, d_in: int, d_out: int, n_head: int, d_head: int):
        super().__init__()

        self.d_in = d_in
        self.d_out = d_out
        self.n_head = n_head
        self.d_head = d_head
        self.scale = self.d_head ** -0.5

        self.qk_proj = nn.Linear(d_in, n_head * d_head * 2)
        self.head_compressor = nn.Sequential(
            nn.Conv2d(d_head, d_out, 1, 1, 0),
            Swish(),
            nn.Conv2d(d_out, d_out, 1, 1, 0),
        )

    def forward(self, f_nodes):
        # src and dst: (B, N_nodes, d_out * d_head)
        q, k = self.qk_proj(f_nodes).split(self.n_head * self.d_head, dim=-1)
        q = rearrange(q, "B N (H C) -> (B H) N C", H=self.n_head)  # (B*O, N_nodes, C)
        kt = rearrange(k, "B N (H C) -> (B H) C N", H=self.n_head)  # (B*O, C, N_nodes)

        # mat: (B, O, N_nodes, N_nodes)
        mat = rearrange(q @ kt * self.scale, "(B H) R C -> B H R C", O=self.d_head)  # (B, O, N_nodes, N_nodes)

        return self.head_compressor(mat)   # (B, O, N_nodes, N_nodes)


class GraphEncoder(nn.Module):
    def __init__(self, d_latent: int, d_head: int, d_expand: int, d_hidden: int, n_heads: int, n_layers: int,
                 dropout: float = 0.1):
        super(GraphEncoder, self).__init__()
        # Initial projection from input to hidden dimension
        self.input_proj = nn.Linear(5, d_hidden)
        self.attention_blocks = nn.ModuleList([
            AttentionBlock(d_hidden, d_head, d_expand, d_hidden, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.mean_proj = nn.Linear(d_hidden, d_latent)
        self.logvar_proj = nn.Linear(d_hidden, d_latent)

    @staticmethod
    def reparameterize(z_mean, z_logvar):
        # Sample epsilon from standard normal distribution
        epsilon = torch.randn_like(z_mean)
        # Compute the latent vector
        z = z_mean + torch.exp(0.5 * z_logvar) * epsilon
        return z

    def forward(self, x):
        x = self.input_proj(x)  # Project input to hidden dimension
        for attn_block in self.attention_blocks:
            x = attn_block(x)  # Pass through each attention block

        # Compute mean and log-variance for the latent space
        z_mean = self.mean_proj(x)
        z_logvar = self.logvar_proj(x)

        return z_mean, z_logvar


class GraphDecoder(nn.Module):
    def __init__(self, d_latent: int, d_head: int, d_expand: int, d_hidden: int, n_heads: int,
                 n_layers: int, dropout: float = 0.1):
        super(GraphDecoder, self).__init__()
        # input shape: (B, N_segs, d_latent)
        self.latent_proj = nn.Linear(d_latent, d_hidden * 2)

        self.attention_blocks = nn.Sequential(*[
            AttentionBlock(d_hidden * 2, d_head, d_expand, d_hidden * 2, n_heads, dropout)
            for _ in range(n_layers)
        ])

        self.joint_proj = nn.Sequential(
            MultiHeadSelfRelationMatrix(d_hidden * 2, d_hidden, 32, 32),
            nn.GroupNorm(8, d_hidden),
            Swish(),
            nn.Conv2d(d_hidden, 32, 1, 1, 0),
            nn.GroupNorm(8, 32),
            Swish(),
            nn.Conv2d(32, 8, 1, 1, 0),
            Swish(),
            nn.Conv2d(8, 1, 1, 1, 0),
            nn.Flatten(1, 2),
            nn.Sigmoid()
        )

        self.segments_head = nn.Sequential(
            AttentionBlock(d_hidden * 2, d_head, d_expand, d_hidden, n_heads, dropout),
            nn.Linear(d_hidden, 5)
        )

    def forward(self, z):
        x = self.latent_proj(z)  # Project latent vector to hidden dimension
        x = self.attention_blocks(x)

        segs = self.segments_head(x)

        joints = self.joint_proj(x)

        segs[..., 4] = torch.sigmoid(segs[..., 4])    # Sigmoid for the valid mask

        return segs, joints

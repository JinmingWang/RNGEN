from .Basics import *

class EdgeBlock(nn.Module):
    def __init__(self, d_in: int, d_out: int, d_head: int):
        super().__init__()

        self.d_in = d_in
        self.d_out = d_out
        self.d_head = d_head
        self.scale = self.d_head ** -0.5

        self.edge_proj = nn.Linear(d_in, d_out * d_head * 2)
        self.adj_mat_proj = nn.Sequential(
            nn.Conv2d(d_out, d_out, 1, 1, 0),
            Swish(),
            nn.Conv2d(d_out, d_out, 1, 1, 0),
        )

    def forward(self, f_nodes):
        # src and dst: (B, N_nodes, d_out * d_head)
        src, dst = self.edge_proj(f_nodes).split(self.d_out * self.d_head, dim=-1)
        src = rearrange(src, "B N (O C) -> (B O) N C", O=self.d_out)  # (B*O, N_nodes, C)
        dst = rearrange(dst, "B N (O C) -> (B O) C N", O=self.d_out)  # (B*O, C, N_nodes)

        # mat: (B, O, N_nodes, N_nodes)
        mat = rearrange(src @ dst, "(B O) H W -> B O H W", O=self.d_out)  # (B, O, N_nodes, N_nodes)

        return self.adj_mat_proj(mat)   # (B, O, N_nodes, N_nodes)


class GraphEncoder(nn.Module):
    def __init__(self, d_in: int, d_latent: int, d_head: int, d_expand: int, d_hidden: int, n_heads: int, n_layers: int,
                 dropout: float = 0.1):
        super(GraphEncoder, self).__init__()
        # Initial projection from input to hidden dimension
        self.input_proj = nn.Linear(d_in, d_hidden)
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
    def __init__(self, d_latent: int, d_out: int, d_head: int, d_expand: int, d_hidden: int, n_heads: int,
                 n_layers: int, dropout: float = 0.1):
        super(GraphDecoder, self).__init__()
        # input shape: (B, N_segs, d_latent)
        self.latent_proj = nn.Sequential(
            Rearrange("B N (P D)", "B (P N) D", P=2),     # (B, 2, N_segs, d_latent//2)
            # Now there are N_nodes tokens
            nn.Linear(d_latent // 2, d_hidden)          # (B, 2, N_segs, d_hidden)
        )

        self.attention_blocks = nn.ModuleList([
            AttentionBlock(d_hidden, d_head, d_expand, d_hidden, n_heads, dropout)
            for _ in range(n_layers)
        ])

        self.mat_block = EdgeBlock(d_in=d_hidden, d_out=1, d_head=d_head)
        self.output_proj = nn.Linear(d_hidden, 2)

    def forward(self, z):
        x = self.latent_proj(z)  # Project latent vector to hidden dimension
        for attn_block in self.attention_blocks:
            x = attn_block(x)  # Pass through each attention block

        edges = torch.sigmoid(self.mat_block(x).squeeze(1))  # Compute adjacency matrix

        nodes = self.output_proj(x)  # Project back to output dimension
        return nodes, edges

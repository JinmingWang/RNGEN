from .Basics import *

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
        # Initial projection from latent space to hidden dimension
        self.latent_proj = nn.Linear(d_latent, d_hidden)
        self.attention_blocks = nn.ModuleList([
            AttentionBlock(d_hidden, d_head, d_expand, d_hidden, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.output_proj = nn.Linear(d_hidden, d_out)

    def forward(self, z):
        x = self.latent_proj(z)  # Project latent vector to hidden dimension
        for attn_block in self.attention_blocks:
            x = attn_block(x)  # Pass through each attention block

        x_recon = self.output_proj(x)  # Project back to output dimension
        return x_recon

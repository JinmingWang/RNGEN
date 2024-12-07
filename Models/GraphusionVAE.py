import torch

from .Basics import *


class EncoderBlock(nn.Module):
    def __init__(self, d_node: int, d_edge: int, need_edge: bool = True):
        super().__init__()

        self.d_node = d_node
        self.d_edge = d_edge
        self.need_edge = need_edge

        self.node_proj = self.__getLayer(d_node, d_node * d_edge)
        self.edge_proj = self.__getLayer(d_edge, d_edge)

        self.node_out_proj = self.__getLayer(d_node * d_edge, d_node)
        if need_edge:
            self.edge_out_proj = self.__getLayer(d_edge, d_edge)

    def __getLayer(self, d_in, d_out):
        return nn.Sequential(
            nn.LayerNorm(d_in),
            Swish(),
            nn.Linear(d_in, d_out)
        )

    def forward(self, f_nodes, f_edges, adj_mat):
        # f_nodes: (B, N, D_node)
        # f_edges: (B, N, N, D_edge)
        # adj_mat: (B, N, N)

        residual_nodes = f_nodes
        redisual_edges = f_edges

        f_nodes = rearrange(self.node_proj(f_nodes), "B N (H D) -> (B H) N D", H=self.d_edge)
        f_edges = self.edge_proj(f_edges)     # (B, N, N, n_heads)

        score = rearrange(f_edges, "B N M H -> B H N M")
        score = score.masked_fill((adj_mat == 0).unsqueeze(1).expand(-1, self.d_edge, -1, -1), -1e9)
        # mask = torch.masked_fill(adj_mat, 0, -1e9).unsqueeze(1).expand(-1, self.n_heads, -1, -1)    # (B, H, N, N)
        score = torch.softmax(score, dim=-1).flatten(0, 1)   # (B * H, N, N)

        f_nodes = rearrange(score @ f_nodes, "(B H) N D -> B N (H D)", H=self.d_edge)

        if self.need_edge:
            return self.node_out_proj(f_nodes) + residual_nodes, self.edge_out_proj(f_edges) + redisual_edges
        else:
            return self.node_out_proj(f_nodes) + residual_nodes


class DecoderBlock(nn.Module):
    def __init__(self, d_in: int, d_head: int, d_expand: int, d_out: int, n_heads: int):
        super(DecoderBlock, self).__init__()

        self.H = n_heads
        self.d_head = d_head
        self.d_in = d_in
        self.scale = nn.Parameter(torch.tensor(d_head, dtype=torch.float32))

        self.d_q = d_head * n_heads
        self.d_k = d_head * n_heads
        self.d_v = d_in * n_heads
        d_qkv = self.d_q + self.d_k + self.d_v

        self.norm_and_pos_enc = nn.Sequential(
            nn.LayerNorm(d_in),
            PositionalEncoding(d_in, 256),
        )

        self.qkv_proj = nn.Sequential(
            Swish(),
            nn.Linear(d_in, d_qkv),
        )

        self.merge_head_proj = nn.Linear(d_in * self.H, d_in)

        torch.nn.init.zeros_(self.merge_head_proj.weight)
        torch.nn.init.zeros_(self.merge_head_proj.bias)

        self.ff = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, d_expand),
            nn.LayerNorm(d_expand),
            Swish(),
            nn.Linear(d_expand, d_out),
        )

        torch.nn.init.zeros_(self.ff[-1].weight)
        torch.nn.init.zeros_(self.ff[-1].bias)

        self.shortcut = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()

    def forward(self, x):
        B, N, in_c = x.shape

        q, k, v = self.qkv_proj(self.norm_and_pos_enc(x)).split([self.d_q, self.d_k, self.d_v], dim=-1)
        q = rearrange(q, 'B N (H C) -> (B H) N C', H=self.H)  # (B*H, N, head_c)
        kt = rearrange(k, 'B N (H C) -> (B H) C N', H=self.H)  # (B*H, head_c, N)
        v = rearrange(v, 'B N (H C) -> (B H) N C', H=self.H)  # (B*H, N, in_c)

        # attn shape: (B*H, N, N)
        score = q @ kt * (self.scale ** -0.5)

        # out shape: (B, N, H*in_c)
        out = rearrange(torch.bmm(score, v), '(B H) N C -> B N (H C)', H=self.H, C=in_c)
        x = x + self.merge_head_proj(out)

        return self.ff(x) + self.shortcut(x)


class EdgeDecoder(nn.Module):
    def __init__(self, d_in, d_edge, d_hidden):
        super(EdgeDecoder, self).__init__()

        self.d_in = d_in
        self.d_edge = d_edge
        self.d_hidden = d_hidden

        self.in_proj = nn.Linear(d_in, d_hidden)

        self.out_proj = nn.Sequential(
            nn.Linear(d_hidden * 2, d_hidden),
            Swish(),
            nn.Linear(d_hidden, d_edge)
        )

    def forward(self, x):
        x = self.in_proj(x)

        # All-pairs concatenation, result shape: (B, N, N, 2D)
        x = torch.cat([x.unsqueeze(2).expand(-1, -1, x.size(1), -1),
                          x.unsqueeze(1).expand(-1, x.size(1), -1, -1)], dim=-1)

        return self.out_proj(x)




class GraphusionVAE(nn.Module):
    def __init__(self, d_node: int, d_edge: int, d_latent: int, d_hidden: int, n_layers:int, n_heads: int = 8):
        super().__init__()

        self.node_proj = nn.Linear(d_node, d_hidden)
        self.edge_proj = nn.Linear(d_edge, 16)

        self.encoder_layers = nn.ModuleList([
            *[EncoderBlock(d_hidden, 16) for _ in range(n_layers)],
            EncoderBlock(d_hidden, 16, need_edge=False)
        ])

        self.mu_head = nn.Linear(d_hidden, d_latent)
        self.logvar_head = nn.Linear(d_hidden, d_latent)

        self.decoder_layers = nn.Sequential(
            nn.Linear(d_latent, d_hidden),
            *[DecoderBlock(d_hidden, 64, 512, d_hidden, n_heads) for _ in range(n_layers + 2)]
        )

        self.degree_head = nn.Sequential(
            nn.Linear(d_hidden, 1),
            nn.Softplus(),
            nn.Flatten()
        )

        self.adj_mat_decoder = EdgeDecoder(d_hidden, 1, d_hidden)

        self.node_head = nn.Linear(d_hidden, d_node)
        torch.nn.init.zeros_(self.node_head.weight)
        torch.nn.init.zeros_(self.node_head.bias)

        self.edge_decoder = EdgeDecoder(d_hidden, d_edge, d_hidden*2)


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, f_nodes, f_edges, adj_mat):
        f_nodes = self.node_proj(f_nodes)
        f_edges = self.edge_proj(f_edges)
        for layer in self.encoder_layers[:-1]:
            f_nodes, f_edges = layer(f_nodes, f_edges, adj_mat)
        f_nodes = self.encoder_layers[-1](f_nodes, f_edges, adj_mat)
        return self.mu_head(f_nodes), self.logvar_head(f_nodes)


    def decode(self, z):
        x = self.decoder_layers(z)

        degrees = self.degree_head(x)

        adj_mat = self.adj_mat_decoder(x)    # (B, N, D)
        adj_mat = torch.sigmoid(adj_mat.squeeze(-1))  # (B, N, N)

        f_nodes = self.node_head(x)

        f_edges = self.edge_decoder(x) * adj_mat.unsqueeze(-1)

        return f_nodes, f_edges, adj_mat, degrees

    def forward(self, f_nodes, f_edges, adj_mat):
        mu, logvar = self.encode(f_nodes, f_edges, adj_mat)
        z = self.reparameterize(mu, logvar)
        return mu, logvar, *self.decode(z)




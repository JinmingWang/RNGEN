from .Basics import *

class NodeBlock(nn.Module):
    def __init__(self, d_in: int,
                 d_out: int,
                 d_traj_enc: int,
                 n_traj: int,
                 n_heads: int = 8,
                 dropout: float = 0.0
                 ):
        super().__init__()

        self.d_in = d_in
        self.d_out = d_out
        self.d_traj_enc = d_traj_enc
        self.n_traj = n_traj
        self.n_heads = n_heads
        self.dropout = dropout

        self.ca_1 = CrossAttentionBlock(d_in=d_in, d_context=d_traj_enc, d_head=d_in // 2, d_expand=d_out * 4,
                                        d_out=d_out, n_heads=self.n_heads, dropout=self.dropout)
        self.sa_1 = AttentionWithTime(d_in=d_out, d_head=d_out // 2, d_expand=d_out * 4, d_out=d_out,
                                      d_time=64, n_heads=self.n_heads, dropout=self.dropout)
        self.ca_2 = CrossAttentionBlock(d_in=d_out, d_context=d_traj_enc, d_head=d_out // 2, d_expand=d_out * 4,
                                        d_out=d_out, n_heads=self.n_heads, dropout=self.dropout)
        self.sa_2 = AttentionWithTime(d_in=d_out, d_head=d_out // 2, d_expand=d_out * 4, d_out=d_out,
                                      d_time=64, n_heads=self.n_heads, dropout=self.dropout)

    def forward(self, f_nodes, traj_enc, t):
        f_nodes = self.ca_1(f_nodes, traj_enc)
        f_nodes = self.sa_1(f_nodes, t)
        f_nodes = self.ca_2(f_nodes, traj_enc)
        f_nodes = self.sa_2(f_nodes, t)
        return f_nodes


class EdgeBlock(nn.Module):
    def __init__(self, d_node: int, d_in: int, d_out: int, d_head: int):
        super().__init__()

        self.d_in = d_in
        self.d_out = d_out
        self.d_head = d_head
        self.scale = self.d_head ** -0.5

        self.edge_proj = nn.Linear(d_node, d_out * d_head * 2)
        self.adj_mat_proj = nn.Sequential(
            nn.Conv2d(d_in + d_out, d_in + d_out, 1, 1, 0),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(d_in + d_out, d_out, 1, 1, 0),
        )

    def forward(self, f_nodes, f_adj):
        # src and dst: (B, N_nodes, d_out * d_head)
        src, dst = self.edge_proj(f_nodes).split(self.d_out * self.d_head, dim=-1)
        src = rearrange(src, "B N (O C) -> (B O) N C", O=self.d_out)  # (B*O, N_nodes, C)
        dst = rearrange(dst, "B N (O C) -> (B O) C N", O=self.d_out)  # (B*O, C, N_nodes)

        # mat: (B, O, N_nodes, N_nodes)
        mat = rearrange(src @ dst, "(B O) H W -> B O H W", O=self.d_out)  # (B, O, N_nodes, N_nodes)
        mat = torch.cat([f_adj, mat], dim=1)

        return self.adj_mat_proj(mat)   # (B, O, N_nodes, N_nodes)



class NodeEdgeModel(nn.Module):
    def __init__(self, n_nodes: int, d_traj_enc: int, n_traj: int, T: int):
        super().__init__()

        self.n_nodes = n_nodes
        self.n_traj = n_traj
        self.d_traj_enc = d_traj_enc

        self.time_embed = nn.Sequential(
            nn.Embedding(T, 128),
            nn.Linear(128, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(inplace=True),
        )

        self.stage_0 = nn.Sequential(
            nn.Linear(3, 64),
            AttentionBlock(d_in=64, d_head=32, d_expand=128, d_out=64, n_heads=8, dropout=0.0)
        )

        self.node_stage_1 = NodeBlock(64, 128, d_traj_enc, n_traj)
        self.edge_stage_1 = EdgeBlock(128, 1, 32, 64)
        self.node_stage_2 = NodeBlock(128, 256, d_traj_enc, n_traj)
        self.edge_stage_2 = EdgeBlock(256, 32, 32, 64)
        self.node_stage_3 = NodeBlock(256, 256, d_traj_enc, n_traj)
        self.edge_stage_3 = EdgeBlock(256, 32, 32, 64)
        self.node_stage_4 = NodeBlock(256, 256, d_traj_enc, n_traj)
        self.edge_stage_4 = EdgeBlock(256, 32, 1, 64)

        self.node_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 3)
        )

    def forward(self, nodes, adj_mat, traj_encoding, t):
        """
        :param nodes: (B, N_nodes, 2), 3 for lng, lat
        :param adj_mat: (B, N_nodes, N_nodes)
        :param traj_encoding: (B, N_traj=32, d_traj_enc=128)
        :return:
        """
        t = self.time_embed(t)  # (B, 64)

        f_nodes = self.stage_0(nodes)
        f_edges = adj_mat.unsqueeze(1)  # (B, 1, N_nodes, N_nodes)

        f_nodes = self.node_stage_1(f_nodes, traj_encoding, t)
        f_edges = self.edge_stage_1(f_nodes, f_edges)

        f_nodes = self.node_stage_2(f_nodes, traj_encoding, t)
        f_edges = self.edge_stage_2(f_nodes, f_edges)

        f_nodes = self.node_stage_3(f_nodes, traj_encoding, t)
        f_edges = self.edge_stage_3(f_nodes, f_edges)

        f_nodes = self.node_stage_4(f_nodes, traj_encoding, t)
        f_edges = self.edge_stage_4(f_nodes, f_edges)

        return self.node_head(f_nodes), f_edges.squeeze(1)

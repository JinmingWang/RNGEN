from .Basics import *

class NodeBlock(nn.Module):
    def __init__(self, in_c: int,
                 out_c: int,
                 traj_enc_c: int,
                 traj_num: int,
                 num_heads: int = 8,
                 dropout: float = 0.0
                 ):
        super().__init__()

        self.in_c = in_c
        self.out_c = out_c
        self.traj_enc_c = traj_enc_c
        self.traj_num = traj_num
        self.num_heads = num_heads
        self.dropout = dropout

        self.ca_1 = CrossAttentionBlock(in_c=in_c, context_c=traj_enc_c, head_c=in_c//4, expand_c=out_c*2,
                                      out_c=out_c, num_heads=self.num_heads, dropout=self.dropout)
        self.sa_1 = AttentionWithTime(in_c=out_c, head_c=out_c//4, expand_c=out_c*2, out_c=out_c,
                                      time_c=64, num_heads=self.num_heads, dropout=self.dropout)
        self.ca_2 = CrossAttentionBlock(in_c=out_c, context_c=traj_enc_c, head_c=out_c // 4, expand_c=out_c*2,
                                      out_c=out_c, num_heads=self.num_heads, dropout=self.dropout)
        self.sa_2 = AttentionWithTime(in_c=out_c, head_c=out_c//4, expand_c=out_c*2, out_c=out_c,
                                      time_c=64, num_heads=self.num_heads, dropout=self.dropout)

    def forward(self, f_nodes, traj_enc, t):
        f_nodes = self.ca_1(f_nodes, traj_enc)
        f_nodes = self.sa_1(f_nodes, t)
        f_nodes = self.ca_2(f_nodes, traj_enc)
        f_nodes = self.sa_2(f_nodes, t)
        return f_nodes


class EdgeBlock(nn.Module):
    def __init__(self, node_c: int, in_c: int, out_c: int, head_c: int):
        super().__init__()

        self.in_c = in_c
        self.out_c = out_c
        self.head_c = head_c
        self.scale = self.head_c ** -0.5

        self.edge_proj = nn.Linear(node_c, out_c * head_c * 2)
        self.adj_mat_proj = nn.Sequential(
            nn.Conv2d(in_c + out_c, in_c + out_c, 1, 1, 0),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_c + out_c, out_c, 1, 1, 0),
        )

    def forward(self, f_nodes, f_adj):
        # src and dst: (B, N_nodes, out_c * head_c)
        src, dst = self.edge_proj(f_nodes).split(self.out_c * self.head_c, dim=-1)
        src = rearrange(src, "B N (O C) -> (B O) N C", O=self.out_c)  # (B*O, N_nodes, C)
        dst = rearrange(dst, "B N (O C) -> (B O) C N", O=self.out_c)  # (B*O, C, N_nodes)

        # mat: (B, O, N_nodes, N_nodes)
        mat = rearrange(src @ dst, "(B O) H W -> B O H W", O=self.out_c)  # (B, O, N_nodes, N_nodes)
        mat = torch.cat([f_adj, mat], dim=1)

        return self.adj_mat_proj(mat)   # (B, O, N_nodes, N_nodes)



class DiffusionNetwork(nn.Module):
    def __init__(self, num_nodes: int, traj_encoding_c: int, traj_num: int, T: int):
        super().__init__()

        self.num_nodes = num_nodes
        self.traj_num = traj_num
        self.traj_encoding_c = traj_encoding_c

        self.time_embed = nn.Sequential(
            nn.Embedding(T, 128),
            nn.Linear(128, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(inplace=True),
        )

        self.stage_0 = nn.Sequential(
            nn.Linear(2, 64),
            AttentionBlock(in_c=64, head_c=32, expand_c=128, out_c=64, num_heads=8, dropout=0.0)
        )

        self.node_stage_1 = NodeBlock(64, 128, traj_encoding_c, traj_num)
        self.edge_stage_1 = EdgeBlock(128, 1, 32, 32)
        self.node_stage_2 = NodeBlock(128, 256, traj_encoding_c, traj_num)
        self.edge_stage_2 = EdgeBlock(256, 32, 32, 32)
        self.node_stage_3 = NodeBlock(256, 512, traj_encoding_c, traj_num)
        self.edge_stage_3 = EdgeBlock(512, 32, 1, 32)

        self.node_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 2)
        )

    def forward(self, nodes, adj_mat, traj_encoding, t):
        """
        :param nodes: (B, N_nodes, 2), 3 for lng, lat
        :param adj_mat: (B, N_nodes, N_nodes)
        :param traj_encoding: (B, N_traj=32, traj_encoding_c=128)
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

        return self.node_head(f_nodes), f_edges.squeeze(1)

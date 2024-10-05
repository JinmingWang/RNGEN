from .Basics import *

class Block(nn.Module):
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

        self.ca = CrossAttentionBlock(in_c=in_c, context_c=traj_enc_c, head_c=in_c//4, expand_c=out_c,
                                      out_c=out_c, num_heads=self.num_heads, dropout=self.dropout)
        self.sa_1 = AttentionWithTime(in_c=out_c, head_c=out_c//4, expand_c=out_c*2, out_c=out_c,
                                      time_c=64, num_heads=self.num_heads, dropout=self.dropout)
        self.sa_2 = AttentionWithTime(in_c=out_c, head_c=out_c//4, expand_c=out_c*2, out_c=out_c,
                                      time_c=64, num_heads=self.num_heads, dropout=self.dropout)

    def forward(self, x, traj_enc, t):
        x = self.ca(x, traj_enc)
        x = self.sa_1(x, t)
        x = self.sa_2(x, t)
        return x


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
            nn.Linear(3, 64),
            AttentionBlock(in_c=64, head_c=32, expand_c=128, out_c=64, num_heads=8, dropout=0.0)
        )

        self.stage_1 = Block(64, 128, traj_encoding_c, traj_num)

        self.stage_2 = Block(128, 256, traj_encoding_c, traj_num)

        self.stage_3 = Block(256, 512, traj_encoding_c, traj_num)

        self.node_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 3)
        )

        self.edge_proj = nn.Linear(512, 512)
        self.mat_proj = nn.Sequential(
            nn.Linear(17, 32),
            nn.LeakyReLU(inplace=True),
            nn.Linear(32, 32),
            nn.LeakyReLU(inplace=True),
            nn.Linear(32, 1)
        )

    def forward(self, nodes, adj_mat, traj_encoding, t):
        """
        :param nodes: (B, N_nodes, 3), 3 for lng, lat, pad_flag
        :param adj_mat: (B, N_nodes, N_nodes)
        :param traj_encoding: (B, N_traj=32, traj_encoding_c=128)
        :return:
        """
        t = self.time_embed(t)  # (B, 64)

        x = self.stage_0(nodes)
        x = self.stage_1(x, traj_encoding, t)
        x = self.stage_2(x, traj_encoding, t)
        x = self.stage_3(x, traj_encoding, t)

        # x: (B, N_nodes, 512)
        node_eps = self.node_head(x)    # (B, N_nodes, 2)

        src, dst = self.edge_proj(x).split(256, dim=-1)    # (B, N_nodes, 256), (B, N_nodes, 256)
        src = rearrange(src, "B N (H C) -> (B H) N C", H=16)     # (B*16, N_nodes, 16)
        dst = rearrange(dst, "B N (H C) -> (B H) C N", H=16)     # (B*16, 16, N_nodes)
        mat = rearrange(src @ dst, "(B C) H W -> B H W C", C=16)     # (B, 16, N_nodes, N_nodes)

        mat = torch.cat([adj_mat.unsqueeze(-1), mat], dim=-1)    # (B, N_nodes, N_nodes, 17)
        adj_mat_eps = self.mat_proj(mat).squeeze(-1)    # (B, N_nodes, N_nodes)

        return node_eps, adj_mat_eps


    def trainStep(self, optimizer, encoder, **data):
        optimizer.zero_grad()

        traj_enc = encoder(data['trajs'])

        reconstruct_trajs = self()
        loss_list = []
        B = len(data['trajs'])
        for b in range(B):
            N = data['trajs'][b].shape[0]
            loss_list.append(self.loss_func(reconstruct_trajs[b, :N], data['trajs'][b]))
        loss = torch.stack(loss_list).mean()
        loss.backward()
        optimizer.step()
        return loss.item(), reconstruct_trajs

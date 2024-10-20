from .Basics import *

class Block(nn.Module):
    def __init__(self, d_in: int,
                 d_out: int,
                 d_traj_enc: int,
                 n_heads: int = 8,
                 dropout: float = 0.0,
                 ):
        super().__init__()

        self.d_in = d_in
        self.d_out = d_out
        self.d_traj_enc = d_traj_enc
        self.n_heads = n_heads
        self.dropout = dropout

        self.ca = CrossAttentionBlockWithTime(d_in=d_in, d_context=d_traj_enc, d_head=d_out // 4, d_expand=d_out * 2,
                                        d_out=d_out, d_time=128, n_heads=self.n_heads, dropout=self.dropout)
        self.sa = AttentionWithTime(d_in=d_out, d_head=d_out // 4, d_expand=d_out * 2, d_out=d_out,
                                      d_time=128, n_heads=self.n_heads, dropout=self.dropout)

    def forward(self, f_segs, traj_enc, t):
        f_segs = self.ca(f_segs, traj_enc, t)
        f_segs = self.sa(f_segs, t)
        return f_segs


class SegmentsModel(nn.Module):
    def __init__(self, d_in: int, d_traj_enc: int, n_layers: int, T: int, pred_x0: bool = False):
        super().__init__()

        self.d_in = d_in
        self.d_traj_enc = d_traj_enc
        self.pred_x0 = pred_x0
        self.n_layers = n_layers

        self.time_embed = nn.Sequential(
            nn.Embedding(T, 128),
            nn.Linear(128, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 128),
            nn.LeakyReLU(inplace=True),
        )

        self.linear = nn.Linear(d_in, 64)

        self.attn = AttentionWithTime(d_in=64, d_head=32, d_expand=128, d_out=256, d_time=128, n_heads=8, dropout=0.0)

        self.stages = SequentialWithAdditionalInputs(
            *[Block(256, 256, d_traj_enc) for _ in range(n_layers)]
        )

        self.head = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, d_in)
        )

    def forward(self, segments, traj_encoding, t):
        """
        :param segments: (B, N_segs, C_seg)
        :param traj_encoding: (B, N_traj=32, traj_encoding_c=128)
        :return:
        """
        t = self.time_embed(t)

        x = self.linear(segments)

        x = self.attn(x, t)

        x = self.stages(x, traj_encoding, t)

        # x: (B, N_segs, 512)
        out = self.head(x)
        # if self.pred_x0:
        #     x_center, y_center, direction, length, valid_mask = torch.unbind(out, dim=-1)
        #     segs = torch.stack([
        #         x_center,
        #         y_center,
        #         torch.sigmoid(direction) * torch.pi,
        #         length,
        #         torch.sigmoid(valid_mask)
        #     ], dim=-1)
        #     return segs

        return out

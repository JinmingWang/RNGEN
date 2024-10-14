from .Basics import *
from .HungarianLoss_SeqMat import HungarianLoss

class Block(nn.Module):
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

        self.ca = CrossAttentionBlock(d_in=d_in, d_context=d_traj_enc, d_head=d_out // 4, d_expand=d_out * 2,
                                        d_out=d_out, n_heads=self.n_heads, dropout=self.dropout)
        self.sa = AttentionWithTime(d_in=d_out, d_head=d_out // 4, d_expand=d_out * 2, d_out=d_out,
                                      d_time=64, n_heads=self.n_heads, dropout=self.dropout)

    def forward(self, f_segs, traj_enc, t):
        f_segs = self.ca(f_segs, traj_enc)
        f_segs = self.sa(f_segs, t)
        return f_segs


class SegmentsModel(nn.Module):
    def __init__(self, d_seg: int, n_seg: int, d_traj_enc: int, n_traj: int, T: int):
        super().__init__()

        self.d_seg = d_seg
        self.n_seg = n_seg
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
            nn.Linear(d_seg, 64),
            AttentionBlock(d_in=64, d_head=32, d_expand=128, d_out=64, n_heads=8, dropout=0.0)
        )

        self.stages = SequentialWithAdditionalInputs(
            Block(64, 128, d_traj_enc, n_traj),
            Block(128, 128, d_traj_enc, n_traj),

            Block(128, 256, d_traj_enc, n_traj),
            Block(256, 256, d_traj_enc, n_traj),

            Block(256, 256, d_traj_enc, n_traj),
            Block(256, 256, d_traj_enc, n_traj),

            Block(256, 256, d_traj_enc, n_traj),
            Block(256, 256, d_traj_enc, n_traj),

            Block(256, 256, d_traj_enc, n_traj),
            Block(256, 256, d_traj_enc, n_traj),

            Block(256, 256, d_traj_enc, n_traj),
            Block(256, 256, d_traj_enc, n_traj),
        )

        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, d_seg)
        )

    def forward(self, segments, traj_encoding, t):
        """
        :param segments: (B, N_segs, C_seg)
        :param traj_encoding: (B, N_traj=32, traj_encoding_c=128)
        :return:
        """
        t = self.time_embed(t)

        x = self.stage_0(segments)

        x = self.stages(x, traj_encoding, t)

        # x: (B, N_segs, 512)
        return self.head(x)

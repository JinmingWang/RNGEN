from .Basics import *
from .HungarianLoss import HungarianLoss

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
        self.sa_1 = AttentionBlock(in_c=out_c, head_c=out_c//4, expand_c=out_c*2, out_c=out_c,
                                   num_heads=self.num_heads, dropout=self.dropout)
        self.sa_2 = AttentionBlock(in_c=out_c, head_c=out_c//4, expand_c=out_c*2, out_c=out_c,
                                   num_heads=self.num_heads, dropout=self.dropout)

    def forward(self, x, traj_enc):
        x = self.ca(x, traj_enc)
        x = self.sa_1(x)
        x = self.sa_2(x)
        return x


class DiffusionNetwork(nn.Module):
    def __init__(self, segment_c: int, num_segments: int, traj_encoding_c: int, traj_num: int):
        super().__init__()

        self.segment_c = segment_c
        self.num_segments = num_segments
        self.traj_num = traj_num
        self.traj_encoding_c = traj_encoding_c

        self.stage_0 = nn.Sequential(
            nn.Linear(segment_c, 64),
            AttentionBlock(in_c=64, head_c=32, expand_c=128, out_c=64, num_heads=8, dropout=0.0),
            AttentionBlock(in_c=64, head_c=32, expand_c=128, out_c=64, num_heads=8, dropout=0.0)
        )

        self.stage_1 = Block(64, 128, traj_encoding_c, traj_num)

        self.stage_2 = Block(128, 256, traj_encoding_c, traj_num)

        self.stage_3 = Block(256, 512, traj_encoding_c, traj_num)

        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, segment_c)
        )

        self.loss_func = HungarianLoss()

    def forward(self, segments, traj_encoding):
        """
        :param segments: (B, N_segs, C_seg)
        :param traj_encoding: (B, N_traj=32, traj_encoding_c=128)
        :return:
        """
        x = self.stage_0(segments)
        x = self.stage_1(x, traj_encoding)
        x = self.stage_2(x, traj_encoding)
        x = self.stage_3(x, traj_encoding)

        # x: (B, N_segs, 512)
        return self.head(x)

    def trainStep(self, optimizer, encoder, **data):
        optimizer.zero_grad()

        traj_enc = encoder(data['trajs'])

        pred_noise = self(data["segments"], traj_enc)

        loss = self.loss_func(pred_noise, data["noise"])

        loss.backward()
        optimizer.step()
        return loss.item()

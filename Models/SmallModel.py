from .Basics import *


class SmallModel(nn.Module):
    def __init__(self, in_c: int, traj_encoding_c: int):
        super().__init__()

        self.traj_encoding_c = traj_encoding_c

        self.stem = nn.Sequential(
            nn.Linear(in_c, 64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, 128),
        )

        self.cross_1 = CrossAttentionBlock(128, traj_encoding_c, 32, 128, 128, 8)
        self.s1 = nn.Sequential(
            AttentionBlock(128, 32, 512, 128, 8),
            AttentionBlock(128, 32, 512, 256, 8),
        )

        self.cross_2 = CrossAttentionBlock(256, traj_encoding_c, 64, 256, 256, 8)
        self.s2 = nn.Sequential(
            AttentionBlock(256, 64, 1024, 256, 8),
            AttentionBlock(256, 64, 1024, 512, 8),
        )

        self.cross_3 = CrossAttentionBlock(512, traj_encoding_c, 128, 512, 512, 8)
        self.s3 = nn.Sequential(
            AttentionBlock(512, 128, 2048, 512, 8),
            AttentionBlock(512, 128, 2048, 512, 8),
        )

        self.head = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, in_c),
        )

    def forward(self, x, traj_encoding):
        x = self.stem(x)
        x = self.cross_1(x, traj_encoding)
        x = self.s1(x)
        x = self.cross_2(x, traj_encoding)
        x = self.s2(x)
        x = self.cross_3(x, traj_encoding)
        x = self.s3(x)
        x = self.head(x)
        return x

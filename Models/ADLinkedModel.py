# ADLinked Model

from .Basics import *

class CrissCrossAttention(nn.Module):
    @cacheArgumentsUponInit
    def __init__(self, d_in: int):
        super(CrissCrossAttention, self).__init__()

        # Typical Attention in CV on data of shape (B, C, H, W) converts data to (B, N, C) where N = H * W
        # Then apply normal attention mechanism
        # So the score is computed between each pixel and all pixels in the image

        # Criss-Cross attention, however, computes the score between each pixel and all pixels in the currecnt row and column
        # The Q will be (B, H*W, C) as usual
        # The K and V will be (B, (H + W - 1), C)

        self.qkv_proj = nn.Conv2d(d_in, d_in * 3, 1, 1, 0)

        self.gamma = nn.Parameter(torch.zeros(1))


    def forward(self, x):
        B, C, H, W = x.shape

        q, k, v = self.qkv_proj(x).split([C, C, C], dim=1)

        q_H = rearrange(q, "B C H W -> (B W) H C")
        q_W = rearrange(q, "B C H W -> (B H) W C")

        k_H = rearrange(k, "B C H W -> (B W) H C")
        k_W = rearrange(k, "B C H W -> (B H) W C")

        v_H = rearrange(v, "B C H W -> (B W) H C")
        v_W = rearrange(v, "B C H W -> (B H) W C")

        # score_H is the attention score within each column
        # The elements within this column is compared to all elements in the column
        score_H = q_H @ k_H.transpose(1, 2)     # (B*W, H, H)
        score_W = q_W @ k_W.transpose(1, 2)     # (B*H, W, W)

        score = torch.cat([
            rearrange(score_H, "(B W) H1 H2 -> B H1 W H2", W=W),
            rearrange(score_W, "(B H) W1 W2 -> B H W1 W2", H=H)
        ], dim=-1)  # (B, H, W, H + W)

        score_H = rearrange(score[..., :H], "B H1 W H2 -> (B W) H1 H2")
        score_W = rearrange(score[..., H:], "B H W1 W2 -> (B H) W1 W2")

        out_H = rearrange(score_H @ v_H, "(B W) H C -> B C H W", W=W)
        out_W = rearrange(score_W @ v_W, "(B H) W C -> B C H W", H=H)

        return self.gamma * (out_H + out_W) + x


class HybridDilatedConvolution(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()

        self.dilated_convs = nn.Sequential(
            nn.Conv2d(d_in, d_in, 3, 1, 1, dilation=1),
            nn.BatchNorm2d(d_in), Swish(),
            nn.Conv2d(d_in, d_in, 3, 1, 2, dilation=2),
            nn.BatchNorm2d(d_in), Swish(),
            nn.Conv2d(d_in, d_in, 3, 1, 5, dilation=5)
        )

        self.conv_out = nn.Conv2d(d_in, d_in, 1, 1, 0)

    def forward(self, x):
        return self.conv_out(x + self.dilated_convs(x))


class AD_Block(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.cca = CrissCrossAttention(d_in)
        self.hdc = HybridDilatedConvolution(d_in)
        self.out_proj = nn.Conv2d(d_in, d_in, 1, 1, 0)

    def forward(self, x):
        return self.out_proj(self.hdc(x) + self.cca(x))


class ResBlock(nn.Sequential):
    def __init__(self, d_in: int):
        super().__init__(
            nn.Conv2d(d_in, d_in, 3, 1, 1),
            nn.BatchNorm2d(d_in), Swish(),
            nn.Conv2d(d_in, d_in, 3, 1, 1),
            nn.BatchNorm2d(d_in)
        )

    def forward(self, x):
        return x + super().forward(x)


class AD_Linked_Net(nn.Module):
    @cacheArgumentsUponInit
    def __init__(self, d_in: int, H: int, W: int):
        super().__init__()

        self.in_proj = nn.Sequential(
            nn.Conv2d(d_in, 3, 1, 1, 0),
            nn.BatchNorm2d(3), Swish(),
            nn.Conv2d(3, 64, 7, 2, 3),
            nn.BatchNorm2d(64), Swish()
        )

        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.MaxPool2d(2),
                *[ResBlock(64) for _ in range(3)]
            ),
            nn.Sequential(
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, 1, 1),
                nn.BatchNorm2d(128), Swish(),
                *[ResBlock(128) for _ in range(4)]
            ),
            nn.Sequential(
                nn.MaxPool2d(2),
                nn.Conv2d(128, 256, 3, 1, 1),
                nn.BatchNorm2d(256), Swish(),
                *[ResBlock(256) for _ in range(6)]
            ),
            nn.Sequential(
                nn.MaxPool2d(2),
                *[ResBlock(256) for _ in range(3)]
            )
        ])

        self.ad_blocks = nn.ModuleList([
            AD_Block(64),
            AD_Block(128),
            AD_Block(256),
            AD_Block(256)
        ])

        self.upsamples = nn.ModuleList([
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Sequential(nn.Conv2d(256, 128, 1, 1, 0),
                          nn.UpsamplingNearest2d(scale_factor=2)),
            nn.Sequential(nn.Conv2d(128, 64, 1, 1, 0),
                            nn.UpsamplingNearest2d(scale_factor=2))
        ])

        self.out_proj = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.BatchNorm2d(3), Swish(),
            nn.Conv2d(3, 1, 3, 1, 1),
            nn.Sigmoid()
        )


    def forward(self, x):
        x = self.in_proj(x)

        tmp_list = []
        for stage, ad_block in zip(self.stages, self.ad_blocks):
            x = stage(x)
            tmp_list.append(ad_block(x))

        x = tmp_list[-1]
        x = self.upsamples[0](x) + tmp_list[-2]
        x = self.upsamples[1](x) + tmp_list[-3]
        x = self.upsamples[2](x) + tmp_list[-4]

        return self.out_proj(x)








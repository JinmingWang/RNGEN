from .Basics import *


class Res(nn.Sequential):
    def __init__(self, in_c: int, expand:int = 4):
        mid_c = in_c * expand
        super().__init__(
            nn.Conv2d(in_c, mid_c, 1, 1, 0),
            nn.BatchNorm2d(mid_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_c, mid_c, 3, 1, 1),
            nn.BatchNorm2d(mid_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_c, in_c, 1, 1, 0)
        )
        
    def forward(self, x):
        return x + super().forward(x)


class NodeExtractor(nn.Sequential):
    def __init__(self):
        super(NodeExtractor, self).__init__(
            nn.Conv2d(1, 16, 5, 2, 2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            Res(16),
            Res(16),
            Res(16),

            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(16, 1, 5, 1, 2),
            nn.Sigmoid()
        )

        # Receptive Field: 3+2+2+2+2=11
        # A Small module just to extract nodes from a thin binary image
        # Becuase the traditional CV algorithms are not robust enough, cannot generalize well, involves tedious parameter tuning
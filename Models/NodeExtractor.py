from .Basics import *

class NodeExtractor(nn.Sequential):
    def __init__(self):
        super(NodeExtractor, self).__init__(
            nn.Conv2d(1, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, 1, 1),
            nn.Sigmoid()
        )

        # Receptive Field: 3+2+2+2=9
        # A Small module just to extract nodes from a thin binary image
        # Becuase the traditional CV algorithms are not robust enough, cannot generalize well, involves tedious parameter tuning
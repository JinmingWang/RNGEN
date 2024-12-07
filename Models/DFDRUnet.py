import torch

from .Basics import *

class DilationRes(nn.Module):
    def __init__(self, d_in: int, d_mid: int, d_out: int):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(d_in, d_mid, 3, 1, 1),
            nn.BatchNorm2d(d_mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_mid, d_out, 3, 1, 2, dilation=2),
        )

        self.shortcut = nn.Identity() if d_in == d_out else nn.Conv2d(d_in, d_out, 1)

    def forward(self, x):
        return self.layers(x) + self.shortcut(x)


class GatedFusionModule(nn.Module):
    def __init__(self, d_in : int):
        super().__init__()

        self.proj_A = nn.Conv2d(d_in, d_in//2, 1)
        self.proj_B = nn.Conv2d(d_in, d_in//2, 1)

        self.proj_merge = nn.Sequential(
            nn.Conv2d(d_in, d_in, 3, 1, 1),
            nn.BatchNorm2d(d_in),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_in, d_in, 3, 1, 1),
            nn.BatchNorm2d(d_in),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_in, d_in // 2, 1),
            nn.Softmax(dim=1)
        )

        self.upsample = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            nn.Conv2d(d_in, d_in // 2, 1),
        )

        self.hidden_proj = nn.Sequential(
            nn.Conv2d(d_in, d_in//2, 3, 1, 1),
            nn.BatchNorm2d(d_in//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_in//2, d_in//2, 3, 1, 1),
        )


    def forward(self, A, B, hidden):
        A = self.proj_A(A)
        B = self.proj_B(B)
        AB = self.proj_merge(torch.cat([A, B], dim=1))

        AB = A * AB + B * AB

        hidden = self.upsample(hidden)

        hidden = self.hidden_proj(torch.cat([AB, hidden], dim=1))

        return AB + hidden


class DRUNet(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()

        self.enc0 = DilationRes(d_in, 16, 16)
        self.enc1 = DilationRes(16, 32, 32)
        self.enc2 = DilationRes(32, 64, 64)
        self.enc3 = DilationRes(64, 128, 128)
        self.enc4 = DilationRes(128, 256, 256)

        self.dec0 = DilationRes(256 + 128, 128, 128)
        self.dec1 = DilationRes(128 + 64, 64, 64)
        self.dec2 = DilationRes(64 + 32, 32, 32)
        self.dec3 = DilationRes(32 + 16, 16, 16)

        self.downsample = nn.MaxPool2d(2)
        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)

    def forward(self, x):
        x0 = self.enc0(x)
        x1 = self.enc1(self.downsample(x0))
        x2 = self.enc2(self.downsample(x1))
        x3 = self.enc3(self.downsample(x2))
        x4 = self.enc4(self.downsample(x3))

        out0 = x4
        out1 = self.dec0(torch.cat([x3, self.upsample(out0)], dim=1))
        out2 = self.dec1(torch.cat([x2, self.upsample(out1)], dim=1))
        out3 = self.dec2(torch.cat([x1, self.upsample(out2)], dim=1))
        out4 = self.dec3(torch.cat([x0, self.upsample(out3)], dim=1))

        return [out0, out1, out2, out3, out4]


class DFDRUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.drunet_image = DRUNet(3)

        self.drunet_traj = DRUNet(1)

        self.init_feature = nn.Parameter(torch.zeros(1, 256, 8, 8))

        self.gfm0 = GatedFusionModule(256)
        self.gfm1 = GatedFusionModule(128)
        self.gfm2 = GatedFusionModule(64)
        self.gfm3 = GatedFusionModule(32)
        self.gfm4 = GatedFusionModule(16)

        self.head = nn.Sequential(
            nn.Conv2d(8, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, image, traj):
        B = image.shape[0]
        image_features = self.drunet_image(image)
        traj_features = self.drunet_traj(traj)

        fuse_features = self.gfm0(image_features[0], traj_features[0], self.init_feature.expand(B, -1 ,-1, -1))
        fuse_features = self.gfm1(image_features[1], traj_features[1], fuse_features)
        fuse_features = self.gfm2(image_features[2], traj_features[2], fuse_features)
        fuse_features = self.gfm3(image_features[3], traj_features[3], fuse_features)
        fuse_features = self.gfm4(image_features[4], traj_features[4], fuse_features)

        return self.head(fuse_features)




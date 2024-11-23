from .Basics import *

# Define a complex UNet block

class UNet2D(nn.Module):
    @cacheArgumentsUponInit
    def __init__(self, n_repeats: int = 2, expansion: int = 2):
        """
        Encoder:
        (256, 256) -> (128, 128) -> skip_connection_out ->
        Neck:
        (64, 64) -> (64, 64) ->
        Decoder:
        skip_connection_in -> (128, 128) -> (256, 256)

        """
        super(UNet2D, self).__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(1, 8, 3, 1, 1),
            Swish()
        )

        self.enc1 = self.getStage(8, 32, "down")
        self.enc2 = self.getStage(32, 64, "down")

        self.neck = self.getStage(64, 64, "same")

        # After skip connection: (128+128, 64, 64)

        self.dec1 = self.getStage(128, 32, "up")
        self.dec2 = self.getStage(32, 8, "up")

        self.head = nn.Sequential(
            nn.Conv2d(8, 1, 1),
            nn.Sigmoid()
        )

    def getStage(self, d_in: int, d_out: int, scaling: Literal["up", "down", "same"]):
        """
        A stage in the UNet architecture
        """

        d_mid = d_in * self.expansion
        layers = nn.Sequential(
            *[Res2D(d_in, d_mid, d_in) for _ in range(self.n_repeats)],
            Conv2dNormAct(d_in, d_out, 1, 1, 0),
        )

        if scaling == "down":
            layers.insert(0, nn.MaxPool2d(2))
        elif scaling == "up":
            layers.append(nn.UpsamplingNearest2d(scale_factor=2))

        return layers

    def forward(self, x):
        x = self.stem(x)

        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)

        neck = self.neck(enc2)

        # Skip connection
        skip = torch.cat([neck, enc2], dim=1)

        dec1 = self.dec1(skip)
        dec2 = self.dec2(dec1)

        return self.head(dec2)

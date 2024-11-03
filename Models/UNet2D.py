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

        self.stem = Conv2dBnAct(1, 32, 3, 1, 1)     # (32, 256, 256)

        self.enc1 = self.getStage(32, 64, "down")   # (64, 128, 128)
        self.enc2 = self.getStage(64, 128, "down")  # (128, 64, 64)

        self.neck = self.getStage(128, 128, "same")    # (256, 64, 64)

        # After skip connection: (128+128, 64, 64)

        self.dec1 = self.getStage(256, 64, "up")   # (64, 128, 128)
        self.dec2 = self.getStage(64, 32, "up")    # (32, 256, 256)

        self.head = nn.Sequential(
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )

    def getStage(self, d_in: int, d_out: int, scaling: Literal["up", "down", "same"]):
        """
        A stage in the UNet architecture
        """
        if scaling == "same":
            final_layer = nn.Identity()
        elif scaling == "down":
            final_layer = nn.MaxPool2d(2)
        else:
            final_layer = nn.UpsamplingNearest2d(scale_factor=2)

        d_mid = d_in * self.expansion
        return nn.Sequential(
            *[Res2D(d_in, d_mid, d_in) for _ in range(self.n_repeats)],
            Conv2dBnAct(d_in, d_out, 3, 1, 1),
            final_layer
        )

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

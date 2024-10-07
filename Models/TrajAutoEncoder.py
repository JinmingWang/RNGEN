from .Basics import *


class Encoder(nn.Module):
    def __init__(self, N_trajs: int, traj_len: int, encode_c: int):
        super().__init__()

        self.N_trajs = N_trajs
        self.traj_len = traj_len
        self.mid_c = traj_len * 2
        self.encode_c = encode_c

        self.attn_blocks = nn.ModuleList([
            AttentionBlock(self.mid_c, 32, 512, self.mid_c, 32, dropout=0.0)
            for _ in range(4)])

    def forward(self, trajs: Tensor) -> Tensor:
        """
        :param trajs: list of (N, 128, 2) tensors
        :return: tensor of shape (B, max_trajs, encode_c)
        """
        B = len(trajs)
        trajs = rearrange(trajs, "B N L C -> B N (L C)")
        for attn_block in self.attn_blocks:
            trajs = attn_block(trajs)
        return trajs


class Decoder(nn.Module):
    def __init__(self, N_trajs: int, traj_len: int, encode_c: int, decode_len: int):
        super().__init__()

        self.N_trajs = N_trajs
        self.traj_len = traj_len
        self.mid_c = traj_len * 2
        self.encode_c = encode_c
        self.decode_len = decode_len

        self.stem = nn.Linear(self.encode_c, self.mid_c)

        self.attn_blocks = nn.ModuleList([
            AttentionBlock(self.mid_c, 32, 512, self.mid_c, 32, dropout=0.0)
            for _ in range(4)])

        self.conv_layers = nn.Sequential(  # input: (B * N, 256)
            nn.Linear(self.mid_c, 64 * decode_len),
            nn.Unflatten(1, (64, decode_len)),
            Res1D(64, 256, 64),
            Res1D(64, 256, 64),
            nn.Conv1d(64, 2, 3, 1, 1),  # output: (B * N, 2, 128)
            Transpose(1, 2)     # (B * N, decode_len, 2)
        )

    def forward(self, traj_enc: Tensor) -> Tensor:
        """
        :param traj_enc: tensor of shape (B, max_trajs, encode_c)
        :return: tensor of shape (B, max_trajs, 128, 2)
        """
        B = traj_enc.shape[0]

        traj_enc = self.stem(traj_enc)

        for attn_block in self.attn_blocks:
            traj_enc = attn_block(traj_enc)

        traj_enc = traj_enc.view(B * self.N_trajs, self.mid_c)
        return self.conv_layers(traj_enc).view(B, self.N_trajs, self.decode_len, 2)



if __name__ == "__main__":
    encoder = Encoder(32, 128)
    decoder = Decoder(32, 128)

    trajs = torch.stack([torch.randn(32, 128, 2) for _ in range(4)], dim=0)

    enc= encoder(trajs)
    print(enc.shape)
    reconstructed_trajs = decoder(enc)
    print(reconstructed_trajs.shape)

    torch.save(encoder.state_dict(), "encoder.pth")
    torch.save(decoder.state_dict(), "decoder.pth")
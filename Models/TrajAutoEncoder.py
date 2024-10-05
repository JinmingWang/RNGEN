from .Basics import *


class Encoder(nn.Module):
    def __init__(self, N_trajs: int, traj_len: int = 128):
        super().__init__()

        self.N_trajs = N_trajs
        self.traj_len = traj_len
        self.mid_c = traj_len * 2
        self.encode_c = traj_len

        self.conv_layers = nn.Sequential(     # (BN, L, 2)
            Transpose(1, 2),    # (BN, 2, L)

            Conv1dBnAct(2, 16, 3, 1, 1),
            Conv1dBnAct(16, 16, 3, 2, 1),

            Conv1dBnAct(16, 32, 3, 1, 1),
            Conv1dBnAct(32, 32, 3, 2, 1),

            Conv1dBnAct(32, 64, 3, 1, 1),
            Conv1dBnAct(64, 64, 3, 2, 1),

            Conv1dBnAct(64, 128, 3, 1, 1),
            Conv1dBnAct(128, 128, 3, 2, 1),
            nn.Flatten(1),  # (BN, 128, L//16)
            nn.Linear(traj_len * 8, 512),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, self.mid_c),
        )

        self.attn_blocks = nn.ModuleList([
            AttentionBlock(self.mid_c, 64, self.mid_c, self.mid_c, 8, dropout=0.0)
            for _ in range(4)])

        self.out_proj = nn.Linear(self.mid_c, self.encode_c)


    def forward(self, trajs: Tensor) -> Tensor:
        """
        :param trajs: list of (N, 128, 2) tensors
        :return: tensor of shape (B, max_trajs, encode_c)
        """
        B = len(trajs)
        trajs = trajs.view(-1, self.traj_len, 2)     # (B * N, 128, 2)
        traj_proj = self.conv_layers(trajs).view(B, self.N_trajs, self.encode_c * 2)  # (B * N, encode_c)

        for attn_block in self.attn_blocks:
            traj_proj = attn_block(traj_proj)

        return self.out_proj(traj_proj)


class Decoder(nn.Module):
    def __init__(self, N_trajs: int, traj_len: int = 128):
        super().__init__()

        self.N_trajs = N_trajs
        self.traj_len = traj_len
        self.mid_c = traj_len * 2
        self.encode_c = traj_len

        self.stem = nn.Linear(self.encode_c, self.mid_c)

        self.attn_blocks = nn.ModuleList([
            AttentionBlock(self.mid_c, 64, 512, self.mid_c, 8, dropout=0.0)
            for _ in range(4)])

        self.conv_layers = nn.Sequential(  # input: (B * N, 256)
            nn.Linear(self.mid_c, 1024),
            nn.LeakyReLU(inplace=True),
            nn.Unflatten(1, (8, 128)),
            Res1D(8, 32, 8),
            Res1D(8, 32, 8),
            nn.Conv1d(8, 2, 3, 1, 1),  # output: (B * N, 2, 128)
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
        return self.conv_layers(traj_enc).view(B, self.N_trajs, self.traj_len, 2)



class TrajAutoEncoder(nn.Module):
    def __init__(self,
                 traj_len: int = 128,
                 encode_c: int = 128,
                 N_trajs: int = 32,
                 include_decoder: bool = True):
        super().__init__()
        self.encoder = Encoder(N_trajs, traj_len)
        if include_decoder:
            self.decoder = Decoder(N_trajs, encode_c)

        self.loss_func = nn.MSELoss()

    def forward(self, trajs: Tensor):
        enc = self.encoder(trajs)
        if hasattr(self, 'decoder'):
            return self.decoder(enc)
        return enc

    def trainStep(self, optimizer: torch.optim.Optimizer, **data):
        optimizer.zero_grad()
        reconstruct_trajs = self(data['trajs'])
        loss_list = []
        B = len(data['trajs'])
        for b in range(B):
            N = data['trajs'][b].shape[0]
            loss_list.append(self.loss_func(reconstruct_trajs[b, :N], data['trajs'][b]))
        loss = torch.stack(loss_list).mean()
        loss.backward()
        optimizer.step()
        return loss.item(), reconstruct_trajs


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
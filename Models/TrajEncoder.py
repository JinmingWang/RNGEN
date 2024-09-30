from .Basics import *

class TrajEncoder(nn.Module):
    def __init__(self, grid_size: int = 16):
        super(TrajEncoder, self).__init__()

        # batch trajectories have dynamic shapes, for example:
        # [(50, 256, 2), (77, 256, 2), (168, 256, 2), ..., (134, 256, 2)]
        # There are B samples in this list
        # each sample contains arbitrary number of trajectories
        # each trajectory are padded or truncated to have 256 points
        # each point has 2 coordinates (lng, lat)

        # The encoder should output a fixed-size vector for each sample

        # For each sample, our goal is to encode (N, 256, 2) to something like (H=gs, W=gs, C=64) <=> (gs^2, 64)

        # We will use a transformer to encode the trajectories

        self.grid_size = grid_size

        row_enc = torch.linspace(-3, 3, grid_size)
        col_enc = torch.linspace(-3, 3, grid_size)

        # pos_encoding shape: (32, 32, 64), pos_encoding[i, j] = concat(row_enc[i], col_enc[j])
        pos_encoding = torch.stack(torch.meshgrid(row_enc, col_enc), dim=-1)    # (gs, gs, 2)

        # gs^2 cells, 64 features per cell
        self.grid_enc = nn.Parameter(pos_encoding.reshape(-1, 2))  # (gs^2, 2)

        # So pos_encoding contains positions, nearly evenly distributed in the range [-3, 3]
        # trajs also contain positions, but they are not evenly distributed
        # This makes it possible to compute attention between pos_encoding and trajs

        self.traj_proj = nn.Sequential(
            Transpose(1, 2),
            nn.Conv1d(3, 8, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(8, 16, 3, 2, 1),
            Transpose(1, 2)
        )

        self.pos_enc = nn.Parameter(torch.linspace(-3, 3, 128).view(1, 128, 1))

        self.cross_attn_1 = NegLogEuclidianAttn(2, 16, 256, 256, 64, 4, dropout=0.0)
        self.cross_attn_2 = NegLogEuclidianAttn(64, 16, 256, 256, 64, 4, dropout=0.0)
        self.cross_attn_3 = NegLogEuclidianAttn(64, 16, 256, 256, 64, 4, dropout=0.0)


    def forward(self, trajs):
        """
        :param trajs: list of (N, 128, 2) tensors
        :return:
        """

        B = len(trajs)

        traj_nums = [len(traj) for traj in trajs]
        combined_trajs = torch.cat(trajs, dim=0)    # (sum_N, 128, 2)
        total_trajs = combined_trajs.shape[0]
        combined_trajs = torch.cat([combined_trajs, self.pos_enc.expand(total_trajs, -1, -1)], dim=2)
        combined_trajs = self.traj_proj(combined_trajs)
        proj_trajs = torch.split(combined_trajs, traj_nums)  # each: (N, 64, 32)
        traj_encodings = self.grid_enc.view(1, 1, self.grid_size**2, 2).repeat(B, 1, 1, 1)  # (B, 1, gs^2, 2)

        result_list = []

        for b in range(B):
            N = trajs[b].shape[0]
            scaling = N ** 0.5
            traj_enc = traj_encodings[b].repeat(N, 1, 1)  # (N, gs^2, 64)
            traj_enc = self.cross_attn_1(traj_enc, proj_trajs[b])
            traj_enc = (torch.sum(traj_enc, dim=0, keepdim=True) / scaling).expand(N, -1, -1)     # (N, gs^2, 64)
            traj_enc = self.cross_attn_2(traj_enc, proj_trajs[b])
            traj_enc = (torch.sum(traj_enc, dim=0, keepdim=True) / scaling).expand(N, -1, -1)     # (1, gs^2, 64)
            traj_enc = self.cross_attn_3(traj_enc, proj_trajs[b])
            traj_enc = torch.sum(traj_enc, dim=0, keepdim=True) / scaling  # (1, gs^2, 64)
            result_list.append(traj_enc)

        return torch.stack(result_list, dim=0)  # (B, gs^2, 2)


    def oldforward(self, trajs):
        """
        :param trajs: list of (N, 256, 2) tensors
        :return:
        """

        B = len(trajs)

        traj_encodings = self.grid_enc.view(1, 1, self.grid_size ** 2, 64).repeat(B, 1, 1, 1)  # (B, 1, gs^2, 64)
        result_list = []

        for b in range(B):
            N = trajs[b].shape[0]
            proj_trajs = self.traj_proj(trajs[b])   # (N, 128, 2) -> (N, 64, 32)
            # tmp_list = [traj_encodings[b]]
            traj_enc = traj_encodings[b]
            for n in range(N):
                traj_enc = self.cross_attn_1(traj_enc, proj_trajs[n].unsqueeze(0))
                traj_enc = self.cross_attn_2(traj_enc, proj_trajs[n].unsqueeze(0))
            result_list.append(traj_enc.unsqueeze(0))
            # result_list[b]: (gs^2, 64)

        return torch.stack(result_list, dim=0)  # (B, gs^2, 64)


class TrajDecoder(nn.Module):
    def __init__(self, grid_size: int = 16):
        super().__init__()

        self.grid_size = grid_size

        # input: (B, 64, gs, gs)
        self.convs = nn.Sequential(
            nn.UpsamplingBilinear2d(32),
            Res(64, 64),
            Res(64, 64),
            nn.UpsamplingBilinear2d(64),
            Res(64, 64),
            Res(64, 32),
            Res(32, 16),
            Res(16, 8),
            nn.Conv2d(8, 2, 3, 1, 1),
        )

        nn.init.zeros_(self.convs[-1].weight)
        nn.init.zeros_(self.convs[-1].bias)

    def forward(self, traj_enc):
        # traj_enc: (B, gs^2, 64)
        B = traj_enc.shape[0]
        traj_enc = traj_enc.view(B, self.grid_size, self.grid_size, -1).permute(0, 3, 1, 2)     # (B, 64, gs, gs)
        return self.convs(traj_enc)







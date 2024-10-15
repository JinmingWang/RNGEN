import torch
from Models import Encoder

TAE = Encoder(N_trajs=64, L_traj=128, D_encode=32)

torch.save(TAE.state_dict(), "encoder.pth")

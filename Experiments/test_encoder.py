import torch
from Models import TrajEncoder
from torchsummary import summary

TAE = TrajEncoder(N_trajs=32, L_traj=64, D_encode=64)

summary(TAE, torch.randn(64, 32, 64, 2))

torch.save(TAE.state_dict(), "encoder.pth")

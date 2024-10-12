import torch
from Models import Encoder

TAE = TrajAutoEncoder()

torch.save(TAE.encoder.state_dict(), "encoder.pth")
torch.save(TAE.decoder.state_dict(), "decoder.pth")

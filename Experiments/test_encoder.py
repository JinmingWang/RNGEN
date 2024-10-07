import torch
from Models import TrajAutoEncoder

TAE = TrajAutoEncoder()

torch.save(TAE.encoder.state_dict(), "encoder.pth")
torch.save(TAE.decoder.state_dict(), "decoder.pth")

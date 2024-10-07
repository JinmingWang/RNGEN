import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
Tensor = torch.Tensor

import numpy as np
from tqdm import tqdm
from datetime import datetime
import os
from typing import List

from Dataset import DEVICE, LaDeCachedDataset
from Models import DiffusionNetwork, Encoder
from Diffusion import DDPM
from Train.Utils import loadModels, saveModels, MovingAvg, PlotManager

# Dataset & Model Parameters
DATA_DIR = "./Dataset/Shanghai_5k"
N_TRAJS = 32
N_NODES = 128
N_SEGS = 64
TRAJ_LEN = 64
DIM_TRAJ_ENC = 128
BETA_MIN = 0.0001
BETA_MAX = 0.05
T = 500

# Training Parameters
LR_ENCODER = 1e-5
LR_DIFFUSION = 1e-4
LR_REDUCE_FACTOR = 0.5
LR_REDUCE_PATIENCE = 10
LR_REDUCE_MIN = 1e-6
EPOCHS = 1000
B = 100
LOG_DIR = f"./Runs/NodeEdgeModel_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/"

# Logging Parameters
MOV_AVG_LEN = 20
LOG_INTERVAL = 2
PLOT_INTERVAL = 50
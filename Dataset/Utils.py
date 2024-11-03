import torch
from typing import List, Tuple, Set, FrozenSet, Literal, Dict
from jaxtyping import Float, Bool
from einops import rearrange, reduce
import random

Tensor = torch.Tensor

Node = Float[Tensor, "B 2"]
Trajectory = Float[Tensor, "B T 2"]

# LaDe Dataset: https://arxiv.org/abs/2306.10675
DATASET_ROOT = "/home/jimmy/Data/LaDe"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



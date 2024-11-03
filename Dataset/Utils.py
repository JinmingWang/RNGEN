import torch
from typing import List, Tuple, Set, FrozenSet, Literal, Dict
from jaxtyping import Float, Bool
from einops import rearrange, einsum, reduce
import random

Tensor = torch.Tensor

Node = Tensor  # (2,)
Trajectory = Tensor  # (T, 2)

# LaDe Dataset: https://arxiv.org/abs/2306.10675
DATASET_ROOT = "/home/jimmy/Data/LaDe"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



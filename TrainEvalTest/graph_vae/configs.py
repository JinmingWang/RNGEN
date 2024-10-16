
from datetime import datetime

COMMENT = "GVAE"

# Training Parameters
D_IN = 5
D_LATENT = 16
D_HEAD = 64
D_EXPAND = 512
D_HIDDEN = 256
N_HEADS = 12
N_LAYERS = 6

KL_WEIGHT = 2e-4

LR = 2e-4
LR_REDUCE_FACTOR = 0.5
LR_REDUCE_PATIENCE = 30
LR_REDUCE_MIN = 1e-7
LR_REDUCE_THRESHOLD = 1e-5
EPOCHS = 2000
B = 64
LOG_DIR = f"./Runs/GraphVAE/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{COMMENT}/"

# Logging Parameters
MOV_AVG_LEN = 4*160
LOG_INTERVAL = 20
EVAL_INTERVAL = 375

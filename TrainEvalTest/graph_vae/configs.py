
from datetime import datetime

COMMENT = "SegsJoints"

# Training Parameters
D_LATENT = 16
D_HEAD = 64
D_EXPAND = 512
D_HIDDEN = 128
N_HEADS = 16

KL_WEIGHT = 1e-3

LR = 1e-4
LR_REDUCE_FACTOR = 0.5
LR_REDUCE_PATIENCE = 10
LR_REDUCE_MIN = 1e-7
LR_REDUCE_THRESHOLD = 5e-5
EPOCHS = 700
B = 128
LOG_DIR = f"./Runs/GraphVAE/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{COMMENT}/"

# Logging Parameters
MOV_AVG_LEN = 1000
LOG_INTERVAL = 20

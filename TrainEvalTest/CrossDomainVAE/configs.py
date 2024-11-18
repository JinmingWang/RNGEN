
from datetime import datetime

COMMENT = "segs_130MB"

# Training Parameters
KL_WEIGHT = 1e-6
LR = 1e-4
LR_REDUCE_FACTOR = 0.5
LR_REDUCE_PATIENCE = 30
LR_REDUCE_MIN = 1e-7
LR_REDUCE_THRESHOLD = 1e-5
EPOCHS = 500
B = 3
LOG_DIR = f"./Runs/CDVAE/{datetime.now().strftime('%Y%m%d_%H%M')[2:]}_{COMMENT}/"

# Logging Parameters
MOV_AVG_LEN = 5
LOG_INTERVAL = 10

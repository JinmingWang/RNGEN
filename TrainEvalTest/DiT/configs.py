
from datetime import datetime

COMMENT = "Sparse"

# Training Parameters
LR = 2e-4
LR_REDUCE_FACTOR = 0.5
LR_REDUCE_PATIENCE = 30
LR_REDUCE_MIN = 1e-7
LR_REDUCE_THRESHOLD = 1e-5
EPOCHS = 1000
B = 32
LOG_DIR = f"./Runs/PathsDiT/{datetime.now().strftime('%Y%m%d_%H%M')[2:]}_{COMMENT}/"

# Logging Parameters
MOV_AVG_LEN = 6     # Epochs
LOG_INTERVAL = 10   # steps
EVAL_INTERVAL = 5   # Epochs

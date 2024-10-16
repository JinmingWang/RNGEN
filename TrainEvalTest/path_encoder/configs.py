
from datetime import datetime

COMMENT = "3_stages"

# Training Parameters
LR_ENCODER = 2e-4
LR_REDUCE_FACTOR = 0.5
LR_REDUCE_PATIENCE = 30
LR_REDUCE_MIN = 1e-7
LR_REDUCE_THRESHOLD = 1e-5
EPOCHS = 5000
B = 64
LOG_DIR = f"./Runs/PathEncoder/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{COMMENT}/"

# Logging Parameters
MOV_AVG_LEN = 4*100
LOG_INTERVAL = 20
EVAL_INTERVAL = 375

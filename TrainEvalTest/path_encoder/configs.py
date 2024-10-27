
from datetime import datetime

COMMENT = "ResAttn"

GRAPH_VAE_WEIGHT = "Runs/GraphVAE/20241025_142500_SegsJoints/last.pth"

# Training Parameters
LR = 2e-4
LR_REDUCE_FACTOR = 0.5
LR_REDUCE_PATIENCE = 30
LR_REDUCE_MIN = 1e-7
LR_REDUCE_THRESHOLD = 1e-5
EPOCHS = 1000
B = 50
LOG_DIR = f"./Runs/PathEncoder/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{COMMENT}/"

# Logging Parameters
MOV_AVG_LEN = 5
LOG_INTERVAL = 20
EVAL_INTERVAL = 5

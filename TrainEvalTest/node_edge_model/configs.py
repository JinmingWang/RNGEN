
from datetime import datetime


# Training Parameters
LR_ENCODER = 1e-4
LR_DIFFUSION = 2e-4
LR_REDUCE_FACTOR = 0.5
LR_REDUCE_PATIENCE = 30
LR_REDUCE_MIN = 1e-7
LR_REDUCE_THRESHOLD = 1e-6
EPOCHS = 600
B = 200
LOG_DIR = f"./Runs/NodeEdgeModel/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/"

# Logging Parameters
MOV_AVG_LEN = 4*160
LOG_INTERVAL = 20
EVAL_INTERVAL = 160

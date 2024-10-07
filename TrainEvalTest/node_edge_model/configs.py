
from datetime import datetime


# Training Parameters
LR_ENCODER = 1e-4
LR_DIFFUSION = 1e-4
LR_REDUCE_FACTOR = 0.5
LR_REDUCE_PATIENCE = 20
LR_REDUCE_MIN = 1e-7
EPOCHS = 1000
B = 100
LOG_DIR = f"./Runs/NodeEdgeModel_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/"

# Logging Parameters
MOV_AVG_LEN = 4*160
LOG_INTERVAL = 10
EVAL_INTERVAL = 160
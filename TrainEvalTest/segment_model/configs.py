
from datetime import datetime

COMMENT = "Very_Simple"

PATH_ENCODER_WEIGHT = ""
# GRAPH_VAE_WEIGHT = "Runs/GraphVAE/20241017_043956_SegsJoints/last.pth"
GRAPH_VAE_WEIGHT = "Runs/GraphVAE/20241025_142500_SegsJoints/last.pth"

RELEASE_PATH_ENC = 0

# Training Parameters
LR = 2e-4
LR_REDUCE_FACTOR = 0.5
LR_REDUCE_PATIENCE = 30
LR_REDUCE_MIN = 1e-7
LR_REDUCE_THRESHOLD = 1e-5
EPOCHS = 500
B = 64
LOG_DIR = f"./Runs/SegmentsModel/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{COMMENT}/"

# Logging Parameters
MOV_AVG_LEN = 5     # Epochs
LOG_INTERVAL = 10   # steps
EVAL_INTERVAL = 5   # Epochs

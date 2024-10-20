
from datetime import datetime

COMMENT = "PathEnc"

PATH_ENCODER_WEIGHT = ""
GRAPH_VAE_WEIGHT = "Runs/GraphVAE/20241017_043956_SegsJoints/last.pth"

RELEASE_PATH_ENC = 0

# Training Parameters
LR = 2e-4
LR_REDUCE_FACTOR = 0.5
LR_REDUCE_PATIENCE = 50
LR_REDUCE_MIN = 1e-7
LR_REDUCE_THRESHOLD = 1e-5
EPOCHS = 1500
B = 64
LOG_DIR = f"./Runs/SegmentsModel/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{COMMENT}/"

# Logging Parameters
MOV_AVG_LEN = 1000     # steps
LOG_INTERVAL = 10   # steps
EVAL_INTERVAL = 3   # Epochs

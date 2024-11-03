from TrainEvalTest.node_edge_model.train import train as train_node_edge
from TrainEvalTest.segment_model.train import train as train_segments
from TrainEvalTest.segment_model.train_pred_x0 import train as train_segments_x0
import time

if __name__ == "__main__":
    # Scheduled task
    print("Waiting for start")
    time.sleep(3600 * 3)
    train_segments_x0()

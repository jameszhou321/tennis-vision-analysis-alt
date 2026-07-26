"""config_legacy.py — Batch Processing Pipeline Configuration (for use by main.py)

Function: Defines video input paths, output paths, model paths, and various processing parameters
"""
import os

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SRC_DIR)

# Path Configurations
VIDEO_PATH = os.path.join(_SRC_DIR, "videos")
OUTPUT_DIR = os.path.join(_SRC_DIR, "data", "rallies_new")
MODEL_PATH = os.path.join(_SRC_DIR, "models", "yolo", "yolo11x-pose.pt")

# Court keypoint model (14 pts) used by court_detector.py to fit the pixel<->real-world
# homography that pose_tracker.py uses for far/near player selection. Same convention/weights
# layout as src/pipeline/offline_tennis_tracker.py.
COURT_MODEL_PATH = os.path.join(_PROJECT_DIR, "runs", "court_finetune", "court_14pts_ultimate",
                                 "weights", "best.pt")

# Video Processing and Queue Parameters
SCOUT_SKIP_FRAMES = 5
SCOUT_SCALE = 0.5
MIN_RALLY_DURATION = 4.0  # Minimum rally duration (seconds)

# Pose Tracking Parameters
YOLO_IMGSZ = 1024
POSE_TRACK_CONF = 0.25    # min confidence for the full-frame person/pose tracker (pose_tracker.py)
POSE_MAX_GAP = 5          # Maximum frame drop compensation limit (frames)
POSE_ALPHA = 0.6          # EMA smoothing coefficient

# Court Homography Parameters (court_detector.py)
COURT_KPT_CONF = 0.4             # min confidence to trust a single court keypoint for homography fitting
HOMOGRAPHY_HISTORY = 5           # frames averaged by HomographyFilter for temporal smoothing
BASELINE_SEARCH_RADIUS_M = 10.0  # baseline-proximity scoring radius (meters) for player selection

# Control Files and Checkpoints
CONTROL_FILE = os.path.join(_SRC_DIR, "logs", "control.txt")
CHECKPOINT_FILE = os.path.join(_SRC_DIR, "logs", "checkpoint.json")
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

# Video Processing and Queue Parameters
SCOUT_SKIP_FRAMES = 5
SCOUT_SCALE = 0.5
MIN_RALLY_DURATION = 4.0  # Minimum rally duration (seconds)

# Pose Tracking Parameters
YOLO_IMGSZ = 1024
CONF_FAR = 0.15
CONF_NEAR = 0.3
POSE_MAX_GAP = 5          # Maximum frame drop compensation limit (frames)
POSE_ALPHA = 0.6          # EMA smoothing coefficient

# Control Files and Checkpoints
CONTROL_FILE = os.path.join(_SRC_DIR, "logs", "control.txt")
CHECKPOINT_FILE = os.path.join(_SRC_DIR, "logs", "checkpoint.json")
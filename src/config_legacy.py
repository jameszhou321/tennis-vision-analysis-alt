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

# Far-half supplemental crop pass (pose_tracker.py), restores the old per-side sensitivity/
# resolution advantage the far player lost when the single-slot ROI approach was replaced.
POSE_TRACK_CONF_FAR = 0.15       # min confidence for the far-half crop pass (restores old CONF_FAR)
FAR_CROP_MARGIN_FRAC = 0.08      # padding (fraction of frame height) around the projected far-half bbox
FAR_CROP_MAX_MATCH_DIST_M = 2.0  # gating distance (meters): skip a crop detection that duplicates a
                                  # full-frame track already seen this frame; also used to associate a
                                  # crop detection frame-to-frame into a pseudo-track
FAR_CROP_TRACK_MAX_GAP = 8       # frames a far-crop pseudo-track can go unmatched before it ends

# Cross-source track stitching (pose_tracker.py's select_players): both the full-frame BoT-SORT
# pass and the far-crop pass independently re-issue a new track ID whenever they briefly lose and
# reacquire the same physical player (marginal confidence flicker, brief occlusion), fragmenting
# what should be one long track across many short-lived IDs. Before scoring, tracks on the same
# real-world side are greedily chained together if the gap since the earlier one's last frame is
# within TRACK_STITCH_MAX_GAP_FRAMES and the position at that point is within
# TRACK_STITCH_MAX_DIST_M of where the later one starts.
TRACK_STITCH_MAX_GAP_FRAMES = 25  # ~1s at 30fps
TRACK_STITCH_MAX_DIST_M = 5.0     # generous vs. realistic sprint distance in ~1s

# Court Homography Parameters (court_detector.py)
COURT_KPT_CONF = 0.4             # min confidence to trust a single court keypoint for homography fitting
HOMOGRAPHY_HISTORY = 5           # frames averaged by HomographyFilter for temporal smoothing
BASELINE_SEARCH_RADIUS_M = 10.0  # baseline-proximity scoring radius (meters) for player selection
MAX_HOMOGRAPHY_STALE_FRAMES = 15 # once the court hasn't been confidently re-detected for this many
                                  # consecutive frames, treat H as unavailable rather than trusting an
                                  # indefinitely-stale fit -- protects against camera cutaways (crowd
                                  # shots, player closeups, replays) being tracked against a homography
                                  # fit for a completely different framing

# Camera-Cut / Off-Court Noise Rejection (main.py's process_fusion_clip and annotate_rally_clip)
SCENE_CUT_THRESHOLD = 30.0    # PySceneDetect ContentDetector threshold; same default process_broadcast_clip
                                # already uses. Crossing a detected cut resets per-frame tracker state (ball
                                # tracker, motion detector, court homography) that would otherwise misread
                                # the discontinuity as plausible motion or a continued ball position.
COURT_REGION_MARGIN_M = 3.0   # a person/ball detection whose real-world position (via the court
                                # homography) falls outside the court's physical extent plus this margin
                                # (meters) is treated as off-court noise (spectators, stands, a TrackNet
                                # false positive in the crowd) rather than counted toward fusion mode's
                                # motion/ball signals

# Control Files and Checkpoints
CONTROL_FILE = os.path.join(_SRC_DIR, "logs", "control.txt")
CHECKPOINT_FILE = os.path.join(_SRC_DIR, "logs", "checkpoint.json")
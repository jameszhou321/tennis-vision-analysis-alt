"""
data_batch_extractor.py — Batch Rally Data Extraction Pipeline

Function: Iterates through data/rallies_new/, runs court detection + pose tracking 
          on each rally video clip, and outputs JSON annotations.
"""
import os
import cv2
import json
import numpy as np
from pathlib import Path
from ultralytics import YOLO

_UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(os.path.dirname(_UTILS_DIR))

# =====================================================================
# 1. Configuration Section
# =====================================================================
DATASET_ROOT = os.path.join(_PROJECT_DIR, "data", "rallies_new")

COURT_MODEL_PATH = os.path.join(_PROJECT_DIR, "runs", "court_finetune", "court_14pts_ultimate", "weights", "best.pt")
TRACKER_MODEL_PATH = os.path.join(_PROJECT_DIR, "models", "court", "best.pt")
POSE_MODEL_PATH = os.path.join(_PROJECT_DIR, "models", "yolo", "yolo11x-pose.pt")

# System log files
PROGRESS_LOG = os.path.join(_PROJECT_DIR, "logs", "pipeline_progress.txt")
SUBPAR_LOG = os.path.join(_PROJECT_DIR, "logs", "pipeline_subpar.txt")
STATS_FILE = os.path.join(_PROJECT_DIR, "logs", "pipeline_stats.json")
ERROR_LOG = "data-tractor/pipeline_errors.txt"  # Tracks unopenable or corrupted videos


# =====================================================================
# 2. State Manager (Breakpoint Resumption & Running Means)
# =====================================================================
class PipelineManager:
    def __init__(self):
        self.processed = set()
        self.global_stats = {"court_conf": 0.0, "player_conf": 0.0, "pose_conf": 0.0, "count": 0}
        self.load_state()

    def load_state(self):
        """Loads breakpoint resumption progress and historical statistics."""
        if os.path.exists(PROGRESS_LOG):
            with open(PROGRESS_LOG, "r", encoding="utf-8") as f:
                self.processed = set(line.strip() for line in f if line.strip())

        if os.path.exists(STATS_FILE):
            try:
                with open(STATS_FILE, "r", encoding="utf-8") as f:
                    self.global_stats = json.load(f)
            except Exception:
                pass

        print(f"Progress loaded: {len(self.processed)} clips completed. "
              f"Global accumulation counter at {self.global_stats['count']}.")

    def mark_processed(self, clip_path):
        """Records a video clip path as successfully processed."""
        self.processed.add(str(clip_path))
        with open(PROGRESS_LOG, "a", encoding="utf-8") as f:
            f.write(f"{clip_path}\n")

    def log_error(self, clip_path, reason):
        """Logs a corrupted or unreadable video, marking it processed to prevent endless retry loops."""
        with open(ERROR_LOG, "a", encoding="utf-8") as f:
            f.write(f"[{clip_path}] - {reason}\n")
        self.mark_processed(clip_path)

    def update_and_check_stats(self, clip_path, clip_stats):
        """Updates global confidence moving targets and checks if the current video performance flags as subpar."""
        if clip_stats["count"] == 0:
            return

        # Calculate local mean confidence scores for the current clip
        cur_court = float(clip_stats["court_conf"] / clip_stats["count"])
        cur_player = float(clip_stats["player_conf"] / clip_stats["count"])
        cur_pose = float(clip_stats["pose_conf"] / clip_stats["count"])

        # Retrieve global running metrics
        g_count = max(1, self.global_stats["count"])
        g_court = float(self.global_stats["court_conf"]) / g_count
        g_player = float(self.global_stats["player_conf"]) / g_count
        g_pose = float(self.global_stats["pose_conf"]) / g_count

        # Evaluate potential anomalies once a stable historical baseline sample is established
        if self.global_stats["count"] > 5:
            reasons = []
            if cur_court < g_court: reasons.append(f"Court Conf low ({cur_court:.2f}<{g_court:.2f})")
            if cur_player < g_player: reasons.append(f"Player Conf low ({cur_player:.2f}<{g_player:.2f})")
            if cur_pose < g_pose: reasons.append(f"Pose Conf low ({cur_pose:.2f}<{g_pose:.2f})")

            if reasons:
                with open(SUBPAR_LOG, "a", encoding="utf-8") as f:
                    f.write(f"{clip_path} -> {' | '.join(reasons)}\n")

        # Accumulate local metrics into the global statistics pool
        self.global_stats["court_conf"] += float(clip_stats["court_conf"])
        self.global_stats["player_conf"] += float(clip_stats["player_conf"])
        self.global_stats["pose_conf"] += float(clip_stats["pose_conf"])
        self.global_stats["count"] += int(clip_stats["count"])

        # Persist updated status
        with open(STATS_FILE, "w", encoding="utf-8") as f:
            json.dump(self.global_stats, f)


# =====================================================================
# 3. Main Extraction Pipeline
# =====================================================================
def main():
    manager = PipelineManager()

    # Discover target clips
    dataset_path = Path(DATASET_ROOT)
    all_clips = list(dataset_path.rglob("raw_clip.mp4"))

    pending_clips = [c for c in all_clips if str(c) not in manager.processed]
    print(f"Total video clips discovered: {len(all_clips)}, Remaining to process: {len(pending_clips)}\n")

    if not pending_clips:
        print("All video clips have been processed successfully!")
        return

    print("⏳ Loading YOLO weights into memory...")
    court_model = YOLO(COURT_MODEL_PATH)
    tracker_model = YOLO(TRACKER_MODEL_PATH)
    pose_model = YOLO(POSE_MODEL_PATH)

    for clip_idx, clip_path in enumerate(pending_clips):
        print(f"\n[{clip_idx + 1}/{len(pending_clips)}] Processing: {clip_path.parent.parent.name}/{clip_path.parent.name}")

        # Crash Protection Layer 1: File access and I/O exceptions
        try:
            cap = cv2.VideoCapture(str(clip_path))
        except Exception as e:
            manager.log_error(clip_path, f"File read exception: {e}")
            continue

        if not cap.isOpened():
            manager.log_error(clip_path, "Failed to initialize VideoCapture stream (corrupted format)")
            continue

        clip_json_data = {"clip_info": str(clip_path), "frames": []}
        clip_stats = {"court_conf": 0.0, "player_conf": 0.0, "pose_conf": 0.0, "count": 0}
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_data = {"frame_id": frame_idx, "court": None, "players": []}
            h_img, w_img = frame.shape[:2]

            # Crash Protection Layer 2: Core multi-model inference wrapper per frame
            try:
                # --- A. Court Keypoints Line Detection ---
                c_res = court_model.predict(frame, conf=0.3, verbose=False)[0]
                if c_res.keypoints is not None and len(c_res.keypoints.data) > 0:
                    kpts = c_res.keypoints.data[0].cpu().numpy().tolist()
                    frame_data["court"] = kpts  # Stores 14 structural court nodes [x, y, conf]
                    
                    confs = [p[2] for p in kpts if p[2] > 0]
                    if confs:
                        clip_stats["court_conf"] += sum(confs) / len(confs)

                # --- B. Multi-Player Tracking & Pose Estimation ---
                t_res = tracker_model.track(frame, persist=True, tracker="botsort.yaml", verbose=False)[0]
                if t_res.boxes is not None and t_res.boxes.id is not None:
                    ids = t_res.boxes.id.int().cpu().tolist()
                    bboxes = t_res.boxes.xyxy.cpu().numpy()
                    p_confs = t_res.boxes.conf.cpu().numpy()

                    for i, tid in enumerate(ids):
                        bx = bboxes[i].astype(int)
                        clip_stats["player_conf"] += p_confs[i]

                        player_info = {
                            "id": tid,
                            "bbox": bx.tolist(),
                            "bbox_conf": float(p_confs[i]),
                            "pose": None
                        }

                        # Apply a 50% spatial dilation expansion factor to the cropped frame bounding box
                        bw, bh = bx[2] - bx[0], bx[3] - bx[1]
                        pad_x = int(bw * 0.25) + 10
                        pad_y = int(bh * 0.25) + 10

                        cx1, cy1 = max(0, bx[0] - pad_x), max(0, bx[1] - pad_y)
                        cx2, cy2 = min(w_img, bx[2] + pad_x), min(h_img, bx[3] + pad_y)

                        crop = frame[cy1:cy2, cx1:cx2]
                        if crop.shape[0] >= 10 and crop.shape[1] >= 10:
                            p_res = pose_model.predict(crop, imgsz=192, verbose=False)[0]
                            if p_res.keypoints is not None and len(p_res.keypoints.data) > 0:
                                kpts = p_res.keypoints.data[0].cpu().numpy().copy()

                                # Map coordinate outputs back to the global resolution space
                                kpts[:, 0] += cx1
                                kpts[:, 1] += cy1

                                # Filter logic: Nullify stray background keypoints falling out of the explicit player box boundary
                                valid_mask = (kpts[:, 0] >= bx[0]) & (kpts[:, 0] <= bx[2]) & \
                                             (kpts[:, 1] >= bx[1]) & (kpts[:, 1] <= bx[3])

                                # Force structural coordinate and visibility confidence parameters to zero for invalid markers
                                kpts[~valid_mask] = [0.0, 0.0, 0.0]
                                player_info["pose"] = kpts.tolist()

                                valid_confs = [p[2] for p in kpts if p[2] > 0]
                                if valid_confs:
                                    clip_stats["pose_conf"] += sum(valid_confs) / len(valid_confs)

                        frame_data["players"].append(player_info)

                clip_stats["count"] += 1
                clip_json_data["frames"].append(frame_data)
                frame_idx += 1

            except Exception as e:
                # Isolates frame-level artifacts/deformities to preserve execution integrity for the remaining video sequence
                print(f"Exception encountered at frame index {frame_idx}, frame dropped. Error details: {e}")
                continue

        cap.release()

        # 4. Persistence Phase
        json_save_path = clip_path.parent / "tracking_data.json"
        try:
            with open(json_save_path, "w", encoding="utf-8") as f:
                json.dump(clip_json_data, f, ensure_ascii=False)
        except Exception as e:
            manager.log_error(clip_path, f"JSON serialization failed: {e}")
            continue

        # Append data stream changes into global records and checkpoint performance levels
        manager.update_and_check_stats(str(clip_path), clip_stats)

        # Update process tracking log
        manager.mark_processed(str(clip_path))
        print(f"   Successfully serialized and cached. Extracted features across {frame_idx} sequential frames.")


if __name__ == "__main__":
    main()
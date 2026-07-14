"""main.py — Batch Video Processing Main Entry Point

Functionality: Traverses the videos/ directory and processes each match.
Kinematics: Robust frame-to-frame bounding box center displacement matching
            to guarantee accurate rally detection without tracker dropouts.
"""
import cv2
import numpy as np
import json
import os
import time
import threading
import queue
import torch
import subprocess
import logging
from ultralytics import YOLO

import config_legacy as config

# Block all internal YOLO logging noise
logging.getLogger("ultralytics").setLevel(logging.ERROR)


class RobustKinematicDetector:
    def __init__(self, movement_threshold=12.0):
        self.movement_threshold = movement_threshold
        self.prev_centers = []

    def calculate_motion_score(self, current_boxes):
        """Calculates global player movement by tracking the displacement of box centers."""
        if len(current_boxes) == 0:
            self.prev_centers = []
            return 0.0

        # Calculate center points for all detected people
        current_centers = []
        for box in current_boxes:
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            current_centers.append((cx, cy))

        if not self.prev_centers:
            self.prev_centers = current_centers
            return 0.0

        total_distance = 0.0
        matched_count = 0

        # Greedy closest-point matching between frames (immune to tracker ID drops)
        for c_cx, c_cy in current_centers:
            distances = [np.sqrt((c_cx - p_cx)**2 + (c_cy - p_cy)**2) for p_cx, p_cy in self.prev_centers]
            if distances:
                total_distance += min(distances)
                matched_count += 1

        self.prev_centers = current_centers

        if matched_count > 0:
            return total_distance / matched_count
        return 0.0


class BatchTennisPipeline:
    def __init__(self):
        self.input_dir = config.VIDEO_PATH
        self.output_base_dir = config.OUTPUT_DIR

        self.video_files = sorted([f for f in os.listdir(self.input_dir) if f.lower().endswith('.mp4')])
        if not self.video_files:
            raise FileNotFoundError(f"No mp4 files found in {self.input_dir}!")

        torch.backends.cudnn.benchmark = True

        self.current_video_idx = 0
        self.current_scout_frame = 0
        self.current_task_count = 0
        self.pending_queue_data = []  

        self._load_checkpoint()

    def _load_checkpoint(self):
        if os.path.exists(config.CHECKPOINT_FILE):
            try:
                with open(config.CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.current_video_idx = data.get("video_idx", 0)
                self.current_scout_frame = data.get("scout_frame", 0)
                self.current_task_count = data.get("gpu_task_count", 0)
                self.pending_queue_data = data.get("pending_queue", [])
            except Exception:
                print("[!] Checkpoint corrupted, restarting from scratch.")

    def _save_checkpoint(self):
        pending = list(self.task_queue.queue)
        state = {
            "video_idx": self.current_video_idx,
            "scout_frame": self.current_scout_frame,
            "gpu_task_count": self.current_task_count,
            "pending_queue": pending
        }
        with open(config.CHECKPOINT_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=4)

    def export_highlight_ffmpeg(self, video_path, start_sec, duration, output_path):
        command = [
            "ffmpeg", "-y",
            "-ss", f"{start_sec:.3f}",
            "-i", video_path,
            "-t", f"{duration:.3f}",
            "-c", "copy",
            output_path
        ]
        try:
            subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            print(f"  💾 [FFmpeg] Exported native audio highlight: {os.path.basename(output_path)}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"  ⚠️ [FFmpeg] Lossless copy failed. Check your system's ffmpeg installation.")

    def producer_scout_thread(self, video_path, total_frames, fps):
        print(f"[CPU] Scouting Kinematics -> {os.path.basename(video_path)}")
        
        scout_model = YOLO("yolov8n.pt")
        device = 'cuda:0' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
        scout_model.to(device)
        
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_scout_frame)
        
        detector = RobustKinematicDetector(movement_threshold=12.0)
        sample_interval = int(fps / 2) or 15
        frame_log = []
        frame_idx = self.current_scout_frame

        while cap.isOpened():
            if self.stop_event.is_set():
                break

            ret, frame = cap.read()
            if not ret: 
                break

            if frame_idx % sample_interval == 0:
                results = scout_model(frame, classes=[0], verbose=False)
                
                avg_velocity = 0.0
                if len(results) > 0 and len(results[0].boxes) > 0:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    avg_velocity = detector.calculate_motion_score(boxes)

                status = "PLAYING" if avg_velocity > 12.0 else "NON-PLAYING"
                frame_log.append({"frame_idx": frame_idx, "status": status})

            frame_idx += 1
        
        cap.release()

        if frame_log:
            current_block = {
                "status": frame_log[0]["status"], 
                "start": frame_log[0]["frame_idx"], 
                "end": frame_log[0]["frame_idx"]
            }
            timeline_blocks = []
            
            for entry in frame_log[1:]:
                if entry["status"] == current_block["status"]:
                    current_block["end"] = entry["frame_idx"]
                else:
                    timeline_blocks.append(current_block)
                    current_block = {
                        "status": entry["status"], 
                        "start": entry["frame_idx"], 
                        "end": entry["frame_idx"]
                    }
            timeline_blocks.append(current_block)

            for block in timeline_blocks:
                if block["status"] == "PLAYING":
                    duration = (block["end"] - block["start"]) / fps
                    if duration >= config.MIN_RALLY_DURATION:
                        task = {
                            'start': block["start"],
                            'end': block["end"],
                            'duration': duration
                        }
                        self.task_queue.put(task)
                        print(f"[CPU] Rally Detected & Enqueued: {duration:.1f}s clip")

        self.current_scout_frame = frame_idx
        time.sleep(1.0)
        self.scout_finished.set()

    def consumer_yolo_thread(self, video_path, video_output_dir, fps, width, height):
        print("[GPU] Processing Thread Engaged")
        model = YOLO(config.MODEL_PATH)
        device = 'cuda:0' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
        model.to(device)
        
        detector = RobustKinematicDetector(movement_threshold=12.0)
        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        task_count = self.current_task_count

        while True:
            try:
                rally = self.task_queue.get(timeout=2.0)
            except queue.Empty:
                if self.scout_finished.is_set(): 
                    break
                if self.stop_event.is_set():
                    break
                continue

            task_count += 1
            duration = rally['duration']
            
            clean_duration_str = f"{int(duration)}s"
            clip_name = f"rally_{task_count:03d}_{clean_duration_str}"
            clip_dir = os.path.join(video_output_dir, clip_name)
            os.makedirs(clip_dir, exist_ok=True)

            ffmpeg_highlight_path = os.path.join(clip_dir, "native_audio_highlight.mp4")
            ann_path = os.path.join(clip_dir, "annotated_clip.mp4")
            json_path = os.path.join(clip_dir, "pose_data.json")

            out_ann = cv2.VideoWriter(ann_path, fourcc, fps, (width, height))
            cap.set(cv2.CAP_PROP_POS_FRAMES, rally['start'])
            curr_frame = rally['start']
            json_data = []

            print(f"[GPU] Generating high-precision renders for: {clip_name}")

            while curr_frame <= rally['end']:
                ret, frame = cap.read()
                if not ret: break

                ann_frame = frame.copy()
                results = model(frame, classes=[0], verbose=False)
                
                avg_velocity = 0.0
                boxes_list = []
                if len(results) > 0 and len(results[0].boxes) > 0:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    avg_velocity = detector.calculate_motion_score(boxes)
                    
                    # Convert float32 coordinate primitives to regular native python floats
                    boxes_list = [[float(coord) for coord in box] for box in boxes]

                    for b in boxes:
                        cv2.rectangle(ann_frame, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 0, 0), 2)

                in_play = avg_velocity > 12.0
                status_text = "STATE: PLAYING" if in_play else "STATE: NOT PLAYING"
                status_color = (0, 255, 0) if in_play else (0, 0, 255)
                
                cv2.rectangle(ann_frame, (30, 30), (450, 100), (15, 15, 15), -1)
                cv2.putText(ann_frame, status_text, (50, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 3, cv2.LINE_AA)

                out_ann.write(ann_frame)
                json_data.append({
                    "frame": int(curr_frame),
                    "motion_score": float(avg_velocity),
                    "in_play": bool(in_play),
                    "detected_boxes": boxes_list
                })
                curr_frame += 1

            out_ann.release()
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=4)

            start_seconds = rally['start'] / fps
            self.export_highlight_ffmpeg(video_path, start_seconds, duration, ffmpeg_highlight_path)

            self.current_task_count = task_count
            self.task_queue.task_done()

        cap.release()

    def process_single_video(self, video_path):
        temp_cap = cv2.VideoCapture(video_path)
        fps = temp_cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(temp_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(temp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(temp_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        temp_cap.release()

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_output_dir = os.path.join(self.output_base_dir, video_name)
        os.makedirs(video_output_dir, exist_ok=True)

        self.task_queue = queue.Queue()
        for item in self.pending_queue_data:
            self.task_queue.put(item)
        self.pending_queue_data = []  

        self.scout_finished = threading.Event()
        self.stop_event = threading.Event()

        scout_t = threading.Thread(target=self.producer_scout_thread, args=(video_path, total_frames, fps))
        yolo_t = threading.Thread(target=self.consumer_yolo_thread, args=(video_path, video_output_dir, fps, width, height))

        scout_t.start()
        yolo_t.start()

        while scout_t.is_alive() or yolo_t.is_alive():
            if os.path.exists(config.CONTROL_FILE):
                try:
                    with open(config.CONTROL_FILE, "r", encoding="utf-8") as f:
                        cmd = f.read().strip().lower()
                    if cmd == "save":
                        self.stop_event.set()
                        break
                except Exception: pass
            time.sleep(2)

        scout_t.join()
        yolo_t.join()
        return self.stop_event.is_set()

    def run(self):
        for idx in range(self.current_video_idx, len(self.video_files)):
            self.current_video_idx = idx
            video_file = self.video_files[idx]
            video_path = os.path.join(self.input_dir, video_file)

            print(f"\n======== Processing Match ({idx + 1}/{len(self.video_files)}): {video_file} ========")
            is_stopped = self.process_single_video(video_path)

            if is_stopped:
                self._save_checkpoint()
                return
            else:
                self.current_scout_frame = 0
                self.current_task_count = 0


if __name__ == '__main__':
    if os.path.exists("checkpoint.json"):
        try: os.remove("checkpoint.json")
        except: pass
        
    pipeline = BatchTennisPipeline()
    pipeline.run()
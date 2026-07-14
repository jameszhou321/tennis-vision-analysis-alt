"""main.py — Batch Video Processing Main Entry Point

Function: Traverses the videos/ directory, runs the full tracking pipeline for each match video, and supports checkpoints for resuming.
Upgrade: Integrated a deadlock-proof, multi-format compatible Spatial Rally Detector.
"""
import cv2
import numpy as np
import json
import os
import time
import threading
import queue
import torch
from ultralytics import YOLO

import config_legacy as config
from court_detector import CourtDetector
from pose_tracker import PoseTracker


class SpatialRallyDetector:
    def __init__(self, fps=30, buffer_seconds=1.5, movement_threshold=3.0):
        """
        Accurately detects whether each frame is in the hitting rally phase (PLAYING) or dead ball phase (NOT PLAYING) 
        based on the spatial distribution and movement status of the players.
        """
        self.fps = fps
        self.buffer_frames = int(fps * buffer_seconds)
        self.movement_threshold = movement_threshold
        
        self.player_history = []  # Records player center point locations from previous frames
        self.frames_since_active = self.buffer_frames
        self.is_playing = False

    def _extract_box(self, player_data):
        """
        Enhanced defensive unpacking: Ensures a stable 4-element bounding box extraction regardless of whether 
        PoseTracker returns a dictionary, an object, or a nested list.
        """
        if not player_data:
            return None
        try:
            # Case 1: Standard dictionary format
            if isinstance(player_data, dict):
                box = player_data.get("box")
                if box is not None:
                    if hasattr(box, "tolist"): box = box.tolist()  # Compatible with numpy arrays
                    if isinstance(box, list) and len(box) >= 4:
                        return box[:4]
            # Case 2: Object with a .box attribute
            elif hasattr(player_data, "box"):
                box = player_data.box
                if hasattr(box, "tolist"): box = box.tolist()
                if hasattr(box, "__iter__") and len(box) >= 4:
                    return list(box)[:4]
        except Exception:
            pass
        return None

    def update(self, far_player_data, near_player_data):
        """
        Core logic: Updates the status according to the positions and movement ranges of the players on both sides in the current frame.
        """
        is_frame_active = False

        far_box = self._extract_box(far_player_data)
        near_box = self._extract_box(near_player_data)

        # 1. Spatial validation: Both far and near players must be identified on the court simultaneously
        if far_box is not None and near_box is not None:
            try:
                # Calculate the center points of the two players in global coordinates
                f_cx = (far_box[0] + far_box[2]) / 2
                f_cy = (far_box[1] + far_box[3]) / 2
                n_cx = (near_box[0] + near_box[2]) / 2
                n_cy = (near_box[1] + near_box[3]) / 2

                current_centers = np.array([[f_cx, f_cy], [n_cx, n_cy]])
                self.player_history.append(current_centers)
                
                if len(self.player_history) > 4:
                    self.player_history.pop(0)

                # 2. Movement validation: Calculate the player movement velocity between consecutive frames
                if len(self.player_history) >= 2:
                    # Calculate the displacements of the far and near players between the last two frames
                    disp_far = np.linalg.norm(self.player_history[-1][0] - self.player_history[-2][0])
                    disp_near = np.linalg.norm(self.player_history[-1][1] - self.player_history[-2][1])
                    max_displacement = max(disp_far, disp_near)

                    # If the players are doing high-frequency changes of direction or running across a wide area, the frame is active
                    if max_displacement > self.movement_threshold:
                        is_frame_active = True
                        
                # 3. Longitudinal relative position validation (prevents misjudging dead ball states when side-by-side or too close; threshold reduced to 50 for flexibility)
                if abs(f_cy - n_cy) < 50:  
                    is_frame_active = False
            except Exception:
                is_frame_active = False
        else:
            # Player data lost (e.g., walking out of frame), default to inactive
            is_frame_active = False

        # 4. Temporal smoothing buffer controls status transition
        if is_frame_active:
            self.is_playing = True
            self.frames_since_active = 0
        else:
            self.frames_since_active += 1

        if self.frames_since_active >= self.buffer_frames:
            self.is_playing = False
            self.player_history.clear()

        return self.is_playing


class BatchTennisPipeline:
    def __init__(self):
        self.input_dir = config.VIDEO_PATH
        self.output_base_dir = config.OUTPUT_DIR

        # Get all mp4 files in the directory and sort them to ensure consistent processing order
        self.video_files = sorted([f for f in os.listdir(self.input_dir) if f.lower().endswith('.mp4')])
        if not self.video_files:
            raise FileNotFoundError(f"No mp4 files found in {self.input_dir}!")

        self.court_detector = CourtDetector(scale=config.SCOUT_SCALE)

        torch.backends.cudnn.benchmark = True

        # Global progress status
        self.current_video_idx = 0
        self.current_scout_frame = 0
        self.current_task_count = 0
        self.pending_queue_data = []  # Temporarily stores queue data during checkpointing

        self._load_checkpoint()

    def _load_checkpoint(self):
        """ Loads local checkpoint to restore to a specific video and specific frame position """
        if os.path.exists(config.CHECKPOINT_FILE):
            try:
                with open(config.CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.current_video_idx = data.get("video_idx", 0)
                self.current_scout_frame = data.get("scout_frame", 0)
                self.current_task_count = data.get("gpu_task_count", 0)
                self.pending_queue_data = data.get("pending_queue", [])

                if self.current_video_idx >= len(self.video_files):
                    print("[*] The videos shown in the checkpoint have all been processed, starting from scratch.")
                    self.current_video_idx = 0
                    self.current_scout_frame = 0
                    self.current_task_count = 0
                    self.pending_queue_data = []
                else:
                    resume_video = self.video_files[self.current_video_idx]
                    print(f"[*] Checkpoint loaded successfully. Preparing to resume processing: {resume_video}")
                    print(f"[*] Progress -> CPU Frame: {self.current_scout_frame}, GPU Completed Tasks: {self.current_task_count}")
            except Exception:
                print("[!] Checkpoint file corrupted, starting from scratch.")

    def _save_checkpoint(self):
        """ Saves global state across files when suspended """
        pending = list(self.task_queue.queue)
        state = {
            "video_idx": self.current_video_idx,
            "scout_frame": self.current_scout_frame,
            "gpu_task_count": self.current_task_count,
            "pending_queue": pending
        }
        with open(config.CHECKPOINT_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=4)
        print(f"[*] Progress exported to {config.CHECKPOINT_FILE}")

    def producer_scout_thread(self, video_path, total_frames, fps, width, height):
        print(f"[CPU] Scout thread started -> {os.path.basename(video_path)}")
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_scout_frame)

        is_active = False
        rally_start_frame = 0
        hits, misses = 0, 0
        current_far_rois, current_near_rois = [], []
        frame_idx = self.current_scout_frame

        HIT_BUF, MISS_BUF = 3, 6

        while cap.isOpened():
            if self.stop_event.is_set():
                self.current_scout_frame = frame_idx
                break

            ret, frame = cap.read()
            if not ret: break

            if frame_idx % config.SCOUT_SKIP_FRAMES == 0:
                far_roi, near_roi = self.court_detector.get_rois(frame, width, height)

                if far_roi is not None:
                    hits += 1
                    misses = 0
                    current_far_rois.append(far_roi)
                    current_near_rois.append(near_roi)

                    if hits > HIT_BUF and not is_active:
                        is_active = True
                        rally_start_frame = max(0, frame_idx - (HIT_BUF * config.SCOUT_SKIP_FRAMES))
                else:
                    misses += 1
                    hits = 0
                    if misses > MISS_BUF and is_active:
                        is_active = False
                        true_end = frame_idx - (MISS_BUF * config.SCOUT_SKIP_FRAMES)
                        duration = (true_end - rally_start_frame) / fps

                        if duration >= config.MIN_RALLY_DURATION:
                            task = {
                                'start': rally_start_frame,
                                'end': true_end,
                                'duration': duration,
                                'far_roi': np.median(current_far_rois, axis=0).astype(int).tolist(),
                                'near_roi': np.median(current_near_rois, axis=0).astype(int).tolist()
                            }
                            self.task_queue.put(task)
                            print(
                                f"[CPU] Rally enqueued | Duration: {duration:.1f}s | Progress: {(frame_idx / total_frames) * 100:.1f}%")

                        current_far_rois.clear()
                        current_near_rois.clear()

            frame_idx += 1

        self.current_scout_frame = frame_idx
        cap.release()
        self.scout_finished.set()

    def consumer_yolo_thread(self, video_path, video_output_dir, fps, width, height):
        print("[GPU] Extractor thread started")
        model = YOLO(config.MODEL_PATH)
        
        # Intelligent device hardware adaptation acceleration optimization
        if torch.cuda.is_available():
            device = 'cuda:0'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
        print(f"[GPU] YOLO11 inference hardware automatically bound to: {device.upper()}")
        model.to(device)
        
        tracker = PoseTracker(model)

        # Initialize the newly added spatial rally detection module
        rally_detector = SpatialRallyDetector(fps=fps, buffer_seconds=1.5, movement_threshold=3.0)

        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        task_count = self.current_task_count

        while True:
            if self.stop_event.is_set():
                self.current_task_count = task_count
                print("[GPU] Suspension command received, exiting safely after completing current clip")
                break

            try:
                rally = self.task_queue.get(timeout=0.5)
            except queue.Empty:
                if self.scout_finished.is_set():
                    break
                continue

            task_count += 1
            duration = rally['duration']

            clip_name = f"rally_{task_count:03d}_{duration:.1f}s"
            clip_dir = os.path.join(video_output_dir, clip_name)
            os.makedirs(clip_dir, exist_ok=True)

            raw_path = os.path.join(clip_dir, "raw_clip.mp4")
            ann_path = os.path.join(clip_dir, "annotated_clip.mp4")
            json_path = os.path.join(clip_dir, "pose_data.json")

            out_raw = cv2.VideoWriter(raw_path, fourcc, fps, (width, height))
            out_ann = cv2.VideoWriter(ann_path, fourcc, fps, (width, height))

            cap.set(cv2.CAP_PROP_POS_FRAMES, rally['start'])
            curr_frame = rally['start']
            json_data = []

            fx1, fy1, fx2, fy2 = rally['far_roi']
            nx1, ny1, nx2, ny2 = rally['near_roi']

            h_far = {'box': None, 'kpts': None, 'miss': 0}
            h_near = {'box': None, 'kpts': None, 'miss': 0}

            print(f"[GPU] Annotating: {clip_name}")

            while curr_frame <= rally['end']:
                ret, frame = cap.read()
                if not ret: break

                out_raw.write(frame)
                ann_frame = frame.copy()
                f_data = {"frame": curr_frame, "far_player": None, "near_player": None}

                # Extract and track the player pose and box data for the current frame
                f_data["far_player"] = tracker.process_and_smooth(
                    frame[fy1:fy2, fx1:fx2], fx1, fy1, True, h_far, ann_frame)

                f_data["near_player"] = tracker.process_and_smooth(
                    frame[ny1:ny2, nx1:nx2], nx1, ny1, False, h_near, ann_frame)

                # Core feature: Call the secure detector reinforced with unpacking protection
                in_play = rally_detector.update(f_data["far_player"], f_data["near_player"])
                f_data["in_play"] = in_play

                # Render the status banner on the visualized video stream
                status_text = "STATE: PLAYING" if in_play else "STATE: NOT PLAYING"
                status_color = (0, 255, 0) if in_play else (0, 0, 255)  # Green (Playing) vs Red (Not Playing)
                
                # Draw a status panel background box in the top-left corner and render the text
                cv2.rectangle(ann_frame, (30, 30), (450, 100), (15, 15, 15), -1)
                cv2.putText(ann_frame, status_text, (50, 75), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 3, cv2.LINE_AA)

                out_ann.write(ann_frame)
                json_data.append(f_data)
                curr_frame += 1

            out_raw.release()
            out_ann.release()
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=4)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.current_task_count = task_count
            self.task_queue.task_done()
            print(f"[GPU] Clip completed: {clip_name}")

        cap.release()

    def process_single_video(self, video_path):
        """ Handles the complete lifecycle of a single video """
        temp_cap = cv2.VideoCapture(video_path)
        fps = temp_cap.get(cv2.CAP_PROP_FPS)
        width = int(temp_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(temp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(temp_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        temp_cap.release()

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_output_dir = os.path.join(self.output_base_dir, video_name)
        os.makedirs(video_output_dir, exist_ok=True)

        # Reset the queue and events for the current video
        self.task_queue = queue.Queue()
        for item in self.pending_queue_data:
            self.task_queue.put(item)
        self.pending_queue_data = []  # Clear buffer after loading

        self.scout_finished = threading.Event()
        self.stop_event = threading.Event()

        scout_t = threading.Thread(target=self.producer_scout_thread,
                                   args=(video_path, total_frames, fps, width, height))
        yolo_t = threading.Thread(target=self.consumer_yolo_thread,
                                  args=(video_path, video_output_dir, fps, width, height))

        scout_t.start()
        yolo_t.start()

        # Monitor control.txt
        while scout_t.is_alive() or yolo_t.is_alive():
            if os.path.exists(config.CONTROL_FILE):
                try:
                    with open(config.CONTROL_FILE, "r", encoding="utf-8") as f:
                        cmd = f.read().strip().lower()
                    if cmd == "save":
                        print("\n[*] Save command received, suspending worker threads...")
                        self.stop_event.set()
                        with open(config.CONTROL_FILE, "w", encoding="utf-8") as f:
                            f.write("saved")
                        break
                except Exception:
                    pass
            time.sleep(2)

        scout_t.join()
        yolo_t.join()

        return self.stop_event.is_set()

    def run(self):
        s_time = time.time()

        for idx in range(self.current_video_idx, len(self.video_files)):
            self.current_video_idx = idx
            video_file = self.video_files[idx]
            video_path = os.path.join(self.input_dir, video_file)

            print(f"\n{'=' * 50}")
            print(f"[*] Starting queue processing ({idx + 1}/{len(self.video_files)}): {video_file}")
            print(f"{'=' * 50}")

            is_stopped = self.process_single_video(video_path)

            if is_stopped:
                self._save_checkpoint()
                print(f"[*] Checkpoint saved successfully. Safe to close the application at any time.")
                return
            else:
                print(f"[*] Video {video_file} processing completed.")
                self.current_scout_frame = 0
                self.current_task_count = 0

        print(f"\n[!!!] Batch processing completed for all videos in the directory [!!!]")
        print(f"Total time elapsed: {(time.time() - s_time) / 60:.2f} minutes.")
        if os.path.exists(config.CHECKPOINT_FILE):
            os.remove(config.CHECKPOINT_FILE)


if __name__ == '__main__':
    pipeline = BatchTennisPipeline()
    pipeline.run()
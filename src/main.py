"""main.py — Hybrid Tennis Match Kinematics & Scene Analyzer with Automated Slicing, Compilation & Pose Annotation

Features:
- Slices detected rallies instantly using FFmpeg native copy to './src/data/rallies_new/<video_name>/'
- Automatically compiles all sliced rallies into a single consolidated 'all_rallies_combined.mp4' file.
- Dual Modes: Static Fence-cam (robust box center kinematics) and Broadcast (CLIP scene cuts).
- Optionally re-renders every cut rally clip with near/far player pose skeletons overlaid
  (near_player + far_player ROIs via CourtDetector, keypoints via PoseTracker), producing
  'rally_XXX_..._annotated.mp4' files plus 'all_rallies_combined_annotated.mp4'.
- Device Agnostic: Automatically leverages Apple Silicon (MPS) or CUDA where available.
"""
import cv2
import numpy as np
import json
import os
import time
import torch
import logging
import subprocess
from PIL import Image
from ultralytics import YOLO

import config_legacy as config
from pose_tracker import PoseTracker
from court_detector import CourtDetector

# Block internal third-party logging noise
logging.getLogger("ultralytics").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)


def get_acceleration_device():
    """Detects and returns the best hardware acceleration device for processing."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def format_timestamp(seconds):
    """Converts raw float seconds into a readable HH:MM:SS format."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def timestamp_to_seconds(ts_str):
    """Converts an HH:MM:SS string back to float seconds."""
    parts = ts_str.split(':')
    if len(parts) == 3:
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    return 0.0


def concat_videos(file_paths, combined_output_path, label="reel"):
    """Concatenates a list of video files (stream-copy) into one file using FFmpeg's concat demuxer."""
    if len(file_paths) <= 1:
        return
    concat_list_path = os.path.join(os.path.dirname(combined_output_path), f"concat_list_{label}.txt")
    try:
        with open(concat_list_path, "w", encoding="utf-8") as f:
            for file_path in file_paths:
                escaped_path = os.path.abspath(file_path).replace("'", "'\\''")
                f.write(f"file '{escaped_path}'\n")

        concat_command = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_list_path,
            "-c", "copy",
            combined_output_path
        ]
        subprocess.run(concat_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        print(f"  🏆 Success! Combined video created: {os.path.basename(combined_output_path)}")
    except subprocess.CalledProcessError:
        print(f"  ⚠️ Failed to compile the combined '{label}' reel.")
    finally:
        if os.path.exists(concat_list_path):
            try:
                os.remove(concat_list_path)
            except OSError:
                pass


def slice_and_combine_rallies(video_path, timeline, output_dir):
    """Slices playing blocks into individual clips and compiles them into one final video.
    Returns the list of sliced clip filepaths (empty list if none were produced)."""
    os.makedirs(output_dir, exist_ok=True)
    
    playing_blocks = [b for b in timeline if b["status"] == "PLAYING (Rally)"]
    if not playing_blocks:
        print("  ℹ️ No playing rally blocks detected to slice.")
        return []

    sliced_files = []
    print(f"\n🎬 Slicing {len(playing_blocks)} rally clips into: '{output_dir}/'...")
    
    for idx, block in enumerate(playing_blocks):
        start_sec = timestamp_to_seconds(block["start"])
        end_sec = timestamp_to_seconds(block["end"])
        duration = end_sec - start_sec

        # Guard against zero-duration or negative slices
        if duration <= 1.0:
            continue

        clean_ts = block["start"].replace(":", "-")
        output_filename = os.path.join(output_dir, f"rally_{idx+1:03d}_{clean_ts}.mp4")

        # Fast lossless stream-copy using FFmpeg
        command = [
            "ffmpeg", "-y",
            "-ss", f"{start_sec:.3f}",
            "-i", video_path,
            "-t", f"{duration:.3f}",
            "-c", "copy",
            output_filename
        ]

        try:
            subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            print(f"  💾 Saved clip {idx+1:02d}: {os.path.basename(output_filename)} [{block['start']} -> {block['end']}]")
            sliced_files.append(output_filename)
        except FileNotFoundError:
            print("  ❌ Error: 'ffmpeg' binary not found. Please run 'brew install ffmpeg'.")
            return
        except subprocess.CalledProcessError:
            print(f"  ⚠️ Failed to slice clip {idx+1} at {block['start']}")

    # Combine all individual clips into a single compilation file
    if len(sliced_files) > 1:
        print(f"\n🔗 Concatenating {len(sliced_files)} clips into a single reel...")
        combined_output_path = os.path.join(output_dir, "all_rallies_combined.mp4")
        concat_videos(sliced_files, combined_output_path, label="raw")

    return sliced_files


def annotate_rally_clip(clip_path, output_path, pose_model):
    """Re-renders a single cut rally clip with near/far player pose skeletons overlaid.

    Reuses CourtDetector (court-line ROI split) + PoseTracker (pose inference, EMA
    smoothing, gap-filling, and keypoint/box drawing) — both originally written
    'for use by main.py' but previously never wired in.

    Frames where the court can't be detected (e.g. broadcast close-ups/replays) are
    written through unannotated rather than dropped, so gaps degrade gracefully.
    """
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        print(f"  ⚠️ Could not open clip for annotation: {os.path.basename(clip_path)}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    court_detector = CourtDetector()
    pose_tracker = PoseTracker(pose_model)
    far_state = {"box": None, "kpts": None, "miss": 0}
    near_state = {"box": None, "kpts": None, "miss": 0}

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        annotated_frame = frame.copy()
        far_roi, near_roi = court_detector.get_rois(frame, width, height)

        if far_roi is not None:
            fx1, fy1, fx2, fy2 = far_roi
            far_crop = frame[fy1:fy2, fx1:fx2]
            pose_tracker.process_and_smooth(far_crop, fx1, fy1, True, far_state, annotated_frame)

        if near_roi is not None:
            nx1, ny1, nx2, ny2 = near_roi
            near_crop = frame[ny1:ny2, nx1:nx2]
            pose_tracker.process_and_smooth(near_crop, nx1, ny1, False, near_state, annotated_frame)

        out.write(annotated_frame)
        frame_idx += 1

    cap.release()
    out.release()
    return frame_idx > 0


class RobustKinematicDetector:
    def __init__(self, movement_threshold=12.0):
        self.movement_threshold = movement_threshold
        self.prev_centers = []

    def calculate_motion_score(self, current_boxes):
        """Calculates player displacement to flag active rallies."""
        if len(current_boxes) == 0:
            self.prev_centers = []
            return 0.0

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

        for c_cx, c_cy in current_centers:
            distances = [np.sqrt((c_cx - p_cx)**2 + (c_cy - p_cy)**2) for p_cx, p_cy in self.prev_centers]
            if distances:
                total_distance += min(distances)
                matched_count += 1

        self.prev_centers = current_centers
        return (total_distance / matched_count) if matched_count > 0 else 0.0


class BatchTennisPipeline:
    def __init__(self, mode="static", annotate=True):
        self.mode = mode  # "static" or "broadcast"
        self.annotate = annotate  # whether to render pose-annotated versions of cut rally clips
        self.input_dir = config.VIDEO_PATH
        self.rallies_output_root = "./src/data/rallies_new"
        self.device = get_acceleration_device()
        self._pose_model = None  # lazy-loaded, only if annotate=True and rallies were found

        self.video_files = sorted([f for f in os.listdir(self.input_dir) if f.lower().endswith('.mp4')])
        if not self.video_files:
            raise FileNotFoundError(f"No mp4 files found in {self.input_dir}!")

        print(f"🚀 Initialized Pipeline | Mode: {self.mode.upper()} | Device: {self.device.upper()} | Pose Annotation: {'ON' if self.annotate else 'OFF'}")

    def get_pose_model(self):
        """Lazily loads (once) the pose model used for annotating cut rally clips."""
        if self._pose_model is None:
            print(f"⏳ Loading pose model: {config.MODEL_PATH}")
            self._pose_model = YOLO(config.MODEL_PATH)
            self._pose_model.to(self.device)
        return self._pose_model

    def annotate_all_rallies(self, sliced_files, match_output_dir):
        """Renders a pose-annotated version of each cut rally clip, plus a combined annotated reel."""
        if not sliced_files:
            return

        annotated_dir = os.path.join(match_output_dir, "annotated")
        os.makedirs(annotated_dir, exist_ok=True)
        pose_model = self.get_pose_model()

        print(f"\n🦴 Annotating {len(sliced_files)} rally clips with player pose overlays...")
        annotated_files = []
        for idx, clip_path in enumerate(sliced_files):
            clip_name = os.path.splitext(os.path.basename(clip_path))[0]
            output_path = os.path.join(annotated_dir, f"{clip_name}_annotated.mp4")
            ok = annotate_rally_clip(clip_path, output_path, pose_model)
            if ok:
                print(f"  🦴 Annotated clip {idx+1:02d}: {os.path.basename(output_path)}")
                annotated_files.append(output_path)
            else:
                print(f"  ⚠️ Failed to annotate: {os.path.basename(clip_path)}")

        if len(annotated_files) > 1:
            print(f"\n🔗 Concatenating {len(annotated_files)} annotated clips into a single reel...")
            combined_output_path = os.path.join(annotated_dir, "all_rallies_combined_annotated.mp4")
            concat_videos(annotated_files, combined_output_path, label="annotated")

    def process_broadcast_clip(self, video_path):
        """Broadcast Mode: Segment camera shots and classify via zero-shot CLIP."""
        from transformers import CLIPProcessor, CLIPModel
        from scenedetect import detect, ContentDetector

        print("⏳ Initializing CLIP transformer models...")
        model_name = "openai/clip-vit-base-patch32"
        model = CLIPModel.from_pretrained(model_name).to(torch.device(self.device))
        processor = CLIPProcessor.from_pretrained(model_name)

        print("🎬 Segmenting broadcast scenes via PySceneDetect...")
        scene_list = detect(video_path, ContentDetector(threshold=30.0))
        
        labels = [
            "a wide broadcast stadium view of a tennis court court during play", 
            "a close up shot of a tennis player or crowd or replay screen"
        ]

        cap = cv2.VideoCapture(video_path)
        timeline = []

        for idx, scene in enumerate(scene_list):
            start_tc, end_tc = scene
            start_str = start_tc.get_timecode()
            end_str = end_tc.get_timecode()
            
            # Read midpoint frame of scene cut for inference
            mid_frame_idx = start_tc.get_frames() + (end_tc.get_frames() - start_tc.get_frames()) // 2
            cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            # Convert OpenCV image to PIL for CLIP consumption
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inputs = processor(text=labels, images=pil_image, return_tensors="pt", padding=True)
            inputs = {k: v.to(torch.device(self.device)) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                probs = outputs.logits_per_image.softmax(dim=-1).cpu().numpy()[0]

            is_gameplay = probs[0] > probs[1]
            status = "PLAYING (Rally)" if is_gameplay else "NON-PLAYING (Break)"
            confidence = probs[0] if is_gameplay else probs[1]

            timeline.append({"start": start_str, "end": end_str, "status": status})
            print(f"  📺 Scene {idx:03d} [{start_str} -> {end_str}] Class: {status} ({confidence:.1%})")

        cap.release()
        return timeline

    def process_static_clip(self, video_path):
        """Static Mode: Analyze player motion index to calculate the rally blocks."""
        yolo_model = YOLO("yolov8n.pt")
        yolo_model.to(self.device)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        sample_interval = int(fps / 2) or 15
        
        detector = RobustKinematicDetector(movement_threshold=12.0)
        frame_log = []
        frame_idx = 0

        print(f"🎬 Tracking movements sequentially through file...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_interval == 0:
                results = yolo_model(frame, classes=[0], verbose=False)
                avg_velocity = 0.0

                if len(results) > 0 and len(results[0].boxes) > 0:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    avg_velocity = detector.calculate_motion_score(boxes)

                status = "PLAYING (Rally)" if avg_velocity > 12.0 else "NON-PLAYING (Break)"
                timestamp = format_timestamp(frame_idx / fps)
                frame_log.append({"timestamp": timestamp, "status": status})

            frame_idx += 1
        
        cap.release()

        # Build clean chronological blocks of consecutive statuses
        if not frame_log:
            return []

        timeline_blocks = []
        current_block = {
            "status": frame_log[0]["status"], 
            "start": frame_log[0]["timestamp"], 
            "end": frame_log[0]["timestamp"]
        }

        for entry in frame_log[1:]:
            if entry["status"] == current_block["status"]:
                current_block["end"] = entry["timestamp"]
            else:
                timeline_blocks.append(current_block)
                current_block = {
                    "status": entry["status"], 
                    "start": entry["timestamp"], 
                    "end": entry["timestamp"]
                }
        timeline_blocks.append(current_block)
        return timeline_blocks

    def run(self):
        for idx, video_file in enumerate(self.video_files):
            video_path = os.path.join(self.input_dir, video_file)
            video_name = os.path.splitext(video_file)[0]
            
            print(f"\n========================================================")
            print(f"🔍 Analyzing Match ({idx + 1}/{len(self.video_files)}): {video_file}")
            print(f"========================================================")

            if self.mode == "broadcast":
                timeline = self.process_broadcast_clip(video_path)
            else:
                timeline = self.process_static_clip(video_path)

            print("\n📊 RALLY TIME CODES DETECTED:")
            print("-" * 50)
            rally_count = 0
            for block in timeline:
                if block["status"] == "PLAYING (Rally)":
                    rally_count += 1
                    print(f"  🎾 Rally {rally_count:02d}: {block['start']} ---> {block['end']}")
            
            if rally_count == 0:
                print("  ℹ️ No continuous rallies found in this match.")
            print("-" * 50)

            # Slices, compiles, and saves everything to src/data/rallies_new/<video_name>/
            match_output_dir = os.path.join(self.rallies_output_root, video_name)
            sliced_files = slice_and_combine_rallies(video_path, timeline, match_output_dir)

            # Optionally re-render each cut rally clip with pose skeleton overlays
            if self.annotate:
                self.annotate_all_rallies(sliced_files, match_output_dir)


if __name__ == '__main__':
    # Toggle between modes here: "static" (fence-cam) or "broadcast" (TV footage)
    # Toggle annotate=False to skip pose-overlay rendering and only cut/combine rallies.
    pipeline = BatchTennisPipeline(mode="broadcast", annotate=True)
    pipeline.run()
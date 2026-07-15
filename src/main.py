"""main.py — Hybrid Tennis Match Kinematics & Scene Analyzer

Features:
- Dual Modes: Static Fence-cam (robust box center kinematics) and Broadcast (CLIP scene cuts).
- Zero-Bloat: Stripped out noisy frame annotator steps, writing outputs cleanly to the terminal.
- Device Agnostic: Automatically leverages Apple Silicon (MPS) or CUDA where available.
"""
import cv2
import numpy as np
import json
import os
import time
import torch
import logging
from PIL import Image
from ultralytics import YOLO

import config_legacy as config

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
    def __init__(self, mode="static"):
        self.mode = mode  # "static" or "broadcast"
        self.input_dir = config.VIDEO_PATH
        self.device = get_acceleration_device()
        
        self.video_files = sorted([f for f in os.listdir(self.input_dir) if f.lower().endswith('.mp4')])
        if not self.video_files:
            raise FileNotFoundError(f"No mp4 files found in {self.input_dir}!")

        print(f"🚀 Initialized Pipeline | Mode: {self.mode.upper()} | Device: {self.device.upper()}")

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
                # FIX: Check for the exact status string to ignore "NON-PLAYING"
                if block["status"] == "PLAYING (Rally)":
                    rally_count += 1
                    print(f"  🎾 Rally {rally_count:02d}: {block['start']} ---> {block['end']}")
            
            if rally_count == 0:
                print("  ℹ️ No continuous rallies found in this match.")
            print("-" * 50)


if __name__ == '__main__':
    # You can quickly swap between modes here: "static" (fence-cam) or "broadcast" (TV footage)
    pipeline = BatchTennisPipeline(mode="broadcast")
    pipeline.run()
"""main.py — Hybrid Tennis Match Kinematics & Scene Analyzer with Automated Slicing, Compilation & Pose Annotation

Features:
- Slices detected rallies instantly using FFmpeg native copy to './src/data/rallies_new/<video_name>/'
- Automatically compiles all sliced rallies into a single consolidated 'all_rallies_combined.mp4' file.
- Three segmentation modes: Static Fence-cam (player-box velocity), Broadcast (CLIP scene cuts),
  and Fusion (audio impact + player motion + ball activity, weighted through a hysteresis state
  machine — see audio_video_fusion.py). Fusion mode reuses the same player-motion detector as
  static mode and the ball tracker (TrackNet or heuristic) as a third fused signal.
- Optionally re-renders every cut rally clip with near/far player pose skeletons overlaid.
  CourtDetector fits a per-frame pixel<->real-world court homography (14-pt keypoint model),
  and PoseTracker full-frame-tracks every person then picks the far/near player by whichever
  track spends the most time near a baseline in real-world coordinates -- this is what keeps
  the chair umpire/line judges near the net from being mistaken for the far player. Produces
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
from ball_tracker import BallTracker
from audio_video_fusion import compute_impact_score_series, get_score_at, RallyStateMachine

# Toggle which ball-tracking backend annotate_rally_clip() uses:
#   "tracknet"  — higher accuracy, needs src/tracknet/{model.py,general.py} + pretrained weights
#                 downloaded first (see ball_tracker_tracknet.py's module docstring for setup).
#   "heuristic" — classical CV (background subtraction + Kalman filter), no setup required.
# Falls back to "heuristic" automatically with a warning if TrackNet isn't set up yet.
BALL_TRACKER_BACKEND = "tracknet"


def get_ball_tracker(device="cpu"):
    """Instantiates the configured ball-tracking backend, falling back to the classical
    heuristic tracker if TrackNet isn't set up yet."""
    if BALL_TRACKER_BACKEND == "tracknet":
        try:
            from ball_tracker_tracknet import TrackNetBallTracker
            return TrackNetBallTracker(device=device)
        except (ImportError, FileNotFoundError) as e:
            print(f"  ⚠️ TrackNet not available ({e}); falling back to heuristic ball tracker.")
            return BallTracker()
    return BallTracker()

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


def build_timeline_blocks(frame_log):
    """Collapses a list of {"timestamp": ..., "status": ...} samples into consecutive-status
    blocks of the form {"status", "start", "end"}. Shared by static and fusion modes."""
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

        # Fast lossless stream-copy using FFmpeg.
        #
        # Two-stage seek: a coarse -ss BEFORE -i (fast keyframe seek, skips most of the file)
        # followed by a small corrective -ss AFTER -i (frame-accurate, only decodes ~2s). With
        # `-c copy`, a single -ss before -i can only cut the *video* stream at the nearest
        # preceding keyframe while the *audio* stream cuts almost exactly at start_sec -- the
        # two streams end up starting at different real timestamps, which is what shows up as
        # "delayed audio" in the exported clip. The two-stage seek plus avoid_negative_ts fixes
        # the sync while keeping nearly all of stream-copy's speed. Note this still can't
        # guarantee a frame-perfect start (that requires re-encoding), only that audio and
        # video agree with each other once cut.
        coarse_seek = max(0.0, start_sec - 2.0)
        fine_seek = start_sec - coarse_seek
        command = [
            "ffmpeg", "-y",
            "-ss", f"{coarse_seek:.3f}",
            "-i", video_path,
            "-ss", f"{fine_seek:.3f}",
            "-t", f"{duration:.3f}",
            "-avoid_negative_ts", "make_zero",
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


def annotate_rally_clip(clip_path, output_path, pose_model, court_model):
    """Re-renders a single cut rally clip with near/far player pose skeletons overlaid.

    Two passes over the clip, mirroring src/pipeline/offline_tennis_tracker.py's proven
    approach (see pose_tracker.py's module docstring for the full rationale):

      Pass 1: for every frame, fit the court homography (CourtDetector) and full-frame
        track every person, recording each track's projected real-world position
        (PoseTracker.track_frame).
      Selection: pick the far/near track IDs by which tracks spent the most time near a
        baseline in real-world coordinates (PoseTracker.select_players) -- this is what
        keeps a stationary chair umpire/line judge near the net from ever winning the
        far-player slot, without any referee-specific exception.
      Pass 2: re-read the clip and draw the two selected, EMA-smoothed tracks plus the
        ball trail.

    Frames where the court can't be detected (e.g. broadcast close-ups/replays) simply
    don't contribute a homography that frame; player tracking picks back up as soon as
    the court is visible again, and pass 2 still writes every frame (annotated or not)
    so output length always matches the source clip.
    """
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        print(f"  ⚠️ Could not open clip for annotation: {os.path.basename(clip_path)}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    court_detector = CourtDetector(court_model)
    pose_tracker = PoseTracker(pose_model, court_detector)

    # ---- Pass 1: fit homography + full-frame track every person ----
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        H = court_detector.estimate_homography(frame)
        pose_tracker.track_frame(frame, frame_idx, H)
        frame_idx += 1

    total_frames = frame_idx
    if total_frames == 0:
        cap.release()
        return False

    # ---- Select the far/near tracks, then EMA-smooth + gap-fill each for rendering ----
    far_id, near_id = pose_tracker.select_players(frame_height=height)
    far_track = pose_tracker.build_render_track(far_id)
    near_track = pose_tracker.build_render_track(near_id)

    # ---- Pass 2: re-read the clip and render the chosen tracks + ball trail ----
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ball_tracker = get_ball_tracker(device=str(pose_model.device) if hasattr(pose_model, "device") else "cpu")
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        annotated_frame = frame.copy()
        PoseTracker.draw(annotated_frame, far_track.get(frame_idx), is_far=True)
        PoseTracker.draw(annotated_frame, near_track.get(frame_idx), is_far=False)

        ball_tracker.update(frame)
        ball_tracker.draw_trail(annotated_frame)

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
    def __init__(self, mode="static", annotate=True, fusion_weights=None):
        self.mode = mode  # "static", "broadcast", or "fusion"
        self.annotate = annotate  # whether to render pose-annotated versions of cut rally clips
        # Weights for "fusion" mode's active_score = w_audio*impact + w_motion*motion + w_ball*ball.
        # Audio weighted highest by default, per the fusion doc's own rationale (impact sounds
        # are a more sensitive indicator of active play than motion/ball alone). Must sum to ~1.0.
        self.fusion_weights = fusion_weights or {"audio": 0.5, "motion": 0.3, "ball": 0.2}
        self.input_dir = config.VIDEO_PATH
        self.rallies_output_root = "./src/data/rallies_new"
        self.device = get_acceleration_device()
        self._pose_model = None   # lazy-loaded, only if annotate=True and rallies were found
        self._court_model = None  # lazy-loaded alongside _pose_model, for CourtDetector's homography

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

    def get_court_model(self):
        """Lazily loads (once) the court keypoint model used by CourtDetector to fit the
        per-frame homography. Loaded once and reused across every clip in the match, same
        as get_pose_model(), so annotate_rally_clip's CourtDetector(court_model) is cheap."""
        if self._court_model is None:
            print(f"⏳ Loading court keypoint model: {config.COURT_MODEL_PATH}")
            self._court_model = YOLO(config.COURT_MODEL_PATH)
            self._court_model.to(self.device)
        return self._court_model

    def annotate_all_rallies(self, sliced_files, match_output_dir):
        """Renders a pose-annotated version of each cut rally clip, plus a combined annotated reel."""
        if not sliced_files:
            return

        annotated_dir = os.path.join(match_output_dir, "annotated")
        os.makedirs(annotated_dir, exist_ok=True)
        pose_model = self.get_pose_model()
        court_model = self.get_court_model()

        print(f"\n🦴 Annotating {len(sliced_files)} rally clips with player pose overlays...")
        annotated_files = []
        for idx, clip_path in enumerate(sliced_files):
            clip_name = os.path.splitext(os.path.basename(clip_path))[0]
            output_path = os.path.join(annotated_dir, f"{clip_name}_annotated.mp4")
            ok = annotate_rally_clip(clip_path, output_path, pose_model, court_model)
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
        return build_timeline_blocks(frame_log)

    def process_fusion_clip(self, video_path):
        """Fusion Mode: Combines audio impact detection, player-motion score, and ball activity
        score (weighted, see self.fusion_weights) through a hysteresis state machine to detect
        rally boundaries. This is the audio-video-ball fusion design from the project doc,
        implemented on top of existing modules:
          - audio impact score: audio_video_fusion.compute_impact_score_series()
          - player motion score: RobustKinematicDetector (already used by static mode)
          - ball activity score: get_ball_tracker() (TrackNet or heuristic backend)
          - hysteresis state machine: audio_video_fusion.RallyStateMachine()
        """
        print("🔊 Extracting and analyzing audio for impact sounds...")
        audio_times, audio_scores = compute_impact_score_series(video_path, hop_sec=0.5)

        yolo_model = YOLO("yolov8n.pt")
        yolo_model.to(self.device)
        motion_detector = RobustKinematicDetector(movement_threshold=12.0)
        ball_tracker = get_ball_tracker(device=self.device)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        sample_interval = int(fps / 2) or 15  # ~2Hz decision cadence, same as static mode
        sample_interval_sec = sample_interval / fps

        state_machine = RallyStateMachine(sample_interval_sec=sample_interval_sec)

        frame_log = []
        ball_window_scores = []
        frame_idx = 0

        print("🎬 Running fused audio + motion + ball analysis...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Ball tracker needs a continuous frame stream (TrackNet's 3-frame input in
            # particular), so it runs every frame; motion/audio/decision only evaluate on the
            # coarser ~2Hz grid below, using the average ball score accumulated since last time.
            ball_result = ball_tracker.update(frame)
            ball_window_scores.append(ball_result["ball_activity_score"])

            if frame_idx % sample_interval == 0:
                results = yolo_model(frame, classes=[0], verbose=False)
                raw_motion = 0.0
                if len(results) > 0 and len(results[0].boxes) > 0:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    raw_motion = motion_detector.calculate_motion_score(boxes)
                # Normalize raw pixel-displacement motion score into 0-1. The original static-mode
                # cutoff (12.0) marked "playing" -- treat that as roughly the midpoint and saturate
                # at ~2x it, rather than just re-deriving the same hard threshold inside the fusion.
                motion_score = min(1.0, raw_motion / 24.0)

                avg_ball_score = float(np.mean(ball_window_scores)) if ball_window_scores else 0.0
                ball_window_scores = []

                t_sec = frame_idx / fps
                impact_score = get_score_at(audio_times, audio_scores, t_sec)

                active_score = (self.fusion_weights["audio"] * impact_score +
                                 self.fusion_weights["motion"] * motion_score +
                                 self.fusion_weights["ball"] * avg_ball_score)

                status = state_machine.update(active_score)
                frame_log.append({"timestamp": format_timestamp(t_sec), "status": status})

            frame_idx += 1

        cap.release()
        return build_timeline_blocks(frame_log)

    def run(self):
        for idx, video_file in enumerate(self.video_files):
            video_path = os.path.join(self.input_dir, video_file)
            video_name = os.path.splitext(video_file)[0]
            
            print(f"\n========================================================")
            print(f"🔍 Analyzing Match ({idx + 1}/{len(self.video_files)}): {video_file}")
            print(f"========================================================")

            if self.mode == "broadcast":
                timeline = self.process_broadcast_clip(video_path)
            elif self.mode == "fusion":
                timeline = self.process_fusion_clip(video_path)
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
    # Toggle between modes here: "static" (fence-cam), "broadcast" (TV footage via CLIP scene
    # classification), or "fusion" (audio impact + player motion + ball activity, weighted).
    # Toggle annotate=False to skip pose-overlay rendering and only cut/combine rallies.
    pipeline = BatchTennisPipeline(mode="broadcast", annotate=True)
    pipeline.run()
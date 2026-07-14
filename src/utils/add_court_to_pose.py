"""
add_court_keypoints.py — Complementing Court Keypoints for Annotated Rallies

Function: Complements court keypoints for legacy data in data/rallies_annotated/
          (where pose_data.json lacks the "court" field). It runs a court detection
          model frame-by-frame and writes 14 keypoints into the "court" field of each frame.
          Supports breakpoint resumption: skips processing if the first valid frame already has a "court" field.
"""
import os
import json
import ctypes
import argparse
import numpy as np
import cv2
from ultralytics import YOLO
from tqdm import tqdm

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_short_path(path):
    """Retrieves the Windows short path name to prevent path parsing issues with special characters/spaces."""
    buf = ctypes.create_unicode_buffer(512)
    if not hasattr(ctypes, "windll"):  # Non-Windows environments use the original path directly
        return path
    ctypes.windll.kernel32.GetShortPathNameW(path, buf, 512)
    return buf.value or path


def _already_done(pose_data):
    """Checks whether the 'court' field already exists by validating the first non-empty frame entry."""
    for entry in (pose_data if isinstance(pose_data, list) else pose_data.values()):
        if entry and entry.get("court") is not None:
            return True
    return False


def process_clip(clip_dir, court_model, force=False):
    pose_path = os.path.join(clip_dir, "pose_data.json")
    video_path = os.path.join(clip_dir, "raw_clip.mp4")

    if not os.path.exists(pose_path) or not os.path.exists(video_path):
        return "Missing files"

    with open(pose_path, "r", encoding="utf-8") as f:
        pose_data = json.load(f)

    if not force and _already_done(pose_data):
        return "Skipped"

    short_path = get_short_path(video_path)
    cap = cv2.VideoCapture(short_path)
    if not cap.isOpened():
        return "Failed to open video"

    is_list = isinstance(pose_data, list)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = court_model(frame, verbose=False)
        kps_out = []
        if results and results[0].keypoints is not None:
            kps = results[0].keypoints
            if kps.xy is not None and len(kps.xy) > 0:
                xy = kps.xy[0].cpu().numpy()      # Shape: (14, 2)
                conf = kps.conf[0].cpu().numpy() if kps.conf is not None else np.ones(len(xy))
                for i in range(len(xy)):
                    kps_out.append([float(xy[i, 0]), float(xy[i, 1]), float(conf[i])])

        # Fill up to 14 points if any are missing
        while len(kps_out) < 14:
            kps_out.append([0.0, 0.0, 0.0])

        if is_list:
            if frame_idx < len(pose_data):
                if pose_data[frame_idx] is None:
                    pose_data[frame_idx] = {"frame": frame_idx, "court": kps_out,
                                            "near_player": None, "far_player": None}
                else:
                    pose_data[frame_idx]["court"] = kps_out
        else:
            key = str(frame_idx)
            if key in pose_data:
                pose_data[key]["court"] = kps_out

        frame_idx += 1

    cap.release()

    with open(pose_path, "w", encoding="utf-8") as f:
        json.dump(pose_data, f, ensure_ascii=False)

    return f"Completed {frame_idx} frames"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=os.path.join(_PROJECT_DIR, "data", "rallies_annotated"))
    parser.add_argument("--model", default=os.path.join(_PROJECT_DIR, "models", "court", "best.pt"))
    parser.add_argument("--force", action="store_true", help="Force re-running already processed rallies")
    args = parser.parse_args()

    print(f"Loading court model: {args.model}")
    court_model = YOLO(args.model)

    clips = sorted(d for d in os.listdir(args.data_root)
                   if os.path.isdir(os.path.join(args.data_root, d)))

    done = skipped = failed = 0
    for clip_name in tqdm(clips, desc="Complementing court keypoints"):
        clip_dir = os.path.join(args.data_root, clip_name)
        result = process_clip(clip_dir, court_model, force=args.force)
        if result == "Skipped":
            skipped += 1
        elif result.startswith("Completed"):
            done += 1
        else:
            failed += 1
            tqdm.write(f"  [Failed] {clip_name}: {result}")

    print(f"\nExecution Summary: {done} processed, {skipped} skipped, {failed} failed")


if __name__ == "__main__":
    main()
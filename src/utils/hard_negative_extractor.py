"""
hard_negative_extractor.py — Extracts low-confidence frames from pipeline_subpar.txt

Usage:
    python src/utils/hard_negative_extractor.py
    python src/utils/hard_negative_extractor.py --frames-per-clip 3 --conf-threshold 0.4

Outputs:
    data/person_sorter/hard_negatives/images/  — Extracted frame images
    data/person_sorter/hard_negatives/manifest.csv — Frame provenance log
"""

import os
import csv
import argparse
from pathlib import Path

import ctypes

import cv2
import numpy as np
from ultralytics import YOLO


def _short_path(path: str) -> str:
    """Convert a Windows path containing Chinese characters to an 8.3 short path for cv2 compatibility"""
    buf = ctypes.create_unicode_buffer(512)
    if not hasattr(ctypes, "windll"):  # Use the original path directly on non-Windows systems
        return path
    ctypes.windll.kernel32.GetShortPathNameW(path, buf, 512)
    return buf.value or path


def imwrite_unicode(path: str, img) -> bool:
    """Image writing with support for paths containing Chinese characters"""
    ext = Path(path).suffix
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        return False
    with open(path, "wb") as f:
        f.write(buf.tobytes())
    return True

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.dirname(_SCRIPT_DIR)
_PROJECT_DIR = os.path.dirname(_SRC_DIR)

SUBPAR_LOG = os.path.join(_PROJECT_DIR, "logs", "pipeline_subpar.txt")
MODEL_PATH = os.path.join(_PROJECT_DIR, "models", "person", "best.pt")
OUTPUT_DIR = os.path.join(_PROJECT_DIR, "data", "person_sorter", "hard_negatives", "images")
MANIFEST_PATH = os.path.join(_PROJECT_DIR, "data", "person_sorter", "hard_negatives", "manifest.csv")


def parse_subpar_log(log_path: str, problem_type: str) -> list[dict]:
    """Parse pipeline_subpar.txt and return the list of clips matching the given problem type"""
    # NOTE: these keyword values are kept in Chinese because they must match
    # the (Chinese-language) issue descriptions written into pipeline_subpar.txt
    # by the upstream logging pipeline. Translating them here would break the
    # matching logic unless the log file's contents are translated as well.
    keyword_map = {
        "player": "运动员偏低",   # "player detection confidence low"
        "pose": "肢体偏低",       # "pose/limb confidence low"
        "court": "球场线偏低",    # "court line confidence low"
    }

    entries = []
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(" -> ", 1)
            if len(parts) != 2:
                continue
            video_path, issues_str = parts[0].strip(), parts[1].strip()

            if problem_type == "all":
                matched = True
            else:
                keyword = keyword_map.get(problem_type, "")
                matched = keyword in issues_str

            if matched:
                entries.append({"video_path": video_path, "issues": issues_str})

    return entries


def extract_worst_frames(
    video_path: str,
    model: YOLO,
    frames_per_clip: int,
    conf_threshold: float,
) -> list[dict]:
    """
    Run inference on a single video and return info for the N lowest-confidence frames.
    Return format: [{"frame_id": int, "frame": ndarray, "min_conf": float}]
    """
    cap = cv2.VideoCapture(_short_path(video_path))
    if not cap.isOpened():
        print(f"  [Skipped] Cannot open: {video_path}")
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Sample uniformly, capped at 60 frames to avoid processing too slowly
    sample_step = max(1, total // 60)

    frame_scores = []
    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % sample_step == 0:
            results = model.predict(frame, verbose=False, conf=conf_threshold)
            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                min_conf = float(boxes.conf.min().item())
            else:
                # No person detected at all — record confidence as 0
                min_conf = 0.0
            frame_scores.append({"frame_id": frame_id, "frame": frame.copy(), "min_conf": min_conf})
        frame_id += 1

    cap.release()

    if not frame_scores:
        return []

    # Sort ascending by confidence, take the N worst frames
    frame_scores.sort(key=lambda x: x["min_conf"])
    return frame_scores[:frames_per_clip]


def main():
    parser = argparse.ArgumentParser(description="Extract low-confidence frames for re-annotation")
    parser.add_argument("--problem-type", default="player",
                        choices=["player", "pose", "court", "all"],
                        help="Problem type to filter by (default: player)")
    parser.add_argument("--frames-per-clip", type=int, default=5,
                        help="Number of frames to extract per clip (default: 5)")
    parser.add_argument("--conf-threshold", type=float, default=0.3,
                        help="YOLO inference confidence threshold (default: 0.3)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    print(f"Parsing log: {SUBPAR_LOG}")
    entries = parse_subpar_log(SUBPAR_LOG, args.problem_type)
    print(f"Found {len(entries)} problem clips (type: {args.problem_type})")

    manifest_rows = []
    saved_count = 0

    for i, entry in enumerate(entries):
        video_path = entry["video_path"]
        # The path may be an old path; try to map it to the current project directory
        if not os.path.exists(video_path):
            # Try to extract the relative portion from the end of the path and rebuild it
            # Old path format: <old_dataset>\【xx】...\rally_xxx\raw_clip.mp4
            parts = Path(video_path).parts
            try:
                # Find the position of the match folder (starts with "【")
                match_idx = next(j for j, p in enumerate(parts) if p.startswith("【"))
                rel_path = os.path.join(*parts[match_idx:])
                new_path = os.path.join(_PROJECT_DIR, "data", "rallies_new", rel_path)
                if os.path.exists(new_path):
                    video_path = new_path
                else:
                    print(f"  [{i+1}/{len(entries)}] File does not exist, skipping: {entry['video_path']}")
                    continue
            except StopIteration:
                print(f"  [{i+1}/{len(entries)}] Could not parse path, skipping: {entry['video_path']}")
                continue

        print(f"  [{i+1}/{len(entries)}] Processing: {Path(video_path).parent.name}")
        worst_frames = extract_worst_frames(
            video_path, model, args.frames_per_clip, args.conf_threshold
        )

        # Use match name + rally name as the filename prefix
        match_name = Path(video_path).parent.parent.name
        rally_name = Path(video_path).parent.name
        prefix = f"{match_name}_{rally_name}"

        for finfo in worst_frames:
            fname = f"{prefix}_frame{finfo['frame_id']:04d}.jpg"
            out_path = os.path.join(OUTPUT_DIR, fname)
            imwrite_unicode(out_path, finfo["frame"])
            manifest_rows.append({
                "filename": fname,
                "source_video": entry["video_path"],
                "frame_id": finfo["frame_id"],
                "min_conf": round(finfo["min_conf"], 4),
                "issues": entry["issues"],
            })
            saved_count += 1

    # Write manifest.csv
    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "source_video", "frame_id", "min_conf", "issues"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nDone! Extracted {saved_count} frames in total")
    print(f"Images saved to: {OUTPUT_DIR}")
    print(f"Manifest saved to: {MANIFEST_PATH}")
    print(f"\nNext step: open {OUTPUT_DIR} with LabelImg to annotate")
    print("Choose YOLO annotation format, classes: 0=player_near, 1=player_far")


if __name__ == "__main__":
    main()
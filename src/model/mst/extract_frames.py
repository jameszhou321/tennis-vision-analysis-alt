"""extract_frames.py — Pre-extract full-frame JPEGs for direct reading in dataset.py (skipping video seeking)

Usage: python extract_frames.py [--data_root ...]
Output: Generates frames/{000000.jpg, ...} under each rally directory
"""
import os
import ctypes
import argparse
import numpy as np
import cv2
from tqdm import tqdm


def get_short_path(path):
    buf = ctypes.create_unicode_buffer(512)
    if not hasattr(ctypes, "windll"):  # Use original path directly on non-Windows systems
        return path
    ctypes.windll.kernel32.GetShortPathNameW(path, buf, 512)
    return buf.value or path


def save_jpg(path, img):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if ok:
        with open(path, "wb") as f:
            f.write(buf.tobytes())


def extract_clip(clip_dir):
    frames_dir = os.path.join(clip_dir, "frames")
    video_path = os.path.join(clip_dir, "raw_clip.mp4")

    if not os.path.exists(video_path):
        return "no_video"

    # Resume from breakpoint: skip if directory exists and the frame count matches the video
    short_path = get_short_path(video_path)
    cap = cv2.VideoCapture(short_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if total <= 0:
        return "no_frames"

    if os.path.isdir(frames_dir):
        existing = len([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
        if existing >= total:
            return "skip"

    os.makedirs(frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(short_path)
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Resize to training resolution to save disk space & accelerate loading speed
        frame = cv2.resize(frame, (320, 192))
        save_jpg(os.path.join(frames_dir, f"{idx:06d}.jpg"), frame)
        idx += 1
    cap.release()
    return f"ok:{idx}"


def main():
    parser = argparse.ArgumentParser()
    _mst_dir = os.path.dirname(os.path.abspath(__file__))
    _project_dir = os.path.dirname(os.path.dirname(os.path.dirname(_mst_dir)))
    parser.add_argument("--data_root", default=os.path.join(_project_dir, "data", "rallies_train"))
    args = parser.parse_args()

    clips = sorted(d for d in os.listdir(args.data_root)
                   if os.path.isdir(os.path.join(args.data_root, d)))

    skipped = done = failed = 0
    for clip_name in tqdm(clips, desc="Extracting full frames"):
        result = extract_clip(os.path.join(args.data_root, clip_name))
        if result == "skip":
            skipped += 1
        elif result.startswith("ok"):
            done += 1
        else:
            failed += 1
            print(f"  Skipped {clip_name}: {result}")

    print(f"\nCompleted: {done} rallies extracted, {skipped} skipped, {failed} failed.")


if __name__ == "__main__":
    main()
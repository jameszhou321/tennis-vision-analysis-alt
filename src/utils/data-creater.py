"""data-creater.py — Person classification training data collection tool

Function: randomly samples frames from data/rallies_new/ and stores them in
data/person_sorter/image/ for labeling
"""
import os
import cv2
import random
import numpy as np
from pathlib import Path


def extract_frames_per_video_folder(base_dir, output_dir, frames_per_video=10):
    base_path = Path(base_dir)
    out_path = Path(output_dir)

    # Ensure the output directory exists
    out_path.mkdir(parents=True, exist_ok=True)

    total_videos_processed = 0
    total_images_saved = 0

    print(f"Starting directory scan: {base_path}")

    # 1. Iterate over the ten top-level video folders (e.g. Video_01, Video_02...)
    for video_folder in base_path.iterdir():
        if not video_folder.is_dir():
            continue

        print(f"Processing video group: {video_folder.name}")

        # Collect all raw_clip.mp4 paths under the current top-level folder
        # rglob recursively searches all subdirectories for the target file
        clip_files = list(video_folder.rglob("raw_clip.mp4"))

        if not clip_files:
            print(f"No raw_clip.mp4 found under {video_folder.name}, skipping.")
            continue

        frames_collected = 0
        attempts = 0
        max_attempts = frames_per_video * 5  # Prevents an infinite loop if everything is corrupted

        # 2. Loop-sample until we collect enough frames or hit the max attempt count
        while frames_collected < frames_per_video and attempts < max_attempts:
            attempts += 1

            # Randomly pick a clip file
            clip_file = random.choice(clip_files)

            # Open the video with OpenCV
            cap = cv2.VideoCapture(str(clip_file))
            if not cap.isOpened():
                # Skip corrupted videos (e.g. "moov atom not found")
                cap.release()
                continue

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                cap.release()
                continue

            # Randomly generate a frame index
            random_frame_idx = random.randint(0, total_frames - 1)

            # Seek to that frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_idx)
            ret, frame = cap.read()

            if ret and frame is not None:
                # Build the filename: <top_folder_name>_<clip_folder_name>_frame<frame_idx>.jpg
                clip_folder_name = clip_file.parent.name
                img_name = f"{video_folder.name}_{clip_folder_name}_frame{random_frame_idx:04d}.jpg"
                img_save_path = out_path / img_name

                # !!! KEY FIX !!!
                # Use cv2.imencode + numpy to write the file — this fully resolves
                # the issue of images failing to save under Chinese-character paths on Windows
                is_success, im_buf_arr = cv2.imencode(".jpg", frame)
                if is_success:
                    im_buf_arr.tofile(str(img_save_path))
                    frames_collected += 1
                    total_images_saved += 1

            # Release the video handle in preparation for the next sample
            cap.release()

        if frames_collected < frames_per_video:
            print(f"{video_folder.name}: only {frames_collected} frames extracted successfully "
                  f"(possibly too few usable clips, or too many corrupted)")
        else:
            print(f"   Successfully extracted {frames_collected} images")

        total_videos_processed += 1

    print("-" * 30)
    print("Frame extraction task completed successfully!")
    print(f"Total video folders processed: {total_videos_processed}")
    print(f"Total labeled images generated and saved: {total_images_saved}")
    print(f"Images saved to: {out_path.absolute()}")


# ==========================================
# Execution
# ==========================================
if __name__ == "__main__":
    import os as _os
    _PROJECT_DIR = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    SOURCE_DIR = _os.path.join(_PROJECT_DIR, "data", "rallies_new")
    TARGET_DIR = _os.path.join(_PROJECT_DIR, "data", "person_sorter", "image")
    extract_frames_per_video_folder(SOURCE_DIR, TARGET_DIR, frames_per_video=10)
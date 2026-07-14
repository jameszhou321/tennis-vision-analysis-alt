"""
Analyze the frame-level distribution of the five action classes in the dataset.
Usage:
  python src/utils/analyze_class_distribution.py
  python src/utils/analyze_class_distribution.py --root data/rallies_train_trimmed
"""

import json
import os
import argparse
from collections import Counter
from pathlib import Path

ACTION_NAMES = {0: "Waiting", 1: "Forehand", 2: "Backhand", 3: "Serve", 4: "Movement"}
FPS = 30


def analyze(data_root):
    data_root = Path(data_root)
    if not data_root.exists():
        print(f"Error: directory does not exist: {data_root}")
        return

    rally_dirs = sorted([d for d in data_root.iterdir() if d.is_dir()])
    total_frames_per_class = Counter()
    total_segments_per_class = Counter()
    total_video_frames = 0
    per_rally = []

    for rally_dir in rally_dirs:
        ann_path = rally_dir / "annotations.json"
        if not ann_path.exists():
            continue

        with open(ann_path, "r", encoding="utf-8") as f:
            annotations = json.load(f)

        # Count the total number of video frames (obtained from raw_clip.mp4)
        video_path = rally_dir / "raw_clip.mp4"
        if video_path.exists():
            import cv2
            cap = cv2.VideoCapture(str(video_path))
            video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        else:
            # Fallback: use the end_time of the last segment * FPS
            if annotations:
                video_frames = int(round(annotations[-1]["end_time"] * FPS))
            else:
                video_frames = 0

        total_video_frames += video_frames

        class_frames = Counter()
        class_segments = Counter()
        for seg in annotations:
            aid = seg.get("action_id")
            dur = seg["end_time"] - seg["start_time"]
            n_frames = int(round(dur * FPS))
            class_frames[aid] += n_frames
            class_segments[aid] += 1

        total_frames_per_class += class_frames
        total_segments_per_class += class_segments

        per_rally.append({
            "name": rally_dir.name,
            "video_frames": video_frames,
            "class_frames": dict(class_frames),
        })

    # ── Print report ──
    labeled_frames = sum(total_frames_per_class.values())

    print("=" * 60)
    print(f"Dataset:              {data_root}")
    print(f"Number of clips:      {len(rally_dirs)}")
    print(f"Total video frames:   {total_video_frames}")
    print(f"Total labeled frames: {labeled_frames} ({labeled_frames/total_video_frames*100:.1f}% of video frames are labeled)")
    print()

    # Per-class distribution
    print(f"{'Action':>10}  {'Frames':>8}  {'Share':>6}  {'Segs':>6}  {'AvgLen(f)':>10}  {'AvgLen(s)':>10}")
    print("-" * 60)
    for aid in sorted(ACTION_NAMES):
        nf = total_frames_per_class.get(aid, 0)
        ns = total_segments_per_class.get(aid, 0)
        pct = nf / labeled_frames * 100 if labeled_frames > 0 else 0
        avg = nf / ns if ns > 0 else 0
        avg_s = avg / FPS
        print(f"{ACTION_NAMES[aid]:>10}  {nf:>8}  {pct:>5.1f}%  {ns:>6}  {avg:>10.1f}  {avg_s:>10.3f}")

    print("-" * 60)

    # Breakdown of "Waiting" frames by position (leading / trailing / rough estimate)
    print()
    wait_leading = 0
    wait_trailing = 0
    wait_middle = 0
    for r in per_rally:
        ann_path = data_root / r["name"] / "annotations.json"
        if not ann_path.exists():
            continue
        with open(ann_path, "r", encoding="utf-8") as f:
            annots = json.load(f)
        for i, seg in enumerate(annots):
            if seg.get("action_id") != 0:
                continue
            dur = seg["end_time"] - seg["start_time"]
            nf = int(round(dur * FPS))
            if i == 0:
                wait_leading += nf
            elif i == len(annots) - 1:
                # Note: should also check whether the second-to-last segment is
                # also "Waiting" (i.e. a contiguous waiting run at the end)
                wait_trailing += nf
            else:
                wait_middle += nf

    print("Waiting-frame breakdown (by position):")
    print(f"  Leading waiting:   {wait_leading:>8} frames ({wait_leading/ max(labeled_frames,1)*100:.1f}%)")
    print(f"  Middle waiting:    {wait_middle:>8} frames ({wait_middle/ max(labeled_frames,1)*100:.1f}%)")
    print(f"  Trailing waiting:  {wait_trailing:>8} frames ({wait_trailing/ max(labeled_frames,1)*100:.1f}%)")

    print()
    print("Per-clip statistics (sorted by waiting-frame share, top 15):")
    print(f"{'Clip':>24}  {'Total':>6}  {'Wait':>6}  {'Wait%':>6}  {'FH':>5}  {'BH':>5}  {'Serve':>5}  {'Move':>5}")
    per_rally.sort(key=lambda r: r["class_frames"].get(0, 0) / max(sum(r["class_frames"].values()), 1), reverse=True)
    for r in per_rally[:15]:
        cf = r["class_frames"]
        total = sum(cf.values())
        wait_pct = cf.get(0, 0) / total * 100 if total > 0 else 0
        fh = cf.get(1, 0)
        bh = cf.get(2, 0)
        sv = cf.get(3, 0)
        mv = cf.get(4, 0)
        name = r["name"][:24]
        print(f"{name:>24}  {total:>6}  {cf.get(0,0):>6}  {wait_pct:>5.1f}%  {fh:>5}  {bh:>5}  {sv:>5}  {mv:>5}")

    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze the frame-level distribution of the five action classes in the dataset")
    parser.add_argument("--root", default="data/rallies_train", help="Dataset root directory")
    args = parser.parse_args()
    analyze(args.root)
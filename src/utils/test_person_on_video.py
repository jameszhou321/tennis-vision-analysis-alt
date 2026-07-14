"""
test_person_on_video.py — Per-frame inference on rallies_new clips, outputting videos with detection boxes.
Usage: python src/utils/test_person_on_video.py
Outputs: results/video_person_test/
  - <rally_name>.mp4    Complete video with detection bounding boxes
  - summary.json        Statistics for each rally (missed frames, false positives, and evaluation scores)
"""
import os
import random
import json
import ctypes
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ── Path Configurations ───────────────────────────────────────────────────
CURRENT_DIR = Path(__file__).parent
PROJECT_DIR = CURRENT_DIR.parent.parent
MODEL_PATH  = PROJECT_DIR / "runs" / "person_training" / "hard_neg_finetune_v12" / "weights" / "best.pt"
RALLY_DIR   = PROJECT_DIR / "data" / "rallies_new"
OUT_DIR     = PROJECT_DIR / "results" / "video_person_test"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_RALLIES = 10   # Number of randomly sampled rallies
CONF      = 0.4  # Inference confidence threshold

CLASS_NAMES = {0: "near", 1: "far"}
COLORS      = {0: (0, 200, 255), 1: (255, 100, 0)}  # BGR: Yellow / Orange


def get_short_path(path_str: str) -> str:
    try:
        buf = ctypes.create_unicode_buffer(260)
        if not hasattr(ctypes, "windll"):  # Non-Windows systems use the original path directly
            return path_str
        ctypes.windll.kernel32.GetShortPathNameW(path_str, buf, 260)
        return buf.value or path_str
    except Exception:
        return path_str


def draw_detections(frame: np.ndarray, detections: list, frame_idx: int, total: int) -> np.ndarray:
    vis = frame.copy()
    h, w = vis.shape[:2]

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["box"]]
        cls   = det["cls"]
        conf  = det["conf"]
        color = COLORS.get(cls, (200, 200, 200))
        label = f"{CLASS_NAMES.get(cls, cls)}:{conf:.2f}"
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, label, (x1, max(y1 - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # Frame number + Detection count (bottom-left corner)
    info = f"frame {frame_idx}/{total}  det={len(detections)}"
    cv2.putText(vis, info, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # Red warning overlay when no detections are found
    if len(detections) == 0:
        cv2.putText(vis, "NO DETECTION", (w // 2 - 80, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    return vis


def process_rally(model, video_path: Path, out_video_path: Path) -> dict:
    """Run per-frame inference on an entire rally clip, save the boxed video, and return telemetry statistics."""
    short = get_short_path(str(video_path))
    cap = cv2.VideoCapture(short)
    if not cap.isOpened():
        return None

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_short = get_short_path(str(out_video_path))
    writer = cv2.VideoWriter(
        out_short,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps, (width, height)
    )

    frame_idx   = 0
    miss_frames = 0   # Count of frames with zero objects detected
    fp_frames   = 0   # Count of frames with > 2 detections (likely false positives, as tennis usually has 2 players)
    det_per_frame = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(source=frame, conf=CONF, verbose=False)
        dets = []
        if results and results[0].boxes:
            for det in results[0].boxes:
                x1, y1, x2, y2 = [float(v) for v in det.xyxy[0]]
                dets.append({
                    "cls":  int(det.cls[0]),
                    "conf": float(det.conf[0]),
                    "box":  (x1, y1, x2, y2),
                })

        n = len(dets)
        det_per_frame.append(n)
        if n == 0:
            miss_frames += 1
        if n > 2:
            fp_frames += 1

        vis = draw_detections(frame, dets, frame_idx, total)
        writer.write(vis)
        frame_idx += 1

    cap.release()
    writer.release()

    miss_rate = miss_frames / frame_idx if frame_idx > 0 else 1.0
    score     = round(1.0 - miss_rate, 3)
    return {
        "total_frames": frame_idx,
        "miss_frames":  miss_frames,
        "fp_frames":    fp_frames,
        "miss_rate":    round(miss_rate, 3),
        "score":        score,
        "avg_det":      round(sum(det_per_frame) / len(det_per_frame), 2) if det_per_frame else 0,
    }


def collect_clips(rally_dir: Path) -> list[Path]:
    """Recursively discover and gather all raw_clip.mp4 files within the source directory."""
    clips = []
    for root, _, files in os.walk(str(rally_dir)):
        for f in files:
            if f == "raw_clip.mp4":
                clips.append(Path(root) / f)
    return clips


def main():
    clips = collect_clips(RALLY_DIR)
    if not clips:
        print(f"No videos found. Verify target directory path: {RALLY_DIR}")
        return

    random.seed(42)
    selected = random.sample(clips, min(N_RALLIES, len(clips)))
    print(f"Discovered {len(clips)} total rallies. Randomly selected {len(selected)} clips.\n")

    print(f"Loading weights matrix model: {MODEL_PATH}")
    model = YOLO(get_short_path(str(MODEL_PATH)))

    results = []
    for clip in selected:
        # Generate the structured output string sequence using "MatchName__RallyName" format
        match_name = clip.parent.parent.name[:30]
        rally_name = clip.parent.name
        out_name   = f"{match_name}__{rally_name}.mp4"
        out_path   = OUT_DIR / out_name

        print(f"  Processing: {match_name} / {rally_name}")
        stats = process_rally(model, clip, out_path)
        if stats is None:
            print(f"    Unable to read clip file, skipping asset tracking.")
            continue

        print(f"    Frames Total={stats['total_frames']}, Missed Frames={stats['miss_frames']} "
              f"({stats['miss_rate']*100:.1f}%), Suspected False Positives={stats['fp_frames']}, "
              f"Performance Score={stats['score']:.2f}")
        results.append({
            "match":    match_name,
            "rally":    rally_name,
            "output":   out_name,
            **stats,
        })

    # Sort results metrics summary list
    results.sort(key=lambda x: x["score"], reverse=True)

    summary_path = OUT_DIR / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    good = [r for r in results if r["score"] >= 0.8]
    bad  = [r for r in results if r["score"] <  0.6]

    print(f"High Performance Clips (Miss Rate ≤ 20%, Count: {len(good)}):")
    for r in good:
        print(f"   [{r['score']:.2f}] {r['output']}")

    print(f"\nLow Performance Clips (Miss Rate > 40%, Count: {len(bad)}):")
    for r in bad:
        print(f"   [{r['score']:.2f}] {r['output']}")

    print(f"\nVideo files exported to: {OUT_DIR}")
    print(f"Evaluation report file:   {summary_path}")


if __name__ == "__main__":
    main()
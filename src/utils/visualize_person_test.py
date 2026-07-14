"""
visualize_person_test.py — Visualizes person detection results.
Usage: python src/utils/visualize_person_test.py
Outputs: results/person_test/viz/
"""
import os
import json
import ctypes
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

CURRENT_DIR = Path(__file__).parent
PROJECT_DIR = CURRENT_DIR.parent.parent
MODEL_PATH = PROJECT_DIR / "models" / "person" / "best.pt"
DATA_DIR = PROJECT_DIR / "data" / "person_sorter" / "images"
LABELS_DIR = PROJECT_DIR / "data" / "person_sorter" / "labels"
RESULTS_JSON = PROJECT_DIR / "results" / "person_test" / "per_image_results.json"
VIZ_DIR = PROJECT_DIR / "results" / "person_test" / "viz"
VIZ_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {0: "near", 1: "far"}
COLORS = {
    "gt": (0, 255, 0),       # Green = Ground Truth (GT)
    "pred_hi": (0, 0, 255),  # Red = High-confidence prediction (conf >= 0.5)
    "pred_lo": (0, 165, 255) # Orange = Low-confidence prediction (conf < 0.5)
}


def get_short_path(path_str):
    try:
        buf = ctypes.create_unicode_buffer(260)
        if not hasattr(ctypes, "windll"):  # Non-Windows systems use the original path directly
            return path_str
        ctypes.windll.kernel32.GetShortPathNameW(path_str, buf, 260)
        return buf.value
    except:
        return path_str


def read_img(path):
    buf = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def load_labels(img_path, split):
    label_path = LABELS_DIR / split / img_path.name.replace(".jpg", ".txt")
    if not label_path.exists():
        return []
    labels = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                labels.append({
                    "class": int(parts[0]),
                    "bbox": tuple(map(float, parts[1:5]))
                })
    return labels


def draw_box(img, bbox, label, color, thickness=2):
    h, w = img.shape[:2]
    x_c, y_c, bw, bh = bbox
    x1 = int((x_c - bw / 2) * w)
    y1 = int((y_c - bh / 2) * h)
    x2 = int((x_c + bw / 2) * w)
    y2 = int((y_c + bh / 2) * h)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(img, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


def visualize_image(model, img_path, split, out_path):
    img = read_img(img_path)
    if img is None:
        return
    h, w = img.shape[:2]

    # Ground Truth Green Boxes
    gt_labels = load_labels(img_path, split)
    for gt in gt_labels:
        cls_name = CLASS_NAMES.get(gt["class"], str(gt["class"]))
        draw_box(img, gt["bbox"], f"GT:{cls_name}", COLORS["gt"])

    # Inference: Run with conf=0.25 to show all candidates
    img_short = get_short_path(str(img_path))
    results = model.predict(source=img_short, conf=0.25, verbose=False)
    if results and results[0].boxes:
        for det in results[0].boxes:
            x_c = float((det.xyxy[0][0] + det.xyxy[0][2]) / 2 / w)
            y_c = float((det.xyxy[0][1] + det.xyxy[0][3]) / 2 / h)
            bw = float((det.xyxy[0][2] - det.xyxy[0][0]) / w)
            bh = float((det.xyxy[0][3] - det.xyxy[0][1]) / h)
            conf = float(det.conf[0])
            cls_name = CLASS_NAMES.get(int(det.cls[0]), str(int(det.cls[0])))
            color = COLORS["pred_hi"] if conf >= 0.5 else COLORS["pred_lo"]
            draw_box(img, (x_c, y_c, bw, bh), f"{cls_name}:{conf:.2f}", color)

    # Plot Legend Overlay
    cv2.putText(img, "GREEN=GT  RED=pred>=0.5  ORANGE=pred<0.5",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # Export out (safe approach for paths containing special/multi-byte characters)
    ext = out_path.suffix
    ok, buf = cv2.imencode(ext, img)
    if ok:
        with open(str(out_path), "wb") as f:
            f.write(buf.tobytes())


def main():
    with open(RESULTS_JSON, "r", encoding="utf-8") as f:
        per_image = json.load(f)

    # Group records by tracking categorization performance
    complete_miss = [r for r in per_image if r["pred_count"] == 0 and r["gt_count"] > 0]
    partial_miss  = [r for r in per_image if 0 < r["pred_count"] < r["gt_count"]]
    correct       = [r for r in per_image if r["tp"] == r["gt_count"] and r["gt_count"] > 0]

    samples = (
        [("miss", r) for r in complete_miss[:10]] +
        [("partial", r) for r in partial_miss[:10]] +
        [("ok", r) for r in correct[:5]]
    )

    print(f"Complete Misses: {len(complete_miss)}, Partial Misses: {len(partial_miss)}, Correct: {len(correct)}")
    print(f"Generating {len(samples)} visualization images...")

    model_short = get_short_path(str(MODEL_PATH))
    model = YOLO(model_short)

    for tag, record in samples:
        rel_path = record["image"]
        # rel_path format: data/person_sorter/images/train/xxx.jpg
        img_path = PROJECT_DIR / rel_path
        split = "train" if "train" in rel_path else "val"
        out_name = f"{tag}_{img_path.stem[:60]}.jpg"
        out_path = VIZ_DIR / out_name
        visualize_image(model, img_path, split, out_path)
        print(f"  Exported: {out_name}")

    print(f"\nVisualization plots saved to: {VIZ_DIR}")


if __name__ == "__main__":
    main()
"""
test_person_detector.py — Full-scale Evaluation for Player Detection Model
Function: Runs inference across all training and validation images to generate accuracy reports and verification summaries.
"""
import os
import json
import ctypes
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict

# ── Path Configurations ───────────────────────────────────────────────
CURRENT_DIR = Path(__file__).parent
PROJECT_DIR = CURRENT_DIR.parent  # Project_Annotation_and_Testing/
MODEL_PATH = PROJECT_DIR / "runs" / "person_training" / "hard_neg_finetune_v12" / "weights" / "best.pt"
DATA_DIR = PROJECT_DIR / "data" / "person_sorter" / "images"
LABELS_DIR = PROJECT_DIR / "data" / "person_sorter" / "labels"
RESULTS_DIR = PROJECT_DIR / "results" / "person_test"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def get_short_path(path_str):
    """Converts long Windows Chinese paths to short DOS paths to prevent parsing issues."""
    try:
        buf = ctypes.create_unicode_buffer(260)
        if not hasattr(ctypes, "windll"):  # Return the original path directly if non-Windows
            return path_str
        ctypes.windll.kernel32.GetShortPathNameW(path_str, buf, 260)
        return buf.value
    except:
        return path_str

# Class ID Mappings
CLASS_NAMES = {0: "player_near", 1: "player_far"}


def load_labels(img_path, split):
    """Reads Ground Truth (GT) from .txt label files in YOLO format."""
    label_path = LABELS_DIR / split / img_path.name.replace(".jpg", ".txt")
    if not label_path.exists():
        return []

    labels = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                x_center, y_center, w, h = map(float, parts[1:5])
                labels.append({
                    "class": cls_id,
                    "bbox": (x_center, y_center, w, h)
                })
    return labels


def iou(box1, box2):
    """Calculates Intersection over Union (IoU) for two YOLO-formatted bounding boxes."""
    def yolo_to_xyxy(x_c, y_c, w, h):
        x1 = x_c - w / 2
        y1 = y_c - h / 2
        x2 = x_c + w / 2
        y2 = y_c + h / 2
        return x1, y1, x2, y2

    x1_1, y1_1, x2_1, y2_1 = yolo_to_xyxy(*box1)
    x1_2, y1_2, x2_2, y2_2 = yolo_to_xyxy(*box2)

    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)

    if inter_x2 < inter_x1 or inter_y2 < inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def test_dataset(split="all"):
    """Evaluates the dataset across specified data splits."""
    if split == "all":
        img_dirs = [DATA_DIR / "train", DATA_DIR / "val"]
    elif split == "train":
        img_dirs = [DATA_DIR / "train"]
    else:
        img_dirs = [DATA_DIR / "val"]

    # Load Model
    print(f"Loading model from: {MODEL_PATH}")
    model_short = get_short_path(str(MODEL_PATH))
    model = YOLO(model_short)

    # Collect all images matching criteria
    all_imgs = []
    for img_dir in img_dirs:
        if img_dir.exists():
            for f in os.listdir(str(img_dir)):
                if f.endswith(".jpg"):
                    all_imgs.append(img_dir / f)

    print(f"Dataset Evaluation Scale: {len(all_imgs)} images found.")

    # Statistical Evaluation Indicators
    stats = {
        "total": len(all_imgs),
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "class_stats": defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0}),
        "per_image": []
    }

    # Frame-by-frame Inference Loops
    for idx, img_path in enumerate(all_imgs):
        if (idx + 1) % 100 == 0:
            print(f"  Progress: {idx + 1}/{len(all_imgs)}")

        # Determine structural data split context
        split = "train" if "train" in str(img_path) else "val"

        # Load Ground Truth Metrics
        gt_labels = load_labels(img_path, split)

        # Execute Model Prediction
        img_short = get_short_path(str(img_path))
        results = model.predict(source=img_short, conf=0.5, verbose=False)
        pred_boxes = []
        if results and len(results) > 0:
            for det in results[0].boxes:
                x_c = (det.xyxy[0][0] + det.xyxy[0][2]) / 2 / results[0].orig_shape[1]
                y_c = (det.xyxy[0][1] + det.xyxy[0][3]) / 2 / results[0].orig_shape[0]
                w = (det.xyxy[0][2] - det.xyxy[0][0]) / results[0].orig_shape[1]
                h = (det.xyxy[0][3] - det.xyxy[0][1]) / results[0].orig_shape[0]
                pred_boxes.append({
                    "class": int(det.cls[0]),
                    "conf": float(det.conf[0]),
                    "bbox": (x_c, y_c, w, h)
                })

        # Bounding Box Greedy Matching System (GT <-> Predictions)
        matched_gt = set()
        matched_pred = set()

        for pred_idx, pred in enumerate(pred_boxes):
            best_iou = 0.5
            best_gt_idx = -1
            for gt_idx, gt in enumerate(gt_labels):
                if gt_idx in matched_gt:
                    continue
                if pred["class"] != gt["class"]:
                    continue
                box_iou = iou(pred["bbox"], gt["bbox"])
                if box_iou > best_iou:
                    best_iou = box_iou
                    best_gt_idx = gt_idx

            if best_gt_idx >= 0:
                matched_gt.add(best_gt_idx)
                matched_pred.add(pred_idx)
                stats["tp"] += 1
                stats["class_stats"][pred["class"]]["tp"] += 1
            else:
                stats["fp"] += 1
                stats["class_stats"][pred["class"]]["fp"] += 1

        # Calculate False Negatives (FN)
        for gt_idx in range(len(gt_labels)):
            if gt_idx not in matched_gt:
                stats["fn"] += 1
                stats["class_stats"][gt_labels[gt_idx]["class"]]["fn"] += 1

        # Record standalone isolated frame metadata metrics
        stats["per_image"].append({
            "image": str(img_path.relative_to(PROJECT_DIR)),
            "gt_count": len(gt_labels),
            "pred_count": len(pred_boxes),
            "tp": len(matched_pred),
            "fp": len(pred_boxes) - len(matched_pred),
            "fn": len(gt_labels) - len(matched_gt)
        })

    # Metric Evaluations
    precision = stats["tp"] / (stats["tp"] + stats["fp"]) if (stats["tp"] + stats["fp"]) > 0 else 0
    recall = stats["tp"] / (stats["tp"] + stats["fn"]) if (stats["tp"] + stats["fn"]) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Build Verification Report Object
    report = {
        "split": split,
        "total_images": stats["total"],
        "total_tp": stats["tp"],
        "total_fp": stats["fp"],
        "total_fn": stats["fn"],
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "class_metrics": {}
    }

    for cls_id, cls_name in CLASS_NAMES.items():
        cls_stat = stats["class_stats"][cls_id]
        cls_p = cls_stat["tp"] / (cls_stat["tp"] + cls_stat["fp"]) if (cls_stat["tp"] + cls_stat["fp"]) > 0 else 0
        cls_r = cls_stat["tp"] / (cls_stat["tp"] + cls_stat["fn"]) if (cls_stat["tp"] + cls_stat["fn"]) > 0 else 0
        cls_f1 = 2 * cls_p * cls_r / (cls_p + cls_r) if (cls_p + cls_r) > 0 else 0

        report["class_metrics"][cls_name] = {
            "tp": cls_stat["tp"],
            "fp": cls_stat["fp"],
            "fn": cls_stat["fn"],
            "precision": round(cls_p, 4),
            "recall": round(cls_r, 4),
            "f1": round(cls_f1, 4)
        }

    return report, stats["per_image"]


if __name__ == "__main__":
    print("Beginning Person Detection Model Evaluation Pipeline...\n")

    # Run comprehensive evaluations on all available dataset sources
    report, per_image = test_dataset("all")

    # Output structural evaluation data configurations
    report_path = RESULTS_DIR / "test_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Save tracking history configuration files per single frame analysis
    per_image_path = RESULTS_DIR / "per_image_results.json"
    with open(per_image_path, "w", encoding="utf-8") as f:
        json.dump(per_image, f, indent=2, ensure_ascii=False)

    # Display execution result summary logs
    print("\n" + "="*60)
    print("EVALUATION RESULT SUMMARY")
    print("="*60)
    print(f"Total Images Evaluated: {report['total_images']}")
    print(f"TP: {report['total_tp']} | FP: {report['total_fp']} | FN: {report['total_fn']}")
    print(f"Global Precision: {report['precision']:.4f}")
    print(f"Global Recall   : {report['recall']:.4f}")
    print(f"Global F1-Score : {report['f1']:.4f}")
    print("\nMetrics Categorized by Sub-classes:")
    for cls_name, metrics in report["class_metrics"].items():
        print(f"  {cls_name}:")
        print(f"    P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")

    print(f"\nEvaluation data profile exported to: {report_path}")
    print(f"Individual frame analytical breakdown saved to: {per_image_path}")
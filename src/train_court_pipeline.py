"""train_court_pipeline.py — Tennis Court Keypoints Model Training Entry Point

Function: Prepares the dataset YAML configuration, starts YOLO fine-tuning training, and exports bad cases for iterative optimization.
"""
import os
import glob
import cv2
import numpy as np
from ultralytics import YOLO

# =====================================================================
# Core Path Configurations
# =====================================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')
PROJECT_DIR = os.path.dirname(CURRENT_DIR).replace('\\', '/')
DATASET_DIR = f"{PROJECT_DIR}/data/court_finetune"
BAD_CASE_DIR = f"{PROJECT_DIR}/data/court_finetune/bad_cases"
RUNS_DIR = f"{PROJECT_DIR}/runs/court_finetune"

COURT_14_PTS_PHYSICAL = np.array([
    [-5.485, -11.885], [5.485, -11.885], [5.485, 11.885], [-5.485, 11.885],
    [0.000, -11.885], [0.000, 11.885],
    [-4.115, -6.400], [4.115, -6.400], [0.000, -6.400],
    [-4.115, 6.400], [4.115, 6.400], [0.000, 6.400],
    [-5.485, 0.000], [5.485, 0.000]
], dtype=np.float32)


def prepare_env():
    caches = glob.glob(f"{DATASET_DIR}/**/*.cache", recursive=True)
    for c in caches:
        try:
            os.remove(c)
            print(f"Cleared old cache file: {c}")
        except:
            pass

    yaml_path = f"{PROJECT_DIR}/configs/court_keypoints_ultimate.yaml"
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(f"path: {DATASET_DIR}\n")
        f.write("train: train/images\nval: val/images\nnames:\n  0: tennis_court\n")
        f.write("kpt_shape: [14, 3]\n")

        # 7:3 Weighting between far-end and near-end keypoints (OKS sigma, smaller values yield higher weights)
        # Assuming points 0 and 1 belong to the far end (top of the frame), points 2 and 3 belong to the near end (bottom of the frame)
        # Far-end sigma=0.0065 (extremely high weight), near-end sigma=0.010 (high weight), remaining points 0.050 (low weight)
        f.write(
            "sigmas: [0.0065, 0.0065, 0.010, 0.010, 0.050, 0.050, 0.050, 0.050, 0.050, 0.050, 0.050, 0.050, 0.050, 0.050]\n")
    return yaml_path


def train_model(yaml_path):
    print("\nStarting court keypoint model training...")

    previous_best = f"{RUNS_DIR}/court_14pts_weighted/weights/best.pt"
    if os.path.exists(previous_best):
        print(f"Inheriting baseline model: {previous_best}")
        model = YOLO(previous_best)
    else:
        fallback = f"{PROJECT_DIR}/models/yolo/yolov8n-pose.pt"
        print(f"[!] Baseline model not found, using fallback: {fallback}")
        model = YOLO(fallback)

    os.makedirs(RUNS_DIR, exist_ok=True)

    model.train(
        data=yaml_path,
        epochs=300,
        imgsz=960,  # Large image input to improve localization precision for minor keypoints
        batch=8,  # A batch size of 8 perfectly fits a 16GB VRAM constraint, providing smoother gradients
        workers=4,  # Increase CPU data loading threads
        cache=False,
        device='cuda:0',
        project=RUNS_DIR,
        name='court_14pts_ultimate',
        exist_ok=True,
        patience=50,
        close_mosaic=280  # Enable Mosaic augmentation for the first 20 epochs, then turn off
    )
    return f"{RUNS_DIR}/court_14pts_ultimate/weights/best.pt"


def export_bad_cases(best_weight_path):
    print(f"\nScanning validation set using weight path: {best_weight_path}")
    os.makedirs(BAD_CASE_DIR, exist_ok=True)

    if not os.path.exists(best_weight_path):
        return

    model = YOLO(best_weight_path)
    val_images = glob.glob(f"{DATASET_DIR}/val/images/*.jpg")
    bad_count = 0

    for img_path in val_images:
        base_name = os.path.basename(img_path)
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), -1)
        if img is None: continue

        results = model.predict(img, conf=0.6, imgsz=960, verbose=False)[0]
        failed, fail_reason = False, ""
        pred_pixel_pts, pred_phys_pts = [], []

        if results.boxes is None or len(results.boxes) == 0:
            failed, fail_reason = True, "No Court Detected"
        else:
            kpts = results.keypoints.data[0].cpu().numpy()
            for i, pt in enumerate(kpts):
                x, y, conf = pt
                if conf > 0.6:
                    pred_pixel_pts.append([x, y])
                    pred_phys_pts.append(COURT_14_PTS_PHYSICAL[i])

            if len(pred_pixel_pts) < 4:
                failed, fail_reason = True, f"Missed Points ({len(pred_pixel_pts)}/14)"
            else:
                H, _ = cv2.findHomography(np.array(pred_phys_pts, dtype=np.float32),
                                          np.array(pred_pixel_pts, dtype=np.float32),
                                          cv2.RANSAC, 5.0)
                if H is None: failed, fail_reason = True, "Matrix Calculation Failed"

        if failed:
            bad_count += 1
            for pt in pred_pixel_pts:
                cv2.circle(img, (int(pt[0]), int(pt[1])), 4, (0, 0, 255), -1)
            cv2.putText(img, f"FAIL: {fail_reason}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            save_path = f"{BAD_CASE_DIR}/Bad_{base_name}"
            cv2.imencode('.jpg', img)[1].tofile(save_path)

    print(f"Diagnostic scan complete! Found {bad_count} Bad Cases, exported to: {BAD_CASE_DIR}")


if __name__ == "__main__":
    yaml_path = prepare_env()
    best_weights = train_model(yaml_path)
    export_bad_cases(best_weights)
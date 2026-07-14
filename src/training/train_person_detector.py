"""
train_person_detector.py — Player Detection Model Training Entry
Function: Fine-tune the YOLO player detection model based on data/person_sorter/
Weight Output: runs/person_training/<run_name>/weights/best.pt

How to start (Execute from the project root directory Project_Annotation_and_Testing/):
    python src/training/train_person_detector.py
"""
import os
import shutil
from pathlib import Path
from ultralytics import YOLO

# ── Path Configuration ──────────────────────────────────────────────────
CURRENT_DIR = Path(__file__).parent
PROJECT_DIR = CURRENT_DIR.parent.parent          # Project_Annotation_and_Testing/
DATASET_YAML = PROJECT_DIR / "configs" / "person_sorter_dataset.yaml"
PRETRAIN_WEIGHTS = PROJECT_DIR / "models" / "person" / "best.pt"
RUNS_DIR = PROJECT_DIR / "runs" / "person_training"
RUN_NAME = "hard_neg_finetune_v1"


def prepare_dataset_yaml():
    """Generate absolute path version of dataset.yaml to prevent YOLO path parsing issues"""
    data_root = (PROJECT_DIR / "data" / "person_sorter").as_posix()
    content = f"""# person_sorter_dataset.yaml — Automatically generated, do not modify paths manually
path: {data_root}
train: images/train
val: images/val

names:
  0: player_near
  1: player_far
"""
    DATASET_YAML.write_text(content, encoding="utf-8")
    print(f"dataset.yaml has been updated: {DATASET_YAML}")
    return str(DATASET_YAML)


def check_data():
    """Verify dataset is ready before training"""
    train_imgs = PROJECT_DIR / "data" / "person_sorter" / "images" / "train"
    val_imgs = PROJECT_DIR / "data" / "person_sorter" / "images" / "val"
    n_train = len([f for f in os.listdir(train_imgs) if f.endswith(".jpg")])
    n_val = len([f for f in os.listdir(val_imgs) if f.endswith(".jpg")])
    print(f"Dataset: train={n_train}, val={n_val}")
    if n_train < 10:
        raise RuntimeError("Insufficient training samples, please run merge_hard_negatives.py first")
    return n_train, n_val


def train():
    yaml_path = prepare_dataset_yaml()
    n_train, n_val = check_data()

    print(f"\nStarting training — Weights will be saved to: {RUNS_DIR / RUN_NAME}")
    print(f"   Base weights: {PRETRAIN_WEIGHTS}")

    model = YOLO(str(PRETRAIN_WEIGHTS))

    model.train(
        data=yaml_path,
        epochs=100,
        imgsz=640,
        batch=4,
        device=0,

        # ── Performance Optimization ────────────────────────────
        cache=True,
        amp=True,
        workers=4,

        # ── Output Path Control (Crucial: prevents scattered weights) ──
        project=str(RUNS_DIR),
        name=RUN_NAME,
        exist_ok=False,          # Throws error if run name already exists to avoid overwriting

        # ── Training Strategy ───────────────────────────────────
        patience=20,
        save_period=10,
        optimizer="AdamW",
        lr0=1e-4,                # Small learning rate for fine-tuning
        warmup_epochs=3,
    )

    # After training, copy the optimal weights to models/person/
    best_src = RUNS_DIR / RUN_NAME / "weights" / "best.pt"
    best_dst = PROJECT_DIR / "models" / "person" / f"best_{RUN_NAME}.pt"
    if best_src.exists():
        shutil.copy2(best_src, best_dst)
        print(f"\nOptimal weights copied to: {best_dst}")
        print(f"   To replace production weights, manually execute:")
        print(f"   copy {best_dst} {PROJECT_DIR / 'models' / 'person' / 'best.pt'}")
    else:
        print(f" best.pt not found, please check if the training completed successfully")


if __name__ == "__main__":
    train()
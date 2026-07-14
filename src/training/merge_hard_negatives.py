"""
merge_hard_negatives.py — Hard Negatives Dataset Merging Script
Function: Merge annotated samples from hard_negatives into train/val sets at an 8:2 ratio.
"""
import os
import shutil
from pathlib import Path
import random

# Path Configuration
CURRENT_DIR = Path(__file__).parent
PROJECT_DIR = CURRENT_DIR.parent.parent
PERSON_SORTER_DIR = PROJECT_DIR / "data" / "person_sorter"

HARD_NEG_IMAGES = PERSON_SORTER_DIR / "hard_negatives" / "images"
HARD_NEG_LABELS = PERSON_SORTER_DIR / "hard_negatives" / "labels"

TRAIN_IMAGES = PERSON_SORTER_DIR / "images" / "train"
TRAIN_LABELS = PERSON_SORTER_DIR / "labels" / "train"
VAL_IMAGES = PERSON_SORTER_DIR / "images" / "val"
VAL_LABELS = PERSON_SORTER_DIR / "labels" / "val"


def main():
    print("Scanning hard_negatives directory...")

    # Get all annotated images (those with corresponding label files)
    all_images = []
    for img_name in os.listdir(HARD_NEG_IMAGES):
        if not img_name.lower().endswith(".jpg"):
            continue
        label_name = img_name.rsplit(".", 1)[0] + ".txt"
        label_path = HARD_NEG_LABELS / label_name
        if label_path.exists():
            all_images.append(img_name)

    print(f"Found {len(all_images)} annotated samples")

    if len(all_images) == 0:
        print("No annotated samples found. Please run hard_negative_reviewer.py first.")
        return

    # 8:2 Split
    random.seed(42)
    random.shuffle(all_images)
    split_idx = int(len(all_images) * 0.8)
    train_list = all_images[:split_idx]
    val_list = all_images[split_idx:]

    print(f"Split ratio: train={len(train_list)}, val={len(val_list)}")

    # Copy to train
    print("\nCopying to train set...")
    for img_name in train_list:
        label_name = img_name.rsplit(".", 1)[0] + ".txt"
        shutil.copy2(HARD_NEG_IMAGES / img_name, TRAIN_IMAGES / img_name)
        shutil.copy2(HARD_NEG_LABELS / label_name, TRAIN_LABELS / label_name)

    # Copy to val
    print("Copying to val set...")
    for img_name in val_list:
        label_name = img_name.rsplit(".", 1)[0] + ".txt"
        shutil.copy2(HARD_NEG_IMAGES / img_name, VAL_IMAGES / img_name)
        shutil.copy2(HARD_NEG_LABELS / label_name, VAL_LABELS / label_name)

    # Count final quantities
    final_train = len(os.listdir(TRAIN_IMAGES))
    final_val = len(os.listdir(VAL_IMAGES))

    print(f"\nMerge complete!")
    print(f"   Train: {final_train} images")
    print(f"   Val: {final_val} images")
    print(f"\nNext step: Run python src/training/train_person_detector.py")


if __name__ == "__main__":
    main()
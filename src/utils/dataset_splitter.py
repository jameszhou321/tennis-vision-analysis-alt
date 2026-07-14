"""
dataset_splitter.py — Dataset Train/Val Split Tool

Function: Randomly splits paired images and label files inside data/person_sorter/ 
          into training and validation subdirectories according to a target ratio, 
          formatting them to adhere to YOLO structure requirements.
"""
import os
import random
import shutil
from pathlib import Path


def split_dataset(data_dir, train_ratio=0.8):
    base_path = Path(data_dir)

    # Original resource source paths
    src_images = base_path / "image"
    src_labels = base_path / "labels"

    # YOLO standard target directory structure mapping
    # Note: YOLO standard convention typically expects 'images' (plural) rather than 'image'
    train_images_dir = base_path / "images" / "train"
    val_images_dir = base_path / "images" / "val"
    train_labels_dir = base_path / "labels" / "train"
    val_labels_dir = base_path / "labels" / "val"

    # Ensure all distinct structural target directories are created
    for dir_path in [train_images_dir, val_images_dir, train_labels_dir, val_labels_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    print("Scanning annotated image-label pairs...")

    # Discover candidate image assets
    valid_exts = {'.jpg', '.jpeg', '.png'}
    all_images = [f for f in src_images.iterdir() if f.suffix.lower() in valid_exts]

    # Filter out clean pairs (drop unannotated frames missing a matching text annotation)
    valid_data_pairs = []
    for img_path in all_images:
        label_name = img_path.stem + ".txt"
        label_path = src_labels / label_name

        if label_path.exists():
            valid_data_pairs.append((img_path, label_path))

    total_valid = len(valid_data_pairs)
    if total_valid == 0:
        print("No paired image and label files found. Please verify directory paths.")
        return

    print(f"Discovered {total_valid} valid annotated image-label asset pairs.")

    # Random permutation alignment
    random.seed(42)  # Fixed seed initialization enforces split reproducibility
    random.shuffle(valid_data_pairs)

    # Determine subset boundary indexes
    split_index = int(total_valid * train_ratio)
    train_data = valid_data_pairs[:split_index]
    val_data = valid_data_pairs[split_index:]

    print(f"Distributing pairs (Train subset: {len(train_data)}, Validation subset: {len(val_data)})...")

    # Internal batch deployment copy helper
    def copy_data(data_list, target_img_dir, target_label_dir):
        for img_src, label_src in data_list:
            shutil.copy2(img_src, target_img_dir / img_src.name)
            shutil.copy2(label_src, target_label_dir / label_src.name)

    # Execute system distribution
    copy_data(train_data, train_images_dir, train_labels_dir)
    copy_data(val_data, val_images_dir, val_labels_dir)

    print("-" * 30)
    print("Dataset distribution complete! Directory structure matches YOLO interface specifications.")


if __name__ == "__main__":
    # Pointing to the targeted localization data root
    DATA_DIR = "data/person_sorter"
    split_dataset(DATA_DIR, train_ratio=0.8)
"""prepare_weighted_dataset.py — Weighted Dataset Merging Tool

Function: Merge new annotated data with the old dataset, and split into train/val by ratio for court model fine-tuning.
"""
import os
import glob
import random
import shutil


def merge_and_split(source_img_dir, source_lbl_dir, target_base_dir, total_samples=1300, split_ratio=0.8):
    # Ensure target folders (old dataset directory) exist
    for split in ['train', 'val']:
        os.makedirs(os.path.join(target_base_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(target_base_dir, split, 'labels'), exist_ok=True)

    print("Scanning your 300 newly annotated data files...")
    img_files = sorted(glob.glob(os.path.join(source_img_dir, "*.jpg")))

    valid_pairs = []
    for img_path in img_files:
        lbl_path = os.path.join(source_lbl_dir, os.path.splitext(os.path.basename(img_path))[0] + ".txt")
        if os.path.exists(lbl_path):
            valid_pairs.append((img_path, lbl_path))

    if len(valid_pairs) < total_samples:
        print(f"Notice: Only found {len(valid_pairs)} valid images. Will merge all of them.")
        total_samples = len(valid_pairs)

    # Intercept the top 300 images you want
    selected_pairs = valid_pairs[:total_samples]

    # Shuffle randomly to ensure even sample distribution between training and validation sets
    random.seed(42)
    random.shuffle(selected_pairs)

    # 8:2 split
    train_count = int(total_samples * split_ratio)
    train_pairs = selected_pairs[:train_count]
    val_pairs = selected_pairs[train_count:]

    print(f"Preparing to merge into the original dataset: Added {len(train_pairs)} training images | Added {len(val_pairs)} validation images")

    # Perform copy and merge operation
    def append_to_dataset(pairs, split_name):
        added_count = 0
        for img_src, lbl_src in pairs:
            base_name = os.path.basename(img_src)
            img_dst = os.path.join(target_base_dir, split_name, 'images', base_name)
            lbl_dst = os.path.join(target_base_dir, split_name, 'labels', os.path.basename(lbl_src))

            # If a file with the same name happens to exist, automatically append a suffix to prevent overwriting old data
            if os.path.exists(img_dst):
                name, ext = os.path.splitext(base_name)
                new_base = f"{name}_v2{ext}"
                img_dst = os.path.join(target_base_dir, split_name, 'images', new_base)
                lbl_dst = os.path.join(target_base_dir, split_name, 'labels', f"{name}_v2.txt")

            shutil.copy(img_src, img_dst)
            shutil.copy(lbl_src, lbl_dst)
            added_count += 1

        # Count the total number of images currently in this folder
        total_now = len(glob.glob(os.path.join(target_base_dir, split_name, 'images', '*.jpg')))
        print(f"  -> Successfully mixed {added_count} images into the {split_name} folder! (Current total {split_name} data: {total_now} images)")

    append_to_dataset(train_pairs, 'train')
    append_to_dataset(val_pairs, 'val')

    print(f"\nDataset expansion complete! Your main dataset has become stronger.")


if __name__ == "__main__":
    import os as _os
    _PROJECT_DIR = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    SOURCE_IMAGES = _os.path.join(_PROJECT_DIR, "_archive", "Second_Train_Dataset", "images")
    SOURCE_LABELS = _os.path.join(_PROJECT_DIR, "_archive", "Second_Train_Dataset", "labels")
    TARGET_DATASET = _os.path.join(_PROJECT_DIR, "data", "court_finetune")
    merge_and_split(SOURCE_IMAGES, SOURCE_LABELS, TARGET_DATASET, total_samples=1300)
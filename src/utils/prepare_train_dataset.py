"""
Copy annotation data from data/rallies_annotated/ to data/rallies_train/.
Only copy the 3 files required for training, skipping annotated_clip.mp4.
Supports resuming from breakpoints.
"""
import os
import shutil

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(_PROJECT_DIR, "data", "rallies_annotated")
DST_DIR = os.path.join(_PROJECT_DIR, "data", "rallies_train")

REQUIRED_FILES = ["raw_clip.mp4", "pose_data.json", "annotations.json"]


def is_complete(dst_rally_dir):
    return all(os.path.exists(os.path.join(dst_rally_dir, f)) for f in REQUIRED_FILES)


def main():
    os.makedirs(DST_DIR, exist_ok=True)

    rally_dirs = sorted(
        d for d in os.listdir(SRC_DIR)
        if os.path.isdir(os.path.join(SRC_DIR, d))
    )

    copied = skipped = missing = 0

    for rally in rally_dirs:
        src_rally = os.path.join(SRC_DIR, rally)
        dst_rally = os.path.join(DST_DIR, rally)

        if is_complete(dst_rally):
            skipped += 1
            continue

        # Check if the source directory contains all required files
        if not all(os.path.exists(os.path.join(src_rally, f)) for f in REQUIRED_FILES):
            print(f"  [SKIP] {rally} — Source directory is missing required files")
            missing += 1
            continue

        os.makedirs(dst_rally, exist_ok=True)
        for fname in REQUIRED_FILES:
            shutil.copy2(os.path.join(src_rally, fname), os.path.join(dst_rally, fname))

        print(f"  [COPY] {rally}")
        copied += 1

    print(f"\nCompleted: Copied {copied}, Skipped {skipped} (already exists), Missing {missing}")
    print(f"Destination directory: {DST_DIR}")


if __name__ == "__main__":
    main()
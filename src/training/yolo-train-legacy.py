"""yolo-train.py — YOLO Person Classification Model Training Script

Function: Fine-tune the YOLO model for player classification based on the data/person_sorter/ dataset.
"""
from ultralytics import YOLO


def main():
    # Load pre-trained model
    model = YOLO("yolo26x.pt")

    # Start highly optimized training
    results = model.train(
        data="data/dataset.yaml",
        epochs=150,
        imgsz=640,  # Tailored for small targets; do not reduce the resolution arbitrarily
        batch=4,  # Adjust based on your VRAM size (e.g., try 8 or 16 on a 3090/4090)
        device=0,

        # --- Core Parameters for Speedup and Memory/VRAM Optimization ---
        cache=True,  # [CORE] Pre-load images into RAM to completely eliminate disk I/O bottleneck
        amp=True,  # [CORE] Enable FP16 automatic mixed precision: halves VRAM usage and doubles speed
        workers=8,  # [CORE] Number of DataLoader threads; 4 or 8 is recommended for Windows
        # accumulate=4,     # [ALTERNATIVE] If VRAM still overflows, uncomment this and use alongside batch=2

        optimizer="MuSGD",  # New feature optimizer in YOLO26
        project="tennis_tracking",
        name="yolo26x_optimized_run",

        # Other practical parameters
        patience=30,  # Early stopping mechanism: terminates training early if accuracy doesn't improve for 30 epochs
        save_period=10  # Backup weights every 10 epochs to prevent data loss from unexpected crashes
    )


if __name__ == "__main__":
    main()
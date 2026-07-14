"""test_dataset.py — Dataset loading validation script"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from dataset import TennisActionDataset
from config import load_config


def test_dataset_pipeline(yaml_path=None):
    cfg = load_config(yaml_path)
    seq_len = cfg["seq_len"]

    print("=" * 50)
    print(f"Data Root Directory: {cfg['data_root']}")

    try:
        print("⏳ Initializing TennisActionDataset...")
        dataset = TennisActionDataset(cfg)
        print(f"Initialization successful. Total slices: {len(dataset)}.")

        if len(dataset) == 0:
            print("No data found. Please check the data_root path.")
            return

        print(f"\n⏳ Reading the first sample (idx=0)...")
        pose, packed, labels = dataset[0]

        print("\nSample loaded successfully. Tensor shapes:")
        print("-" * 50)
        print(f"  pose:   {tuple(pose.shape)}   Expected: ({seq_len}, 97)")
        print(f"  packed: {tuple(packed.shape)}  Expected: ({seq_len}, 3, 320, 960)")
        print(f"  labels: {tuple(labels.shape)}  Expected: ({seq_len},)")
        print("-" * 50)
        print(f"Labels (First 30 frames): {labels[:30].tolist()}")
        print("=" * 50)
        print("Test passed!")

    except Exception:
        import traceback
        print("\nDataset threw an exception:")
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    test_dataset_pipeline(args.config)
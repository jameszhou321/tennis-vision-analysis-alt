"""run_ablation.py — Sequentially trains all experimental configurations (excluding main)

Usage:
python src/model/mst/run_ablation.py              # Train all configurations
python src/model/mst/run_ablation.py --group ablation
python src/model/mst/run_ablation.py --dry-run
python src/model/mst/run_ablation.py --smoke
"""
import os
import sys
import subprocess
import argparse
import time

_mst_dir = os.path.dirname(os.path.abspath(__file__))
_project_dir = os.path.dirname(os.path.dirname(os.path.dirname(_mst_dir)))

# (group, config_relative_path)
CONFIGS = [
    # --- Hyperparameters Comparison ---
    ("hyperparams", "configs/hyperparams/hp_embed96.yaml"),
    ("hyperparams", "configs/hyperparams/hp_embed256.yaml"),
    ("hyperparams", "configs/hyperparams/hp_depth4.yaml"),
    ("hyperparams", "configs/hyperparams/hp_depth12.yaml"),
    ("hyperparams", "configs/hyperparams/hp_vtokens8.yaml"),
    ("hyperparams", "configs/hyperparams/hp_vtokens32.yaml"),
    # --- Ablation Studies ---
    ("ablation",    "configs/ablation/abl_no_pose.yaml"),
    ("ablation",    "configs/ablation/abl_no_crops.yaml"),
    ("ablation",    "configs/ablation/abl_no_visual.yaml"),
    ("ablation",    "configs/ablation/abl_global_only.yaml"),
    # --- Components Comparison ---
    ("components",  "configs/components/cmp_focal_loss.yaml"),
    ("components",  "configs/components/cmp_ce_loss.yaml"),
    ("components",  "configs/components/cmp_no_merge.yaml"),
    ("components",  "configs/components/cmp_resnet_backbone.yaml"),
    ("components",  "configs/components/cmp_frozen_backbone.yaml"),
]

TRAIN_SCRIPT = os.path.join(_mst_dir, "train.py")
PYTHON = sys.executable


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Only print commands without executing them")
    parser.add_argument("--smoke",   action="store_true", help="Run only 1 slice and 1 epoch per config to validate pipeline")
    parser.add_argument("--group",   default=None,
                        choices=["hyperparams", "ablation", "components"],
                        help="Train only the specified group")
    args = parser.parse_args()

    selected = [(g, c) for g, c in CONFIGS if args.group is None or g == args.group]

    print(f"Total {len(selected)} configurations scheduled for training" + (f" (Group: {args.group})" if args.group else ""))

    results = []
    for idx, (group, cfg_rel) in enumerate(selected, 1):
        cfg_path = os.path.join(_project_dir, cfg_rel)
        name = os.path.splitext(os.path.basename(cfg_path))[0]
        cmd = [PYTHON, TRAIN_SCRIPT, "--config", cfg_path]
        if args.smoke:
            cmd.append("--smoke")

        print(f"\n{'='*60}")
        print(f"▶ [{idx}/{len(selected)}] {group}/{name}")
        print(f"  Command: {' '.join(cmd)}")
        print(f"{'='*60}")

        if args.dry_run:
            results.append((group, name, "dry-run", 0))
            continue

        t0 = time.time()
        ret = subprocess.run(cmd, cwd=_project_dir)
        elapsed = time.time() - t0

        status = "Success" if ret.returncode == 0 else f"Failed(code={ret.returncode})"
        results.append((group, name, status, elapsed))
        print(f"\n{status}  Time elapsed: {elapsed/60:.1f} minutes")

    print(f"\n{'='*60}")
    print("Training Summary:")
    for group, name, status, elapsed in results:
        t = f"{elapsed/60:.1f} min" if elapsed else ""
        print(f"  {status}  [{group}] {name}  {t}")


if __name__ == "__main__":
    main()
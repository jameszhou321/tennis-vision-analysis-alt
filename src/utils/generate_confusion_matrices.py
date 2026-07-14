"""
For each fully trained model, run sample-by-sample inference on the test set
to generate an exact confusion matrix.
Outputs two versions: count (raw counts) and percentage.

Usage:
  cd project_annotation_and_testing
  .venv/Scripts/python src/utils/generate_confusion_matrices.py
"""

import os, sys, json, random, csv
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Try to use a Chinese-capable font (kept in case any Chinese text is rendered)
_CHINESE_FONT = None
for font_name in ['Microsoft YaHei', 'SimHei', 'SimSun', 'Arial Unicode MS', 'Noto Sans CJK SC']:
    try:
        _CHINESE_FONT = fm.findfont(font_name, fallback_to_default=False)
        if _CHINESE_FONT:
            break
    except:
        continue

if _CHINESE_FONT:
    plt.rcParams['font.family'] = fm.FontProperties(fname=_CHINESE_FONT).get_name()
    plt.rcParams['axes.unicode_minus'] = False

from torch.utils.data import DataLoader
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../model/mst"))
from dataset import TennisActionDataset
from model_main import MSTFormer
from config import load_config

ACTION_NAMES = ["idle", "forehand", "backhand", "serve", "move"]
ACTION_NAMES_CN = ["待机", "正手", "反手", "发球", "移动"]

def split_dataset(data_root, train_ratio=0.8, seed=42):
    """Split logic identical to train.py"""
    import cv2
    random.seed(seed)
    clips = []
    total_frames = 0
    for d in os.listdir(data_root):
        clip_path = os.path.join(data_root, d)
        if not os.path.isdir(clip_path):
            continue
        video = os.path.join(clip_path, "raw_clip.mp4")
        anno = os.path.join(clip_path, "annotations.json")
        if not os.path.exists(video) or not os.path.exists(anno):
            continue
        cap = cv2.VideoCapture(video)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if frames > 0:
            clips.append({"path": clip_path, "frames": frames})
            total_frames += frames

    random.shuffle(clips)
    train_dirs, test_dirs = [], []
    target = total_frames * train_ratio
    current = 0
    for c in clips:
        if current < target:
            train_dirs.append(c["path"])
            current += c["frames"]
        else:
            test_dirs.append(c["path"])
    return train_dirs, test_dirs


def find_model_dirs(models_root):
    """Scan all fully trained models (i.e. those with best.pth and a valid training log)"""
    models = []
    for config_name in sorted(os.listdir(models_root)):
        config_dir = os.path.join(models_root, config_name)
        if not os.path.isdir(config_dir):
            continue
        for ts_name in sorted(os.listdir(config_dir)):
            ts_dir = os.path.join(config_dir, ts_name)
            best_path = os.path.join(ts_dir, "best.pth")
            config_path = os.path.join(ts_dir, "config.yaml")
            csv_path = os.path.join(ts_dir, "train_log.csv")
            if os.path.exists(best_path) and os.path.exists(config_path):
                models.append({
                    "config_name": config_name,
                    "ts_name": ts_name,
                    "ts_dir": ts_dir,
                    "best_path": best_path,
                    "config_path": config_path,
                    "csv_path": csv_path,
                })
    return models


def compute_cm(preds, labels, num_classes=5):
    """Compute the exact confusion matrix"""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for p, l in zip(preds, labels):
        if l >= 0 and l < num_classes:
            cm[l, p] += 1
    return cm


def plot_cm(cm, title, save_path, fmt="d", vmax=None):
    """Plot the confusion matrix"""
    num_classes = cm.shape[0]
    if vmax is None:
        vmax = cm.max()

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    im = ax.imshow(cm, cmap='Blues', interpolation='nearest', vmin=0, vmax=vmax)

    # Color bar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(ACTION_NAMES, fontsize=10)
    ax.set_yticklabels(ACTION_NAMES, fontsize=10)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Ground Truth", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight='bold')

    # Annotate values inside each cell
    for i in range(num_classes):
        for j in range(num_classes):
            val = cm[i, j]
            color = 'white' if val > vmax * 0.6 else 'black'
            if fmt == "d":
                text = str(int(val))
            else:
                text = f"{val:.1f}"
            ax.text(j, i, text, ha='center', va='center', fontsize=9, color=color)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def run_inference(model, test_loader, device, num_classes=5, keyframe_only=False):
    """Run inference, collecting all predictions and labels"""
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for pose, packed, labels, kf_labels in test_loader:
            pose   = pose.to(device, non_blocking=True)
            packed = packed.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.amp.autocast("cuda"):
                if keyframe_only:
                    kf_logits = model(pose, packed)
                    continue  # keyframe_only does not involve action classification
                else:
                    action_logits, kf_logits = model(pose, packed)

            preds = action_logits.argmax(-1)
            mask = labels != -100
            all_preds.append(preds[mask].cpu())
            all_labels.append(labels[mask].cpu())

    if not all_preds:
        return None, None
    return torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()


def generate_report(models_root, report_dir):
    """Main pipeline"""
    os.makedirs(report_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    models = find_model_dirs(models_root)
    print(f"Found {len(models)} trained models")

    # Filter out the smoke test and keyframe_only runs
    models = [m for m in models if not (m["config_name"] == "main" and m["ts_name"] == "20260424_194625")]
    # Exclude keyframe_only configs
    # (there is no keyframe_only config, so all are skipped)

    results = []

    for m in models:
        print(f"\n{'='*60}")
        print(f"Model: {m['config_name']}/{m['ts_name']}")
        print(f"{'='*60}")

        # Load config
        cfg = load_config(m["config_path"])
        cfg["_yaml_path"] = m["config_path"]
        data_root = cfg.get("data_root", "data/rallies_train")
        train_ratio = cfg.get("train_ratio", 0.8)

        if cfg.get("keyframe_only", False):
            print("  Skipping keyframe_only model")
            results.append((m, None, None))
            continue

        # Reconstruct the data split
        train_dirs, test_dirs = split_dataset(data_root, train_ratio)
        print(f"  Train: {len(train_dirs)} clips, Test: {len(test_dirs)} clips")

        # Create the test dataset
        test_ds = TennisActionDataset(cfg, clip_dirs=test_dirs, augment=False)
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

        # Load the model
        model = MSTFormer(cfg).to(device)
        state = torch.load(m["best_path"], map_location=device, weights_only=True)
        model.load_state_dict(state, strict=False)
        print(f"  Loaded best.pth from {m['best_path']}")

        # Run inference
        preds, labels = run_inference(model, test_loader, device)

        if preds is None or len(preds) == 0:
            print("  No predictions collected")
            results.append((m, None, None))
            continue

        # Compute the confusion matrix
        cm = compute_cm(preds, labels)
        acc = np.trace(cm) / cm.sum() * 100

        # Per-class metrics
        per_class_recall = []
        per_class_precision = []
        for i in range(5):
            tp = cm[i, i]
            fn = cm[i, :].sum() - tp
            fp = cm[:, i].sum() - tp
            rec = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
            prec = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
            per_class_recall.append(rec)
            per_class_precision.append(prec)

        print(f"  Accuracy: {acc:.2f}%")
        print(f"  Per-class recall: {[f'{r:.1f}%' for r in per_class_recall]}")
        print(f"  Confusion matrix:\n{cm}")

        results.append((m, cm, {
            "acc": acc,
            "recall": per_class_recall,
            "precision": per_class_precision,
            "total": cm.sum(),
            "preds": preds,
            "labels": labels,
        }))

        # Plot the count-based confusion matrix
        safe_name = f"{m['config_name']}_{m['ts_name'][:8]}"
        vmax = cm.max()

        plot_cm(cm,
                f"{m['config_name']}\nAccuracy: {acc:.2f}% (n={cm.sum()})",
                os.path.join(report_dir, f"confusion_cnt_{safe_name}.png"),
                fmt="d", vmax=vmax)

        # Plot the percentage version (row-normalized)
        cm_pct = np.zeros_like(cm, dtype=float)
        for i in range(5):
            row_sum = cm[i, :].sum()
            if row_sum > 0:
                cm_pct[i] = cm[i] / row_sum * 100

        plot_cm(cm_pct,
                f"{m['config_name']} — Row %\nAccuracy: {acc:.2f}%",
                os.path.join(report_dir, f"confusion_pct_{safe_name}.png"),
                fmt=".1f", vmax=100)

        # Column-normalized (precision view)
        cm_col_pct = np.zeros_like(cm, dtype=float)
        for j in range(5):
            col_sum = cm[:, j].sum()
            if col_sum > 0:
                cm_col_pct[:, j] = cm[:, j] / col_sum * 100

        plot_cm(cm_col_pct,
                f"{m['config_name']} — Column %\nAccuracy: {acc:.2f}%",
                os.path.join(report_dir, f"confusion_colpct_{safe_name}.png"),
                fmt=".1f", vmax=100)

        # Generate the combined PDF-style figure (three panels in one)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

        for ax, cm_data, title, fmt, vm in [
            (axes[0], cm, "Count", "d", vmax),
            (axes[1], cm_pct, "Row % (Recall)", ".1f", 100),
            (axes[2], cm_col_pct, "Col % (Precision)", ".1f", 100),
        ]:
            im = ax.imshow(cm_data, cmap='Blues', interpolation='nearest',
                          vmin=0, vmax=vm if isinstance(vm, (int, float)) else cm_data.max())
            ax.set_xticks(range(5))
            ax.set_yticks(range(5))
            ax.set_xticklabels(ACTION_NAMES, fontsize=8)
            ax.set_yticklabels(ACTION_NAMES, fontsize=8)
            ax.set_xlabel("Predicted", fontsize=9)
            ax.set_ylabel("Ground Truth", fontsize=9)
            ax.set_title(title, fontsize=10, fontweight='bold')

            for i2 in range(5):
                for j2 in range(5):
                    val = cm_data[i2, j2]
                    vm2 = vm if isinstance(vm, (int, float)) else cm_data.max()
                    color = 'white' if val > vm2 * 0.6 else 'black'
                    if fmt == "d":
                        text = str(int(val))
                    else:
                        text = f"{val:.1f}"
                    ax.text(j2, i2, text, ha='center', va='center', fontsize=7, color=color)

        fig.suptitle(f"{m['config_name']} — Acc: {acc:.2f}%", fontsize=13, fontweight='bold')
        fig.tight_layout()
        fig.subplots_adjust(top=0.85)
        fig.savefig(os.path.join(report_dir, f"confusion_triple_{safe_name}.png"),
                   dpi=150, bbox_inches='tight')
        plt.close(fig)

    # Generate the summary table
    report_md = os.path.join(report_dir, "confusion_report.md")
    lines = ["# Exact Confusion Matrix Report (Sample-by-Sample Inference)",
             "",
             f"**Generated on:** 2026-04-28",
             f"**Method:** Load each model's best.pth, run sample-by-sample inference on the test set, and tabulate the confusion matrix",
             "",
             "## Summary Table",
             "",
             "| Model | Accuracy | Test Samples | Per-Class Recall (idle/FH/BH/serve/move) |",
             "|---|---:|---:|:---|",
             ]

    results.sort(key=lambda x: -(x[2]["acc"] if x[2] else 0))

    for m, cm, info in results:
        if info is None:
            continue
        rec_str = "/".join([f"{r:.1f}%" for r in info["recall"]])
        lines.append(f"| {m['config_name']}/{m['ts_name'][:8]} "
                     f"| {info['acc']:.2f}% | {info['total']} "
                     f"| {rec_str} |")

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- `confusion_cnt_*.png` = count-based confusion matrix")
    lines.append("- `confusion_pct_*.png` = row-normalized percentage (recall view)")
    lines.append("- `confusion_colpct_*.png` = column-normalized percentage (precision view)")
    lines.append("- `confusion_triple_*.png` = combined three-panel comparison figure")
    lines.append("")

    with open(report_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n{'='*60}")
    print(f"Done! All matrices saved to {report_dir}")
    print(f"Report: {report_md}")


if __name__ == "__main__":
    _project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    models_root = os.path.join(_project_root, "models", "action")
    report_dir = os.path.join(_project_root, "models", "report")
    generate_report(models_root, report_dir)
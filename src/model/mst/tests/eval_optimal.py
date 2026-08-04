"""Evaluate MSTFormer model and generate confusion matrices."""
import sys, os, random, argparse
import torch
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import TennisActionDataset
from model_main import MSTFormer
from config import load_config

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

_MST_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(_MST_DIR)))

CLASSES = ["Idle", "Forehand", "Backhand", "Serve", "Movement"]

def split_dataset(data_root, test_root=None, train_ratio=0.8, seed=42):
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

    if test_root is not None:
        test_dirs = [os.path.join(test_root, os.path.relpath(d, data_root))
                     for d in test_dirs]
    return train_dirs, test_dirs


def compute_cm(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    return cm, cm_norm


def plot_confusion_matrix(cm_norm, title, save_path):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues", vmin=0, vmax=1,
                xticklabels=CLASSES, yticklabels=CLASSES, annot_kws={"size": 12})
    plt.title(title, fontsize=16)
    plt.ylabel("True Action", fontsize=14)
    plt.xlabel("Model Prediction", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved confusion matrix: {save_path}")


def print_classification_report(y_true, y_pred, name):
    print(f"\n{'='*60}")
    print(f"  {name} Classification Report")
    print(f"{'='*60}")
    print(classification_report(y_true, y_pred, labels=[0,1,2,3,4],
                                target_names=CLASSES, digits=4))

    # Per-class detailed statistics
    cm, cm_norm = compute_cm(y_true, y_pred)
    print(f"\n  Per-class detailed statistics:")
    print(f"  {'Class':<8} {'GT':<8} {'Pred':<10} {'Correct':<8} {'Precision':<8} {'Recall':<8}")
    print(f"  {'-'*52}")
    for i, name_i in enumerate(CLASSES):
        gt_count = (np.array(y_true) == i).sum()
        pred_count = (np.array(y_pred) == i).sum()
        correct = cm[i, i]
        precision = correct / pred_count if pred_count > 0 else 0
        recall = correct / gt_count if gt_count > 0 else 0
        print(f"  {name_i:<8} {gt_count:<8} {pred_count:<10} {correct:<8} {precision*100:<7.2f}% {recall*100:<7.2f}%")

    # Error analysis: which classes are most often confused
    print(f"\n  Main confusion directions (as % of GT):")
    for i in range(5):
        row = cm[i]
        total = row.sum()
        if total == 0:
            continue
        # Find the target class most often misclassified into
        misclass = [(j, row[j]) for j in range(5) if j != i and row[j] > 0]
        misclass.sort(key=lambda x: -x[1])
        if misclass and misclass[0][1] / total >= 0.05:
            top_mis = ", ".join(f"{CLASSES[j]}({c}/{total}={c/total*100:.1f}%)" for j, c in misclass[:3])
            print(f"    {CLASSES[i]:<6}: → {top_mis}")


def print_top_k_accuracy(y_true, y_pred, k=2):
    """Requires softmax logits; approximated via the confusion matrix here."""
    pass


def main(yaml_path, weights_path):
    # Derive model name and output dir from the weights path
    model_dir = os.path.dirname(weights_path)
    model_name = os.path.basename(os.path.dirname(model_dir)) + "/" + os.path.basename(model_dir)
    out_dir = os.path.join(model_dir, "eval")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Config: {yaml_path}")
    print(f"Weights: {weights_path}")
    print(f"Output: {out_dir}")

    cfg = load_config(yaml_path)
    device = cfg["device"]

    # Dataset split (matches training, seed=42)
    if cfg.get("test_data_root"):
        train_dirs, test_dirs = split_dataset(cfg["data_root"], cfg["test_data_root"])
    else:
        train_dirs, test_dirs = split_dataset(cfg["data_root"])

    print(f"\nDataset: {cfg['data_root']}")
    print(f"  Train clips: {len(train_dirs)}")
    print(f"  Test clips: {len(test_dirs)}")

    train_ds = TennisActionDataset(cfg, clip_dirs=train_dirs)
    test_ds  = TennisActionDataset(cfg, clip_dirs=test_dirs)

    loader_kwargs = dict(batch_size=cfg["batch_size"], shuffle=False,
                         num_workers=cfg["num_workers"])
    train_loader = DataLoader(train_ds, **loader_kwargs)
    test_loader  = DataLoader(test_ds, **loader_kwargs)

    # Load model
    model = MSTFormer(cfg).to(device)
    state = torch.load(weights_path, map_location=device)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    print("\nModel weights loaded successfully")
    model.eval()

    def get_predictions(loader, desc):
        all_preds, all_labels = [], []
        with torch.no_grad():
            for pose, packed, labels, _kf in tqdm(loader, desc=desc):
                pose   = pose.to(device, non_blocking=True)
                packed = packed.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.amp.autocast("cuda"):
                    output = model(pose, packed)
                logits = output[0]  # (action_logits, keyframe_logits)
                preds = logits.argmax(-1)
                mask = labels != -100
                all_preds.extend(preds[mask].cpu().numpy())
                all_labels.extend(labels[mask].cpu().numpy())
        return np.array(all_labels), np.array(all_preds)

    print("\nEvaluating train set...")
    train_true, train_pred = get_predictions(train_loader, "Train")
    train_acc = accuracy_score(train_true, train_pred)
    print(f"  Train accuracy: {train_acc*100:.2f}%")

    print("\nEvaluating test set...")
    test_true, test_pred = get_predictions(test_loader, "Test")
    test_acc = accuracy_score(test_true, test_pred)
    print(f"  Test accuracy: {test_acc*100:.2f}%")

    # Confusion matrices
    _, train_cm_norm = compute_cm(train_true, train_pred)
    _, test_cm_norm  = compute_cm(test_true, test_pred)

    plot_confusion_matrix(train_cm_norm, f"Train Confusion Matrix (Acc={train_acc*100:.2f}%)",
                          os.path.join(out_dir, "confusion_matrix_train.png"))
    plot_confusion_matrix(test_cm_norm, f"Test Confusion Matrix (Acc={test_acc*100:.2f}%)",
                          os.path.join(out_dir, "confusion_matrix_test.png"))

    # Train set raw-count confusion matrix
    cm_train_raw, _ = compute_cm(train_true, train_pred)
    plot_confusion_matrix(cm_train_raw.astype(float),
                          f"Train Confusion Matrix (Counts)",
                          os.path.join(out_dir, "confusion_matrix_train_counts.png"))

    cm_test_raw, _ = compute_cm(test_true, test_pred)
    plot_confusion_matrix(cm_test_raw.astype(float),
                          f"Test Confusion Matrix (Counts)",
                          os.path.join(out_dir, "confusion_matrix_test_counts.png"))

    # Classification reports
    print_classification_report(train_true, train_pred, "Train")
    print_classification_report(test_true, test_pred, "Test")

    # Save numeric confusion matrices
    np.savetxt(os.path.join(out_dir, "cm_train.csv"), cm_train_raw, delimiter=",", fmt="%d")
    np.savetxt(os.path.join(out_dir, "cm_test.csv"), cm_test_raw, delimiter=",", fmt="%d")

    print(f"\nEvaluation complete, results saved to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="config path (override)")
    parser.add_argument("--weights", default=None, help="weights path (override)")
    args = parser.parse_args()

    if args.config and args.weights:
        # Use externally specified config and weights
        yaml_path = args.config
        weights_path = args.weights
    else:
        yaml_path = os.path.join(_PROJECT_DIR, "configs", "optimal.yaml")
        weights_path = os.path.join(_PROJECT_DIR, "models", "action",
                                    "optimal", "20260429_134556", "best.pth")
    main(yaml_path, weights_path)
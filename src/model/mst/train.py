"""train.py — MSTFormer v2 Training Script

Usage: python train.py [--config configs/main.yaml]
"""
import os
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
import torch
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
import argparse
import csv
import random
import shutil
import threading
from datetime import datetime
import torch.nn as nn
import torch.nn.functional as F
import cv2
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm

from dataset import TennisActionDataset
from model_main import MSTFormer
from config import load_config
from augment import AugmentBuffer


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

    # Test set uses the full dataset (untrimmed) — remap paths
    if test_root is not None:
        test_dirs = [os.path.join(test_root, os.path.relpath(d, data_root))
                     for d in test_dirs]

    print(f"Train Set: {len(train_dirs)} videos (data_root: {data_root})")
    print(f"  Test Set: {len(test_dirs)} videos (data_root: {test_root or data_root})")
    return train_dirs, test_dirs


class FocalLoss(nn.Module):
    def __init__(self, gamma, weight, ignore_index=-100):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight,
                             ignore_index=self.ignore_index, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        valid = targets != self.ignore_index
        return loss[valid].mean()


def build_criterion(cfg, device):
    weights = torch.tensor(cfg["class_weights"], dtype=torch.float32).to(device)
    if cfg.get("loss", "cross_entropy") == "focal":
        return FocalLoss(cfg.get("focal_gamma", 2.0), weights)
    return nn.CrossEntropyLoss(weight=weights, ignore_index=-100)


def train_model(cfg):
    device = cfg["device"]

    if "_smoke_clip" in cfg:
        one = cfg["_smoke_clip"]
        train_dirs = test_dirs = [one]
        # smoke mode reduces VRAM pressure while preserving virtual batch size semantics
        cfg["batch_size"] = 1
        cfg["seq_len"] = 60
        cfg["num_workers"] = 0
        cfg["accumulation_steps"] = cfg["virtual_batch_size"]  # Accumulate every step when batch=1
    else:
        train_dirs, test_dirs = split_dataset(cfg["data_root"],
                                               test_root=cfg.get("test_data_root"),
                                               train_ratio=cfg["train_ratio"])

    train_ds = TennisActionDataset(cfg, clip_dirs=train_dirs, augment=cfg.get("reshuffle_augment", True))
    test_ds = TennisActionDataset(cfg, clip_dirs=test_dirs, augment=False)

    nw = cfg["num_workers"]
    common_kwargs = dict(
        batch_size=cfg["batch_size"],
        num_workers=nw,
        pin_memory=cfg["pin_memory"],
    )
    # persistent_workers=True causes worker to hold an old copy of chunks; after reshuffle, the main process
    # len() might exceed the worker's chunk range, triggering an IndexError.
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=False,
                              persistent_workers=False, **common_kwargs)
    # When image augmentation is enabled, wrap the DataLoader in an async thread pool to avoid blocking the DataLoader workers
    if cfg.get("image_augment", False):
        n_threads = max(4, cfg.get("num_workers", 2) * 2)
        n_prefetch = max(3, cfg.get("num_workers", 2) + 1)
        train_loader = AugmentBuffer(train_loader, num_threads=n_threads, prefetch=n_prefetch)
        print(f"  Async Augmentation: {n_threads} threads, prefetch={n_prefetch}")
    test_loader = DataLoader(test_ds, shuffle=False,
                             persistent_workers=nw > 0, **common_kwargs)

    model = MSTFormer(cfg).to(device)
    criterion = build_criterion(cfg, device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg["learning_rate"],
                                  weight_decay=cfg["weight_decay"])

    warmup = cfg.get("warmup_epochs", 5)
    total_epochs = cfg["total_epochs"]
    scheduler = SequentialLR(optimizer, schedulers=[
        LinearLR(optimizer, start_factor=0.1, total_iters=warmup),
        CosineAnnealingLR(optimizer, T_max=total_epochs - warmup,
                          eta_min=cfg["learning_rate"] * 0.01),
    ], milestones=[warmup])

    scaler = torch.amp.GradScaler("cuda")
    accum = cfg["accumulation_steps"]
    num_classes = cfg["num_classes"]
    kf_loss_weight = cfg.get("keyframe_loss_weight", 0.5)
    keyframe_only = cfg.get("keyframe_only", False)

    config_stem = os.path.splitext(os.path.basename(cfg.get("_yaml_path", "main.yaml")))[0]
    _mst_dir = os.path.dirname(os.path.abspath(__file__))
    _project_dir = os.path.dirname(os.path.dirname(os.path.dirname(_mst_dir)))
    run_dir = os.path.join(_project_dir, "models", "action", config_stem,
                           datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)
    best_path  = os.path.join(run_dir, "best.pth")
    final_path = os.path.join(run_dir, "final.pth")
    log_path   = os.path.join(run_dir, "train_log.csv")

    # Save configuration snapshot
    yaml_src = cfg.get("_yaml_path")
    if yaml_src and os.path.exists(yaml_src):
        shutil.copy2(yaml_src, os.path.join(run_dir, "config.yaml"))

    # Initialize CSV log
    _log_fields = [
        "epoch", "lr", "train_loss", "train_acc",
        "test_acc", "kf_precision", "kf_recall", "kf_f1",
        "pred_idle", "pred_fh", "pred_bh", "pred_serve", "pred_move",
        "gt_idle",   "gt_fh",   "gt_bh",   "gt_serve",   "gt_move",
        "best_metric",
    ]
    with open(log_path, "w", newline="", encoding="utf-8") as _f:
        csv.DictWriter(_f, fieldnames=_log_fields).writeheader()

    best_metric = -1.0
    best_state: dict | None = None
    _save_thread: threading.Thread | None = None

    def _async_save(state_dict, path):
        torch.save(state_dict, path)

    print(f"Starting training for {total_epochs} epochs, Warmup: {warmup} epochs")
    print("=" * 75)

    for epoch in range(total_epochs):
        train_ds.reshuffle()
        model.train()
        total_loss = correct = total = 0

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{total_epochs}] Training")
        for i, (pose, packed, labels, kf_labels) in enumerate(pbar):
            pose      = pose.to(device, non_blocking=True)
            packed    = packed.to(device, non_blocking=True)
            labels    = labels.to(device, non_blocking=True)
            kf_labels = kf_labels.to(device, non_blocking=True)

            with torch.amp.autocast("cuda"):
                if keyframe_only:
                    kf_logits = model(pose, packed)
                    loss = F.cross_entropy(kf_logits.view(-1, 2), kf_labels.view(-1)) / accum
                else:
                    action_logits, kf_logits = model(pose, packed)
                    loss_action = criterion(action_logits.view(-1, num_classes), labels.view(-1))
                    loss_kf = F.cross_entropy(kf_logits.view(-1, 2), kf_labels.view(-1))
                    loss = (loss_action + kf_loss_weight * loss_kf) / accum

            scaler.scale(loss).backward()
            if (i + 1) % accum == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            with torch.no_grad():
                if not keyframe_only:
                    preds = action_logits.argmax(-1)
                    mask = labels != -100
                    correct += ((preds == labels) & mask).sum().item()
                    total += mask.sum().item()
            total_loss += loss.item() * accum
            pbar.set_postfix(loss=f"{loss.item()*accum:.4f}")

        scheduler.step()
        train_acc = correct / total if total > 0 else 0
        avg_loss = total_loss / len(train_loader)

        # Evaluation
        torch.cuda.empty_cache()
        model.eval()
        t_correct = t_total = 0
        all_preds, all_labels = [], []
        kf_tp = kf_fp = kf_fn = 0
        with torch.no_grad():
            for pose, packed, labels, kf_labels in test_loader:
                pose      = pose.to(device, non_blocking=True)
                packed    = packed.to(device, non_blocking=True)
                labels    = labels.to(device, non_blocking=True)
                kf_labels = kf_labels.to(device, non_blocking=True)
                with torch.amp.autocast("cuda"):
                    if keyframe_only:
                        kf_logits = model(pose, packed)
                    else:
                        action_logits, kf_logits = model(pose, packed)
                if not keyframe_only:
                    preds = action_logits.argmax(-1)
                    mask = labels != -100
                    t_correct += ((preds == labels) & mask).sum().item()
                    t_total += mask.sum().item()
                    all_preds.append(preds[mask].cpu())
                    all_labels.append(labels[mask].cpu())
                kf_preds = kf_logits.argmax(-1)
                kf_tp += ((kf_preds == 1) & (kf_labels == 1)).sum().item()
                kf_fp += ((kf_preds == 1) & (kf_labels == 0)).sum().item()
                kf_fn += ((kf_preds == 0) & (kf_labels == 1)).sum().item()

        test_acc = t_correct / t_total if t_total > 0 else 0
        all_preds = torch.cat(all_preds) if all_preds else torch.zeros(0, dtype=torch.long)
        all_labels = torch.cat(all_labels) if all_labels else torch.zeros(0, dtype=torch.long)
        ld = torch.bincount(all_labels, minlength=num_classes).tolist()
        pd = torch.bincount(all_preds, minlength=num_classes).tolist()
        kf_prec = kf_tp / (kf_tp + kf_fp) if (kf_tp + kf_fp) > 0 else 0.0
        kf_rec  = kf_tp / (kf_tp + kf_fn) if (kf_tp + kf_fn) > 0 else 0.0

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch [{epoch+1}/{total_epochs}]  lr={lr_now:.2e}")
        print(f"   Loss: {avg_loss:.4f}", end="")
        if not keyframe_only:
            train_acc = correct / total if total > 0 else 0
            print(f" | Train Acc: {train_acc*100:.2f}% | Test Acc: {test_acc*100:.2f}%")
            print(f"   GT:   Idle={ld[0]} Forehand={ld[1]} Backhand={ld[2]} Serve={ld[3]} Move={ld[4]}")
            print(f"   Pred: Idle={pd[0]} Forehand={pd[1]} Backhand={pd[2]} Serve={pd[3]} Move={pd[4]}")
        else:
            print()
        print(f"   Keyframe: Precision={kf_prec*100:.1f}% Recall={kf_rec*100:.1f}% (TP={kf_tp} FP={kf_fp} FN={kf_fn})")

        # Track best model: F1 score for keyframe_only, test_acc otherwise
        if keyframe_only:
            cur_metric = (2 * kf_prec * kf_rec / (kf_prec + kf_rec)) if (kf_prec + kf_rec) > 0 else 0.0
        else:
            cur_metric = test_acc
        if cur_metric > best_metric:
            best_metric = cur_metric
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if _save_thread is not None:
                _save_thread.join()
            _save_thread = threading.Thread(target=_async_save, args=(best_state, best_path), daemon=True)
            _save_thread.start()
            print(f"   New Best {'F1' if keyframe_only else 'Acc'}={run_dir} {cur_metric*100:.2f}%, saving in background...")

        # CSV log entries
        kf_f1 = (2 * kf_prec * kf_rec / (kf_prec + kf_rec)) if (kf_prec + kf_rec) > 0 else 0.0
        row = {
            "epoch":       epoch + 1,
            "lr":          f"{lr_now:.2e}",
            "train_loss":  f"{avg_loss:.4f}",
            "train_acc":   f"{train_acc*100:.2f}" if not keyframe_only else "",
            "test_acc":    f"{test_acc*100:.2f}"  if not keyframe_only else "",
            "kf_precision": f"{kf_prec*100:.2f}",
            "kf_recall":    f"{kf_rec*100:.2f}",
            "kf_f1":        f"{kf_f1*100:.2f}",
            "pred_idle":  pd[0] if not keyframe_only else "",
            "pred_fh":    pd[1] if not keyframe_only else "",
            "pred_bh":    pd[2] if not keyframe_only else "",
            "pred_serve": pd[3] if not keyframe_only else "",
            "pred_move":  pd[4] if not keyframe_only else "",
            "gt_idle":    ld[0] if not keyframe_only else "",
            "gt_fh":      ld[1] if not keyframe_only else "",
            "gt_bh":      ld[2] if not keyframe_only else "",
            "gt_serve":   ld[3] if not keyframe_only else "",
            "gt_move":    ld[4] if not keyframe_only else "",
            "best_metric": f"{best_metric*100:.2f}",
        }
        with open(log_path, "a", newline="", encoding="utf-8") as _f:
            csv.DictWriter(_f, fieldnames=_log_fields).writerow(row)

        print("-" * 75)

    # Save final weights
    torch.save(model.state_dict(), final_path)
    if _save_thread is not None:
        _save_thread.join()
    print(f"Training complete. Final weights saved to: {final_path}")
    print(f"   Best weights: {best_path}  ({'F1' if keyframe_only else 'Acc'}={best_metric*100:.2f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None, help="Path to the YAML configuration file")
    parser.add_argument("--smoke", action="store_true", help="Run 1 epoch with 1 sample to validate pipeline")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["_yaml_path"] = args.config or "main.yaml"

    if args.smoke:
        cfg["total_epochs"] = 1
        cfg["warmup_epochs"] = 0
        data_root = cfg["data_root"]
        one_clip = next(
            os.path.join(data_root, d) for d in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, d))
        )
        cfg["_smoke_clip"] = one_clip

    train_model(cfg)
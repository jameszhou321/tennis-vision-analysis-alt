"""
Trim waiting segments in the dataset to mitigate class imbalance.
Reads data from data/rallies_train/, processes it, and outputs to data/rallies_train_trimmed/.

Three methods (independent probabilities, stackable, leaves original data assets unaltered):
  Method 1 (75%): Retain a maximum of 2s for trailing wait segments. Trims only the last item during consecutive waits.
  Method 2 (70%): Wait segments between two serves: keep up to 1.5s after Serve 1, and up to 1.0s before Serve 2.
  Method 3 (50%): Retain only the last 2s for leading wait segments.
"""

import json
import random
import shutil
import os
import sys
from pathlib import Path

# ─── Configuration ──────────────────────────────────────────────────────────
SRC_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "rallies_train"
DST_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "rallies_train_trimmed"
SEED = 42
PROB_M1 = 0.75   # Trailing wait -> max 2s
PROB_M2 = 0.70   # Wait segments between serves trim
PROB_M3 = 0.50   # Leading wait -> last 2s

ACTION_WAIT = 0
ACTION_SERVE = 3

# ─── I/O Operations ─────────────────────────────────────────────────────────


def load_annotations(rally_path):
    ann_path = os.path.join(rally_path, "annotations.json")
    if not os.path.exists(ann_path):
        return None
    with open(ann_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_annotations(annotations, dst_path):
    ann_path = os.path.join(dst_path, "annotations.json")
    with open(ann_path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, ensure_ascii=False, indent=2)


# ─── Condition Evaluation ───────────────────────────────────────────────────


def count_trailing_wait(annots):
    """Count the number of consecutive trailing wait segments (=0 means trailing segment is not a wait)."""
    cnt = 0
    for a in reversed(annots):
        if a["action_id"] == ACTION_WAIT:
            cnt += 1
        else:
            break
    return cnt


def has_two_serves(annots):
    return sum(1 for a in annots if a["action_id"] == ACTION_SERVE) >= 2


def has_leading_wait(annots):
    return bool(annots) and annots[0]["action_id"] == ACTION_WAIT


# ─── Method 1: Trim Trailing Wait ───────────────────────────────────────────


def trim_trailing_wait(annots):
    """Retain a maximum of 2s for trailing wait segments. Trims only the last item during consecutive waits."""
    consecutive = count_trailing_wait(annots)
    if consecutive == 0:
        return annots

    # Process only the last wait segment
    last = annots[-1]
    duration = last["end_time"] - last["start_time"]
    if duration > 2.0:
        last["end_time"] = round(last["start_time"] + 2.0, 3)
    return annots


# ─── Method 2: Wait Segments Between Serves ─────────────────────────────────


def trim_between_serves(annots):
    """Wait segments between two serves:
    - Retain wait data within 1.5s after Serve 1 ends.
    - Retain wait data within 1.0s before Serve 2 starts.
    - Completely remove all other wait segments.
    """
    serve_indices = [i for i, a in enumerate(annots) if a["action_id"] == ACTION_SERVE]
    if len(serve_indices) < 2:
        return annots

    idx1, idx2 = serve_indices[0], serve_indices[1]
    serve1_end = annots[idx1]["end_time"]
    serve2_start = annots[idx2]["start_time"]

    keep_end = serve1_end + 1.5    # Retain up to 1.5s after Serve 1
    keep_start = serve2_start - 1.0  # Retain from 1.0s before Serve 2

    # Executed in two phases:
    # 1) Collect all operations to perform (log changes, leave original list unchanged)
    # 2) Execute processing backwards from back to front (to prevent array index shifting errors)

    actions = []  # (index_in_annots, 'remove') or (index, 'modify', new_start, new_end)
    new_segs_before_serve2 = []  # dicts to insert before serve2

    between_indices = list(range(idx1 + 1, idx2))
    for i in between_indices:
        s = annots[i]
        if s["action_id"] != ACTION_WAIT:
            continue

        ws, we = s["start_time"], s["end_time"]

        # Evaluate overlay bounds within designated retention zones
        in_keep1 = ws < keep_end    # Within the 1.5s window after Serve 1
        in_keep2 = we > keep_start  # Within the 1.0s window before Serve 2

        if not in_keep1 and not in_keep2:
            actions.append((i, "remove"))

        elif in_keep1 and in_keep2:
            if keep_end >= keep_start:
                # Retention zones overlap; extract intersection bounds
                new_s = max(ws, keep_start)
                new_e = min(we, keep_end)
                if new_s < new_e:
                    actions.append((i, "modify", new_s, new_e))
                else:
                    actions.append((i, "remove"))
            else:
                # Non-overlapping; bifurcate data into two distinct segments
                # First half segment: [ws, keep_end]
                if keep_end > ws:
                    actions.append((i, "modify", ws, keep_end))
                else:
                    actions.append((i, "remove"))
                # Second half segment: [keep_start, we]
                if we > keep_start:
                    new_segs_before_serve2.append({
                        "start_time": round(keep_start, 3),
                        "end_time": round(we, 3),
                        "action_name": s["action_name"],
                        "action_id": s["action_id"]
                    })

        elif in_keep1:
            new_e = min(we, keep_end)
            actions.append((i, "modify", ws, new_e))

        else:  # in_keep2 only
            new_s = max(ws, keep_start)
            actions.append((i, "modify", new_s, we))

    # Execute processing backwards from back to front
    actions.sort(key=lambda x: x[0], reverse=True)
    for act in actions:
        idx = act[0]
        if act[1] == "remove":
            annots.pop(idx)
        else:
            _, _, new_s, new_e = act
            annots[idx]["start_time"] = round(new_s, 3)
            annots[idx]["end_time"] = round(new_e, 3)

    # Insert parsed new tracking segment pieces immediately before Serve 2
    # Look up the current sequence position for Serve 2
    cur_serve_indices = [i for i, a in enumerate(annots) if a["action_id"] == ACTION_SERVE]
    if len(cur_serve_indices) >= 2:
        pos = cur_serve_indices[1]
        for seg in new_segs_before_serve2:
            annots.insert(pos, seg)
            pos += 1

    return annots


# ─── Method 3: Trim Leading Wait ────────────────────────────────────────────


def trim_leading_wait(annots):
    """Retain only the last 2s for leading wait segments."""
    if not annots or annots[0]["action_id"] != ACTION_WAIT:
        return annots
    first = annots[0]
    duration = first["end_time"] - first["start_time"]
    if duration > 2.0:
        first["start_time"] = round(first["end_time"] - 2.0, 3)
    return annots


# ─── Replicating and Processing ─────────────────────────────────────────────


def copy_with_hardlinks(src, dst):
    """Replicate directory layout using hardlinks to conserve storage, ignoring annotations.json."""
    if dst.exists():
        shutil.rmtree(dst)

    def _ignore(src_dir, names):
        return {"annotations.json"}

    shutil.copytree(src, dst, ignore=_ignore, copy_function=os.link,
                    dirs_exist_ok=True)


def process_rally(src_path, dst_path, rng):
    """Load annotations -> selectively apply the 3 processing methods by probability -> export alterations."""
    raw_annots = load_annotations(src_path)
    if raw_annots is None:
        return []

    annots = json.loads(json.dumps(raw_annots))  # Deep copy allocation
    applied = []

    # Method 1 Execution Block
    if count_trailing_wait(annots) > 0 and rng.random() < PROB_M1:
        annots = trim_trailing_wait(annots)
        applied.append("m1")

    # Method 2 Execution Block
    if has_two_serves(annots) and rng.random() < PROB_M2:
        annots = trim_between_serves(annots)
        applied.append("m2")

    # Method 3 Execution Block
    if has_leading_wait(annots) and rng.random() < PROB_M3:
        annots = trim_leading_wait(annots)
        applied.append("m3")

    # Export modifications to disk only if data metrics have shifted
    if applied:
        save_annotations(annots, dst_path)

    return applied


# ─── Main Pipeline Process ──────────────────────────────────────────────────


def main():
    if not SRC_ROOT.exists():
        print(f"Error: Source directory does not exist at {SRC_ROOT}")
        sys.exit(1)

    DST_ROOT.mkdir(parents=True, exist_ok=True)

    rally_dirs = sorted([d for d in SRC_ROOT.iterdir() if d.is_dir()])
    total = len(rally_dirs)
    rng = random.Random(SEED)

    print(f"Source Root Path: {SRC_ROOT}")
    print(f"Destination Root Path: {DST_ROOT}")
    print(f"Total Segment Assets: {total}")
    print(f"Method 1 (Trailing Wait -> 2s): P={PROB_M1*100:.0f}%")
    print(f"Method 2 (Wait Between Serves):  P={PROB_M2*100:.0f}%")
    print(f"Method 3 (Leading Wait -> 2s):  P={PROB_M3*100:.0f}%")
    print(f"Randomization Seed: {SEED}")
    print(f"File System Protocol: OS Hardlinks (Zero extra space allocated)")
    print()

    stats = {"m1": 0, "m2": 0, "m3": 0, "any": 0, "skipped_no_ann": 0}

    for i, src in enumerate(rally_dirs):
        dst = DST_ROOT / src.name
        indent = " " * 4
        print(f"[{i+1}/{total}] {src.name}")

        ann_path = src / "annotations.json"
        if not ann_path.exists():
            print(f"{indent}Skipped (annotations.json missing from directory)")
            stats["skipped_no_ann"] += 1
            continue

        # Replicate directory components using OS Hardlinks
        copy_with_hardlinks(src, dst)

        # Parse annotations structure array based on running probabilities
        local_rng = random.Random(SEED + i)
        applied = process_rally(src, dst, local_rng)

        if applied:
            print(f"{indent}Processed: {' + '.join(applied)}")
            stats["any"] += 1
            for m in applied:
                stats[m] += 1
        else:
            print(f"{indent}No Modification Hits (Random skip or conditions unmet)")

    print()
    print("=" * 50)
    print("Processing Statistics Summary:")
    print(f"  Total Clips:               {total}")
    print(f"  Valid Annotated Clips:     {total - stats['skipped_no_ann']}")
    print(f"  Modified (At least 1 method): {stats['any']}")
    print(f"  Method 1 (Trailing Wait -> 2s):  {stats['m1']}")
    print(f"  Method 2 (Wait Between Serves):   {stats['m2']}")
    print(f"  Method 3 (Leading Wait -> 2s):  {stats['m3']}")
    print(f"  Skipped Assets (No Metadata):     {stats['skipped_no_ann']}")
    print(f"  Export Output Directory:    {DST_ROOT}")
    print("=" * 50)


if __name__ == "__main__":
    main()
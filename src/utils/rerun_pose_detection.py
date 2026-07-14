"""
Rerun pose detection on player1/player2 cropped images with a low threshold, 
map coordinates back to the original frame, and filter using the person bbox.
Overwrite the 'keypoints' field in pose_data.json (retaining 'bbox' and 'court' fields).
Count empty detection frames and write them to logs/pose_rerun_stats.json.
Supports resuming from breakpoints (--force to force rerun).
"""
import os
import json
import ctypes
import argparse
import numpy as np
import cv2
from ultralytics import YOLO
from tqdm import tqdm

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CROP_SIZE = 320
CONF_THRESH = 0.1   # Extremely low threshold to detect as many keypoints as possible


def get_short_path(path):
    buf = ctypes.create_unicode_buffer(512)
    if not hasattr(ctypes, "windll"):  # Non-Windows systems use the original path directly
        return path
    ctypes.windll.kernel32.GetShortPathNameW(path, buf, 512)
    return buf.value or path


def _calc_win(bbox, base_win):
    """Calculate crop window size from bbox [x1, y1, x2, y2] and base_win (consistent with extract_crops.py)."""
    box_side = int(max(bbox[2] - bbox[0], bbox[3] - bbox[1]))
    return max(base_win, box_side)


def _crop_to_orig(x_crop, y_crop, cx, cy, win):
    """Map cropped image coordinates -> original frame coordinates."""
    x_orig = (x_crop / CROP_SIZE) * win + (cx - win / 2)
    y_orig = (y_crop / CROP_SIZE) * win + (cy - win / 2)
    return x_orig, y_orig


def _in_bbox(x, y, bbox):
    return bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]


def _run_pose_on_crop(crop_path, pose_model):
    """Run pose detection on the cropped image, returning 17 keypoints [[x, y, conf], ...] or None."""
    if not os.path.exists(crop_path):
        return None
    raw = np.fromfile(crop_path, dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if img is None:
        return None
    results = pose_model(img, verbose=False, conf=CONF_THRESH)
    if not results or results[0].keypoints is None:
        return None
    kps = results[0].keypoints
    if kps.xy is None or len(kps.xy) == 0:
        return None
    # Pick the person with the highest confidence
    if kps.conf is not None and len(kps.conf) > 1:
        best = int(kps.conf.mean(dim=1).argmax())
    else:
        best = 0
    xy = kps.xy[best].cpu().numpy()
    conf = kps.conf[best].cpu().numpy() if kps.conf is not None else np.ones(len(xy))
    return [[float(xy[i, 0]), float(xy[i, 1]), float(conf[i])] for i in range(len(xy))]


def process_clip(clip_dir, pose_model, force=False):
    pose_path = os.path.join(clip_dir, "pose_data.json")
    video_path = os.path.join(clip_dir, "raw_clip.mp4")
    p1_dir = os.path.join(clip_dir, "player1")
    p2_dir = os.path.join(clip_dir, "player2")

    if not os.path.exists(pose_path):
        return None, "Missing pose_data.json"
    if not os.path.isdir(p1_dir) or not os.path.isdir(p2_dir):
        return None, "Missing crop image directories"

    with open(pose_path, "r", encoding="utf-8") as f:
        pose_data = json.load(f)

    # Resume from breakpoint: check if already processed (marker flag)
    if not force:
        first = pose_data[0] if isinstance(pose_data, list) and pose_data else None
        if first and first.get("_pose_rerun"):
            return None, "Skipped"

    # Get video width to calculate base_win
    short_path = get_short_path(video_path) if os.path.exists(video_path) else None
    base_win = 300  # Default value
    if short_path:
        cap = cv2.VideoCapture(short_path)
        vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()
        if vid_w > 0:
            base_win = round(vid_w / 6.4)

    is_list = isinstance(pose_data, list)
    entries = pose_data if is_list else list(pose_data.values())

    empty_near = empty_far = total = 0

    for entry in entries:
        if entry is None:
            continue
        fid = entry.get("frame", total)
        name = f"{fid:06d}.jpg"
        total += 1

        for role, crop_dir in [("near_player", p1_dir), ("far_player", p2_dir)]:
            player = entry.get(role)
            if player is None:
                if role == "near_player":
                    empty_near += 1
                else:
                    empty_far += 1
                continue

            bbox = player.get("bbox")
            if bbox is None:
                # If bbox is None (interpolated frame), keep keypoints as an empty list
                player["keypoints"] = []
                if role == "near_player":
                    empty_near += 1
                else:
                    empty_far += 1
                continue

            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            win = _calc_win(bbox, base_win)

            crop_path = os.path.join(crop_dir, name)
            raw_kps = _run_pose_on_crop(crop_path, pose_model)

            if raw_kps is None:
                # Retain original keypoints but set all confidences to zero
                orig_kps = player.get("keypoints", [])
                new_kps = [[kp[0], kp[1], 0.0] for kp in orig_kps] if orig_kps else []
                player["keypoints"] = new_kps
                if role == "near_player":
                    empty_near += 1
                else:
                    empty_far += 1
                continue

            # Map coordinates back to original frame + filter by bbox
            new_kps = []
            for kp in raw_kps:
                x_orig, y_orig = _crop_to_orig(kp[0], kp[1], cx, cy, win)
                c = kp[2] if _in_bbox(x_orig, y_orig, bbox) else 0.0
                new_kps.append([x_orig, y_orig, c])

            player["keypoints"] = new_kps

        entry["_pose_rerun"] = True  # Mark as processed

    with open(pose_path, "w", encoding="utf-8") as f:
        json.dump(pose_data, f, ensure_ascii=False)

    stats = {
        "total_frames": total,
        "empty_near": empty_near,
        "empty_far": empty_far,
        "empty_near_pct": round(empty_near / total * 100, 1) if total else 0,
        "empty_far_pct": round(empty_far / total * 100, 1) if total else 0,
    }
    return stats, "Success"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=os.path.join(_PROJECT_DIR, "data", "rallies_annotated"))
    parser.add_argument("--model", default=os.path.join(_PROJECT_DIR, "models", "yolo", "yolo11x-pose.pt"))
    parser.add_argument("--force", action="store_true", help="Force rerun on already processed rallies")
    args = parser.parse_args()

    print(f"Loading pose model: {args.model}  Confidence threshold: {CONF_THRESH}")
    pose_model = YOLO(args.model)

    clips = sorted(d for d in os.listdir(args.data_root)
                   if os.path.isdir(os.path.join(args.data_root, d)))

    all_stats = {}
    done = skipped = failed = 0

    for clip_name in tqdm(clips, desc="Rerunning pose detection"):
        clip_dir = os.path.join(args.data_root, clip_name)
        stats, msg = process_clip(clip_dir, pose_model, force=args.force)
        if msg == "Skipped":
            skipped += 1
        elif msg == "Success":
            done += 1
            all_stats[clip_name] = stats
        else:
            failed += 1
            tqdm.write(f"  [Failed] {clip_name}: {msg}")

    # Write stats file
    logs_dir = os.path.join(_PROJECT_DIR, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    stats_path = os.path.join(logs_dir, "pose_rerun_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)

    # Summary
    if all_stats:
        avg_near = np.mean([s["empty_near_pct"] for s in all_stats.values()])
        avg_far  = np.mean([s["empty_far_pct"]  for s in all_stats.values()])
        print(f"\nAverage empty detection rate — near_player: {avg_near:.1f}%, far_player: {avg_far:.1f}%")

    print(f"Done: {done} processed, {skipped} skipped, {failed} failed")
    print(f"Stats file written to: {stats_path}")


if __name__ == "__main__":
    main()
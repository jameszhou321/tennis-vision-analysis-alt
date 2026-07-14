"""
Merge new annotation data from data/rallies_annotating/ into data/rallies_annotated/.
tracking_data.json → pose_data.json (including 'court' field)
id=1 → near_player, id=2 → far_player
Rally numbering continues starting from the maximum index in rallies_annotated + 1.
"""
import os
import json
import shutil
import re

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_ROOT = os.path.join(_PROJECT_DIR, "data", "rallies_annotating")
DST_ROOT = os.path.join(_PROJECT_DIR, "data", "rallies_annotated")


def _get_max_rally_num(dst_root):
    max_num = 0
    for name in os.listdir(dst_root):
        m = re.match(r"rally_(\d+)_", name)
        if m:
            max_num = max(max_num, int(m.group(1)))
    return max_num


def _convert_tracking_to_pose(tracking_path):
    """Convert tracking_data.json to pose_data.json format (list, indexed by frame_id)."""
    with open(tracking_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    frames = data.get("frames", [])
    if not frames:
        return []

    total = max(fr["frame_id"] for fr in frames) + 1
    pose_list = [None] * total

    for fr in frames:
        fid = fr["frame_id"]
        court = fr.get("court", [])
        players = {p["id"]: p for p in fr.get("players", [])}

        entry = {"frame": fid, "court": court}

        for role, pid in [("near_player", 1), ("far_player", 2)]:
            p = players.get(pid)
            if p:
                entry[role] = {
                    "bbox": p["bbox"],
                    "keypoints": p["pose"],
                }
            else:
                entry[role] = None

        pose_list[fid] = entry

    # Fill empty frames (in case frame_id is non-consecutive)
    result = []
    for i, entry in enumerate(pose_list):
        if entry is None:
            result.append({"frame": i, "court": [], "near_player": None, "far_player": None})
        else:
            result.append(entry)

    return result


def main():
    next_num = _get_max_rally_num(DST_ROOT) + 1
    print(f"Max index in rallies_annotated: {next_num - 1}, new rally starting from index {next_num}")

    copied = skipped = failed = 0

    match_dirs = sorted(
        d for d in os.listdir(SRC_ROOT)
        if os.path.isdir(os.path.join(SRC_ROOT, d)) and not d.startswith("_")
    )

    for match_dir in match_dirs:
        match_path = os.path.join(SRC_ROOT, match_dir)
        rally_dirs = sorted(
            d for d in os.listdir(match_path)
            if os.path.isdir(os.path.join(match_path, d))
        )

        for rally_dir in rally_dirs:
            src_rally = os.path.join(match_path, rally_dir)

            # Extract duration suffix
            m = re.search(r"(\d+\.\d+s)$", rally_dir)
            duration = m.group(1) if m else "0.0s"

            # Check for required files
            tracking_path = os.path.join(src_rally, "tracking_data.json")
            anno_path = os.path.join(src_rally, "annotations.json")
            video_path = os.path.join(src_rally, "raw_clip.mp4")

            if not all(os.path.exists(p) for p in [tracking_path, anno_path, video_path]):
                print(f"  [SKIP] {match_dir}/{rally_dir} — Missing required files")
                skipped += 1
                continue

            new_name = f"rally_{next_num:03d}_{duration}"
            dst_rally = os.path.join(DST_ROOT, new_name)

            if os.path.exists(dst_rally):
                print(f"  [SKIP] {new_name} — Destination already exists")
                skipped += 1
                next_num += 1
                continue

            try:
                os.makedirs(dst_rally, exist_ok=True)

                # Convert pose data
                pose_data = _convert_tracking_to_pose(tracking_path)
                pose_out = os.path.join(dst_rally, "pose_data.json")
                with open(pose_out, "w", encoding="utf-8") as f:
                    json.dump(pose_data, f, ensure_ascii=False)

                # Copy other files
                shutil.copy2(anno_path, os.path.join(dst_rally, "annotations.json"))
                shutil.copy2(video_path, os.path.join(dst_rally, "raw_clip.mp4"))

                print(f"  [COPY] {match_dir}/{rally_dir} → {new_name}")
                copied += 1
                next_num += 1

            except Exception as e:
                print(f"  [ERR]  {match_dir}/{rally_dir}: {e}")
                failed += 1

    print(f"\nCompleted: Copied {copied}, Skipped {skipped}, Failed {failed}")
    print(f"Next available index for rallies_annotated: {next_num}")


if __name__ == "__main__":
    main()
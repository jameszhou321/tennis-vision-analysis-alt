"""dataset.py — Action Recognition Dataset Loader v2

Supports a 3-way visual stream (full frame + player 1 + player 2), controlled via config switches.
pose_tensor dimensions: 125 = 17×3 (absolute coords + conf) + 17×2 (relative center offsets) + 2 (person center, relative to court center)
+ 2 (velocity, court relative coordinate difference) + 2 (acceleration)
+ 6 (ball: position 2 + velocity 2 + acceleration 2, temporarily all zeros)
+ 28 (14 court points × 2, zeroed out when conf < 0.3)
"""
import os
import json
import ctypes
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset

POSE_DIM = 125  # 91 + 6 (ball) + 28 (14 court points × 2)

# Unified height is 320. Width after 3-way horizontal concatenation = 320 + 320 + 320 = 960.
# Normalization is performed on the GPU side; the CPU side maintains uint8 to save PCIe bandwidth.
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


def _get_short_path(path):
    buf = ctypes.create_unicode_buffer(512)
    if not hasattr(ctypes, "windll"):  # Use original path directly on non-Windows systems
        return path
    ctypes.windll.kernel32.GetShortPathNameW(path, buf, 512)
    return buf.value or path


def _resize_uint8(img_bgr, h, w):
    """Resizes BGR uint8 image and returns RGB uint8 numpy array [H, W, 3]"""
    img = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ── Image Data Augmentation (Used during training) ────────────────────────────────────────────

def _apply_augmentations(rgb):
    """Sequentially applies augmentations to a single RGB uint8 [H, W, 3] image."""
    # 1) Color Jitter (brightness/contrast/saturation/hue)
    if np.random.rand() < 0.6:
        brightness = np.random.uniform(0.7, 1.3)
        contrast   = np.random.uniform(0.7, 1.3)
        saturation = np.random.uniform(0.7, 1.3)
        hue        = np.random.uniform(-0.04, 0.04)
        img = rgb.astype(np.float32)
        img = img * contrast + 128 * (1 - contrast)
        img = np.clip(img * brightness, 0, 255)
        hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] *= saturation
        hsv[:, :, 0] += hue * 180
        hsv = np.clip(hsv, 0, 255).astype(np.uint8)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # 2) Gaussian Noise
    if np.random.rand() < 0.4:
        noise = np.random.randn(*rgb.shape).astype(np.float32) * 8
        rgb = np.clip(rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # 3) Gaussian Blur
    if np.random.rand() < 0.3:
        k = np.random.choice([3, 5])
        rgb = cv2.GaussianBlur(rgb, (k, k), 0)

    # 4) Random Erasing (Cutout)
    if np.random.rand() < 0.3:
        h, w = rgb.shape[:2]
        area = h * w
        for _ in range(np.random.randint(1, 3)):
            erase_ratio = np.random.uniform(0.02, 0.12)
            ew = max(8, int(np.sqrt(area * erase_ratio)))
            eh = max(8, int(np.sqrt(area * erase_ratio)))
            x = np.random.randint(0, w - ew) if w > ew else 0
            y = np.random.randint(0, h - eh) if h > eh else 0
            color = np.random.randint(0, 256, size=(3,)).tolist()
            rgb[y:y+eh, x:x+ew] = color

    # 5) Translucent Color Overlay (Requested by user)
    if np.random.rand() < 0.25:
        h, w = rgb.shape[:2]
        overlay = np.full((h, w, 3), np.random.randint(0, 256, size=(3,)), dtype=np.uint8)
        alpha = np.random.uniform(0.05, 0.2)
        rgb = np.clip(rgb.astype(np.float32) * (1 - alpha) + overlay.astype(np.float32) * alpha,
                      0, 255).astype(np.uint8)

    return rgb


def _parse_player(frame_data, player_key):
    """Extracts the bbox center and 17 keypoints of the specified player from single-frame data. Returns None if failed."""
    p = frame_data.get(player_key) if isinstance(frame_data, dict) else None
    if p is None:
        return None
    kps = p.get("keypoints", [])
    bbox = p.get("bbox")
    if len(kps) != 17 or bbox is None:
        return None
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    return {"cx": cx, "cy": cy, "kps": kps}


def _build_pose_vec(player, cx_prev, cy_prev, vx_prev, vy_prev, court_kps=None, W=1920.0, H=1080.0):
    """
    Builds a single-frame 125-dimensional physical feature vector.
    player: Return value from _parse_player (dict), returns an all-zero vector if None.
    court_kps: 14 court keypoints [[x,y,conf],...] or None.
    cx_prev/cy_prev: Court relative coordinates of the previous frame (normalized), None for the first frame.
    Returns (vec_125, rel_cx, rel_cy, vx, vy) for the next frame's use (coordinates are all court relative normalized values).
    """
    # Court center (mean of points with conf >= 0.3, falls back to frame center)
    court_cx, court_cy = W / 2, H / 2
    if court_kps:
        valid = [kp for kp in court_kps if len(kp) >= 3 and kp[2] >= 0.3]
        if valid:
            court_cx = float(np.mean([kp[0] for kp in valid]))
            court_cy = float(np.mean([kp[1] for kp in valid]))

    if player is None:
        return np.zeros(POSE_DIM, dtype=np.float32), cx_prev, cy_prev, vx_prev, vy_prev

    cx, cy = player["cx"], player["cy"]
    kps = player["kps"]

    # 17×3: Absolute coordinates + conf
    abs_part = []
    for kp in kps:
        abs_part.extend([kp[0] / W, kp[1] / H, kp[2]])

    # 17×2: Offset relative to person center
    rel_part = []
    for kp in kps:
        rel_part.extend([(kp[0] - cx) / W, (kp[1] - cy) / H])

    # Person center (normalized coordinates relative to the court center)
    rel_cx = (cx - court_cx) / W
    rel_cy = (cy - court_cy) / H
    center_part = [rel_cx, rel_cy]

    # Velocity (inter-frame difference of court relative coordinates)
    vx = rel_cx - cx_prev if cx_prev is not None else 0.0
    vy = rel_cy - cy_prev if cy_prev is not None else 0.0

    # Acceleration
    ax = vx - vx_prev
    ay = vy - vy_prev

    # 14 court point coordinates (zeroed out when conf < 0.3, 28 dimensions total)
    court_part = []
    for i in range(14):
        if court_kps and i < len(court_kps):
            kp = court_kps[i]
            if len(kp) >= 3 and kp[2] >= 0.3:
                court_part.extend([kp[0] / W, kp[1] / H])
            else:
                court_part.extend([0.0, 0.0])
        else:
            court_part.extend([0.0, 0.0])

    vec = np.array(abs_part + rel_part + center_part + [vx, vy, ax, ay]
                   + [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # Ball: position xy + velocity xy + acceleration xy
                   + court_part,
                   dtype=np.float32)
    return vec, rel_cx, rel_cy, vx, vy


def _read_crop(path):
    raw = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if img is None:
        return np.zeros((320, 320, 3), dtype=np.uint8)
    return _resize_uint8(img, 320, 320)


class TennisActionDataset(Dataset):
    def __init__(self, cfg, clip_dirs=None, augment=False):
        self.data_root = cfg["data_root"]
        self.seq_len = cfg["seq_len"]
        self.min_seq_len = cfg.get("min_seq_len", max(30, cfg["seq_len"] // 2))
        self.use_visual = cfg.get("use_visual", True)
        self.use_player_crops = cfg.get("use_player_crops", True)
        self.augment = augment  # Enabled for training set, disabled for test set

        if clip_dirs is None:
            clip_dirs = [
                os.path.join(self.data_root, d)
                for d in os.listdir(self.data_root)
                if os.path.isdir(os.path.join(self.data_root, d))
            ]
        self.clip_dirs = clip_dirs

        self.meta_cache = {}
        self.chunks = []

        print(f"[dataset] Preloading JSON and slicing (seq_len={self.seq_len})...")
        for clip_dir in self.clip_dirs:
            pose_path = os.path.join(clip_dir, "pose_data.json")
            anno_path = os.path.join(clip_dir, "annotations.json")
            if not os.path.exists(pose_path) or not os.path.exists(anno_path):
                continue

            with open(pose_path, "r", encoding="utf-8") as f:
                pose_json = json.load(f)
            with open(anno_path, "r", encoding="utf-8") as f:
                anno_json = json.load(f)

            if isinstance(pose_json, dict):
                frame_keys = [int(k) for k in pose_json if k.isdigit()]
                total_frames = max(frame_keys) + 1 if frame_keys else 0
            else:
                total_frames = len(pose_json)

            if total_frames == 0:
                continue

            self.meta_cache[clip_dir] = {
                "pose": pose_json,
                "anno": anno_json,
                "total_frames": total_frames,
            }

        self._build_chunks()
        print(f"[dataset] Successfully generated {len(self.chunks)} slices.")

    def _build_chunks(self):
        """Fixed slicing, used for test set or initialization."""
        self.chunks = []
        for clip_dir, meta in self.meta_cache.items():
            total_frames = meta["total_frames"]
            for start in range(0, total_frames, self.seq_len):
                self.chunks.append({
                    "clip_dir": clip_dir,
                    "start_frame": start,
                    "end_frame": min(start + self.seq_len, total_frames),
                })

    def reshuffle(self):
        """Called before the start of each epoch to randomly re-slice training samples."""
        if not self.augment:
            return
        self.chunks = []
        for clip_dir, meta in self.meta_cache.items():
            total_frames = meta["total_frames"]
            if total_frames < self.min_seq_len:
                continue
            start = 0
            while start < total_frames:
                remaining = total_frames - start
                if remaining < self.min_seq_len:
                    break
                length = np.random.randint(self.min_seq_len, min(self.seq_len, remaining) + 1)
                self.chunks.append({
                    "clip_dir": clip_dir,
                    "start_frame": start,
                    "end_frame": start + length,
                })
                start += length

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        clip_dir = chunk["clip_dir"]
        start = chunk["start_frame"]
        end = chunk["end_frame"]
        actual_len = end - start

        cache = self.meta_cache[clip_dir]
        pose_json = cache["pose"]
        anno_json = cache["anno"]

        pose_tensor = torch.zeros(self.seq_len, POSE_DIM)
        labels = torch.full((self.seq_len,), -100, dtype=torch.long)
        keyframe_labels = torch.zeros(self.seq_len, dtype=torch.long)
        
        # Concat uint8 tensor: [T, 3, 320, 960] (full frame padded to 320×320 + p1 320×320 + p2 320×320)
        # Normalization is performed on the GPU side; the CPU side maintains uint8 to save PCIe bandwidth.
        if self.use_visual:
            packed_frames = torch.zeros(self.seq_len, 3, 320, 960, dtype=torch.uint8)
        else:
            packed_frames = torch.zeros(1, dtype=torch.uint8)

        frames_dir = os.path.join(clip_dir, "frames")
        use_frames_dir = self.use_visual and os.path.isdir(frames_dir)

        cap = None
        if self.use_visual and not use_frames_dir:
            short_path = _get_short_path(os.path.join(clip_dir, "raw_clip.mp4"))
            cap = cv2.VideoCapture(short_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0 or np.isnan(fps):
                fps = 30.0
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        else:
            fps = 30.0

        # Pre-calculate keyframe set (±2 frames tolerance)
        key_frames = set()
        if isinstance(anno_json, list):
            for seg in anno_json:
                for t_sec in (seg.get("start_time", -1), seg.get("end_time", -1)):
                    if t_sec < 0:
                        continue
                    fi = round(t_sec * fps)
                    for delta in (-2, -1, 0, 1, 2):
                        key_frames.add(fi + delta)

        p1_dir = os.path.join(clip_dir, "player1")
        p2_dir = os.path.join(clip_dir, "player2")
        has_crops = (self.use_visual and self.use_player_crops
                     and os.path.isdir(p1_dir) and os.path.isdir(p2_dir))

        cx_prev = cy_prev = vx_prev = vy_prev = None

        for t in range(actual_len):
            global_idx = start + t
            current_time = global_idx / fps

            # Action Label
            action_id = 0
            if isinstance(anno_json, list):
                for seg in anno_json:
                    if seg.get("start_time", 0) <= current_time <= seg.get("end_time", 0):
                        action_id = seg.get("action_id", 0)
                        break
            labels[t] = action_id

            # Keyframe Label
            keyframe_labels[t] = 1 if global_idx in key_frames else 0

            # Pose (125-dimensional)
            frame_data = (pose_json.get(str(global_idx))
                          if isinstance(pose_json, dict)
                          else (pose_json[global_idx] if global_idx < len(pose_json) else None))
            player = _parse_player(frame_data, "near_player") if frame_data else None
            court_kps = frame_data.get("court") if isinstance(frame_data, dict) else None
            vec, cx_prev, cy_prev, vx_prev, vy_prev = _build_pose_vec(
                player, cx_prev, cy_prev,
                vx_prev if vx_prev is not None else 0.0,
                vy_prev if vy_prev is not None else 0.0,
                court_kps=court_kps,
            )
            pose_tensor[t] = torch.from_numpy(vec)

            # Full Frame Visual (padded to 320×320, written into the [:, :, 0:320] slice of the packed tensor)
            if use_frames_dir:
                fp = os.path.join(frames_dir, f"{global_idx:06d}.jpg")
                if os.path.exists(fp):
                    raw = np.fromfile(fp, dtype=np.uint8)
                    img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
                    if img is not None:
                        rgb = _resize_uint8(img, 320, 320)
                        packed_frames[t, :, :, :320] = torch.from_numpy(rgb.transpose(2, 0, 1))
            elif cap is not None:
                ret, frame = cap.read()
                if ret:
                    rgb = _resize_uint8(frame, 320, 320)
                    packed_frames[t, :, :, :320] = torch.from_numpy(rgb.transpose(2, 0, 1))

            # Cropped Images (p1 written into [:, :, 320:640], p2 written into [:, :, 640:960])
            if has_crops:
                name = f"{global_idx:06d}.jpg"
                p1_path = os.path.join(p1_dir, name)
                p2_path = os.path.join(p2_dir, name)
                if os.path.exists(p1_path):
                    rgb1 = _read_crop(p1_path)
                    packed_frames[t, :, :, 320:640] = torch.from_numpy(rgb1.transpose(2, 0, 1))
                if os.path.exists(p2_path):
                    rgb2 = _read_crop(p2_path)
                    packed_frames[t, :, :, 640:960] = torch.from_numpy(rgb2.transpose(2, 0, 1))

        if cap is not None:
            cap.release()

        return pose_tensor, packed_frames, labels, keyframe_labels
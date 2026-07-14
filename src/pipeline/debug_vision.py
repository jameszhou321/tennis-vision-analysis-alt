"""debug_vision.py — Visual Debugging Tool

Function: Overlays tracking and detection results onto video frames to debug and validate pipeline outputs.
"""
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from scipy.optimize import least_squares
from collections import defaultdict, deque

# =====================================================================
# 1. Paths and Model Configurations
# =====================================================================
VIDEO_PATH = "data/rallies_annotated/rally_001_19.8s/raw_clip.mp4"
OUTPUT_PATH = r"output_god_mode.mp4"

COURT_MODEL_PATH = "runs/court_finetune/court_14pts_ultimate/weights/best.pt"
TRACKER_MODEL_PATH = r"best.pt"  # Your exclusive YOLO26x athlete tracking model
POSE_MODEL_PATH = r"yolo11x-pose.pt"

# Physical 2D coordinates of the 14 keypoints on the tennis court
COURT_PHYSICAL = np.array([
    [-5.485, -11.885], [5.485, -11.885], [5.485, 11.885], [-5.485, 11.885],
    [0.000, -11.885], [0.000, 11.885], [-4.115, -6.400], [4.115, -6.400],
    [0.000, -6.400], [-4.115, 6.400], [4.115, 6.400], [0.000, 6.400],
    [-5.485, 0.000], [5.485, 0.000]
], dtype=np.float32)

COURT_LINES = [(0, 1), (2, 3), (0, 3), (1, 2), (6, 7), (9, 10), (4, 5), (12, 13)]


# =====================================================================
# 2. Mathematical Mapping and Temporal Cleaning
# =====================================================================
def get_weighted_homography(phys_pts, pixel_pts, weights):
    """Computes weighted homography, optimizing projection residuals via the Levenberg-Marquardt (LM) algorithm."""
    H_init, _ = cv2.findHomography(phys_pts, pixel_pts, cv2.RANSAC, 5.0)
    if H_init is None: return None

    def residuals(h):
        H = np.append(h, 1.0).reshape(3, 3)
        pts_3d = np.concatenate([phys_pts, np.ones((len(phys_pts), 1))], axis=1)
        proj = (H @ pts_3d.T).T
        proj[:, :2] /= (proj[:, 2:] + 1e-8)
        return ((proj[:, :2] - pixel_pts) * weights[:, np.newaxis]).flatten()

    res = least_squares(residuals, x0=(H_init / H_init[2, 2]).flatten()[:8], method='lm')
    return np.append(res.x, 1.0).reshape(3, 3)


def interpolate_track(track_dict, max_gap=20):
    """Fills tracking gaps via linear interpolation and applies EMA smoothing while safely inheriting extra data."""
    if not track_dict: return {}
    f_indices = sorted(track_dict.keys())

    # Extract core numerical data for computations
    data = []
    for f in f_indices:
        d = track_dict[f]
        data.append([f, d['real'][0], d['real'][1], d['box'][0], d['box'][1], d['box'][2], d['box'][3]])

    df = pd.DataFrame(data, columns=['f', 'x', 'y', 'x1', 'y1', 'x2', 'y2'])
    df = df.set_index('f')

    # Reindex to create NaN rows for interpolation
    full_idx = np.arange(f_indices[0], f_indices[-1] + 1)
    df = df.reindex(full_idx)

    # Dynamically limit max_gap to prevent pandas sliding window from going out of bounds
    actual_limit = min(max_gap, len(df) - 1)
    if actual_limit > 0:
        df = df.interpolate(method='linear', limit=actual_limit)

    # EMA (Exponential Weighted Moving Average) smoothing
    df = df.ewm(alpha=0.3).mean().dropna()

    new_track = {}
    for f, row in df.iterrows():
        f_int = int(f)
        # Properly inherit skeletal keypoint metadata
        original_data = track_dict.get(f_int, {})
        original_kpts = original_data.get('keypoints', None)

        new_track[f_int] = {
            "real": np.array([row['x'], row['y']]),
            "box": [int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])],
            "keypoints": original_kpts
        }
    return new_track


# =====================================================================
# 3. Fade-Tail Radar Renderer
# =====================================================================
class RadarRenderer:
    def __init__(self, w=300, h=600):
        self.w, self.h = w, h
        self.scale = 22
        self.cx, self.cy = w // 2, h // 2
        # deque maintains up to 25 historical frames to create the motion tail effect
        self.history = defaultdict(lambda: deque(maxlen=25))

    def draw(self, frame, active_players):
        radar = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        # Radar background map and court boundary lines
        cv2.rectangle(radar, (0, 0), (self.w, self.h), (25, 45, 25), -1)
        hw, hh = 5.485 * self.scale, 11.885 * self.scale
        cv2.rectangle(radar, (int(self.cx - hw), int(self.cy - hh)), (int(self.cx + hw), int(self.cy + hh)),
                      (230, 230, 230), 2)
        cv2.line(radar, (int(self.cx - hw), self.cy), (int(self.cx + hw), self.cy), (230, 230, 230), 1)

        for tid, data in active_players.items():
            pos = data["real"]
            px, py = int(pos[0] * self.scale + self.cx), int(-pos[1] * self.scale + self.cy)
            self.history[tid].append((px, py))

            # Render the motion tail fading from dim to bright
            h_list = list(self.history[tid])
            for i in range(1, len(h_list)):
                alpha = i / len(h_list)
                color = (0, int(200 * alpha), int(255 * alpha))
                cv2.line(radar, h_list[i - 1], h_list[i], color, max(1, int(4 * alpha)))

            # Draw the current position head node
            cv2.circle(radar, (px, py), 6, (0, 255, 255), -1)

        # Alpha blending back onto the original frame view
        roi = frame[20:20 + self.h, 20:20 + self.w]
        cv2.addWeighted(radar, 0.8, roi, 0.2, 0, roi)
        return frame


# =====================================================================
# 4. Main System Pipeline
# =====================================================================
def main():
    print("⏳ Loading model weights...")
    court_model = YOLO(COURT_MODEL_PATH)
    tracker_model = YOLO(TRACKER_MODEL_PATH)
    pose_model = YOLO(POSE_MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    fps, width, height = int(cap.get(5)), int(cap.get(3)), int(cap.get(4))

    tracks_db = defaultdict(dict)
    h_db = {}
    f_idx = 0
    last_H = None

    print("\n[Phase 1/3] Extracting visual features (Top-Down execution active)...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # --- 1. Tennis Court Detection ---
        c_res = court_model.predict(frame, conf=0.3, verbose=False)[0]
        if c_res.keypoints is not None and len(c_res.keypoints.data) > 0:
            kpts = c_res.keypoints.data[0].cpu().numpy()
            if len(kpts) >= 4:
                mask = kpts[:, 2] > 0.4
                if np.sum(mask) >= 4:
                    last_H = get_weighted_homography(COURT_PHYSICAL[mask], kpts[mask, :2], kpts[mask, 2])
        h_db[f_idx] = last_H

        # --- 2. Athlete Tracking and Fallback Foot Localization ---
        if last_H is not None:
            t_res = tracker_model.track(frame, persist=True, tracker="botsort.yaml", verbose=False)[0]
            if t_res.boxes is not None and t_res.boxes.id is not None:
                ids = t_res.boxes.id.int().cpu().tolist()
                bboxes = t_res.boxes.xyxy.cpu().numpy()
                H_inv = np.linalg.inv(last_H)

                for i, tid in enumerate(ids):
                    bx = bboxes[i].astype(int)

                    # Expand Bbox by 50% to increase the receptive field for the pose model
                    bw, bh = bx[2] - bx[0], bx[3] - bx[1]
                    pad_x = int(bw * 0.25) + 10
                    pad_y = int(bh * 0.25) + 10

                    cx1, cy1 = max(0, bx[0] - pad_x), max(0, bx[1] - pad_y)
                    cx2, cy2 = min(width, bx[2] + pad_x), min(height, bx[3] + pad_y)
                    athlete_crop = frame[cy1:cy2, cx1:cx2]

                    feet_px = np.array([(bx[0] + bx[2]) / 2, bx[3]])  # Fallback anchor: bottom-center of the box
                    kpts_global = None

                    # Whenever the crop patch remains valid, pass it into the Pose model for details extraction
                    if athlete_crop.shape[0] >= 10 and athlete_crop.shape[1] >= 10:
                        p_res = pose_model.predict(athlete_crop, imgsz=192, verbose=False)[0]
                        if p_res.keypoints is not None and len(p_res.keypoints.data) > 0:
                            # Clone array and transform coordinates straight back into global scene perspective
                            kp = p_res.keypoints.data[0].cpu().numpy().copy()
                            kp[:, 0] += cx1
                            kp[:, 1] += cy1
                            kpts_global = kp

                            # Hierarchical fallback foot localization logic
                            l_ankle, r_ankle = kp[15], kp[16]
                            l_knee, r_knee = kp[13], kp[14]

                            if l_ankle[2] > 0.35 and r_ankle[2] > 0.35:  # Both ankles visible
                                feet_px = (l_ankle[:2] + r_ankle[:2]) / 2.0
                            elif l_ankle[2] > 0.35:  # Left ankle only
                                feet_px = l_ankle[:2]
                            elif r_ankle[2] > 0.35:  # Right ankle only
                                feet_px = r_ankle[:2]
                            elif l_knee[2] > 0.4 and r_knee[2] > 0.4:  # Knees fallback (e.g., net occlusion)
                                feet_px = (l_knee[:2] + r_knee[:2]) / 2.0

                    # Map coordinates back into physical 2D plane geometry
                    pt = np.array([[[feet_px[0], feet_px[1]]]], dtype=np.float32)
                    real_pos = cv2.perspectiveTransform(pt, H_inv)[0][0]
                    tracks_db[tid][f_idx] = {"real": real_pos, "box": bx, "keypoints": kpts_global}

        f_idx += 1
        if f_idx % 100 == 0: print(f"   Processed {f_idx} frames...")

    print("\n[Phase 2/3] Performing temporal interpolation compensation and data cleaning...")
    final_tracks = {}
    for tid in tracks_db.keys():
        processed = interpolate_track(tracks_db[tid])
        if len(processed) > 10: final_tracks[tid] = processed

    # Filter out the two IDs with the longest lifespan tracking metrics as P1 and P2
    id_scores = {tid: len(data) for tid, data in final_tracks.items()}
    top_2_ids = sorted(id_scores, key=id_scores.get, reverse=True)[:2]

    print(f"\n[Phase 3/3] Rendering and writing target video stream (Locked IDs: {top_2_ids})...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    writer = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    radar = RadarRenderer()

    for f in range(f_idx):
        ret, frame = cap.read()
        if not ret: break

        # Draw Court Markings
        if h_db[f] is not None:
            H = h_db[f]
            for l in COURT_LINES:
                p1 = np.append(COURT_PHYSICAL[l[0]], 1.0)
                p2 = np.append(COURT_PHYSICAL[l[1]], 1.0)
                px1 = (H @ p1)
                px1 /= px1[2]
                px2 = (H @ p2)
                px2 /= px2[2]
                cv2.line(frame, (int(px1[0]), int(px1[1])), (int(px2[0]), int(px2[1])), (0, 255, 0), 2)

        # Draw Athletes and Pose Skeleton Overlay
        active_this_frame = {}
        for tid in top_2_ids:
            if f in final_tracks[tid]:
                d = final_tracks[tid][f]
                active_this_frame[tid] = d

                # Draw tracking bounding box
                b = d["box"]
                if b is not None:
                    cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 255, 255), 2)
                    cv2.putText(frame, f"ID {tid}", (b[0], b[1] - 10), 0, 0.6, (0, 255, 255), 2)

                # Draw skeletal joint points
                kpts = d.get("keypoints")
                if kpts is not None:
                    for kp in kpts:
                        if kp[2] > 0.4:
                            cv2.circle(frame, (int(kp[0]), int(kp[1])), 4, (0, 0, 255), -1)

        # Overlay Radar View
        frame = radar.draw(frame, active_this_frame)

        writer.write(frame)
        cv2.imshow("God Mode Ultimate Tracker", frame)
        if cv2.waitKey(1) == ord('q'): break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"\nCore graduation project module completed successfully! Output saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
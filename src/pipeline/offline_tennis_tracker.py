"""offline_tennis_tracker.py — Offline Tennis Tracking Main Module

Function: Read rally videos, use court keypoint models + pose models, and output videos annotated with tracking details.
"""
import os as _os
import cv2
import numpy as np
from collections import defaultdict
from scipy.optimize import least_squares
from ultralytics import YOLO
_SRC_DIR = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))).replace('\\', '/')
_PROJECT_DIR = _os.path.dirname(_SRC_DIR).replace('\\', '/')

# =====================================================================
# 1. Global Configuration Area
# =====================================================================
VIDEO_PATH = f"{_PROJECT_DIR}/data/rallies_annotated/rally_001_19.8s/raw_clip.mp4"
COURT_MODEL_PATH = f"{_PROJECT_DIR}/runs/court_finetune/court_14pts_ultimate/weights/best.pt"
POSE_MODEL_PATH = f"{_PROJECT_DIR}/models/yolo/yolo11x-pose.pt"  # Or yolo26x-pose.pt
OUTPUT_PATH = f"{_PROJECT_DIR}/results/output_offline_tracker.mp4"

COURT_14_PTS_PHYSICAL = np.array([
    [-5.485, -11.885], [5.485, -11.885], [5.485, 11.885], [-5.485, 11.885],
    [0.000, -11.885], [0.000, 11.885],
    [-4.115, -6.400], [4.115, -6.400], [0.000, -6.400],
    [-4.115, 6.400], [4.115, 6.400], [0.000, 6.400],
    [-5.485, 0.000], [5.485, 0.000]
], dtype=np.float32)

COURT_LINES_PHYSICAL = [
    ([-5.485, -11.885], [5.485, -11.885]), ([-5.485, 11.885], [5.485, 11.885]),
    ([-5.485, -11.885], [-5.485, 11.885]), ([5.485, -11.885], [5.485, 11.885]),
    ([-4.115, -11.885], [-4.115, 11.885]), ([4.115, -11.885], [4.115, 11.885]),
    ([-4.115, -6.400], [4.115, -6.400]), ([-4.115, 6.400], [4.115, 6.400]),
    ([0.000, -6.400], [0.000, 6.400]), ([-5.485, 0.000], [5.485, 0.000])
]

BASE_WEIGHTS = np.array([7, 7, 7, 7, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], dtype=np.float32)


# =====================================================================
# 2. Core Algorithm Library
# =====================================================================
def reprojection_residuals(h_elements, src_pts, dst_pts, weights):
    H = np.append(h_elements, 1.0).reshape(3, 3)
    src_pts_3d = np.concatenate([src_pts, np.ones((len(src_pts), 1))], axis=1)
    proj_pts_3d = (H @ src_pts_3d.T).T
    proj_pts_3d[:, 2] = np.where(proj_pts_3d[:, 2] == 0, 1e-7, proj_pts_3d[:, 2])
    return (((proj_pts_3d[:, :2] / proj_pts_3d[:, 2:]) - dst_pts) * weights[:, np.newaxis]).flatten()


def get_weighted_homography(phys_pts, pixel_pts, weights):
    H_init, _ = cv2.findHomography(phys_pts, pixel_pts, cv2.RANSAC, 5.0)
    if H_init is None: return None
    res = least_squares(reprojection_residuals, x0=(H_init / H_init[2, 2]).flatten()[:8],
                        args=(phys_pts, pixel_pts, weights), method='lm')
    return np.append(res.x, 1.0).reshape(3, 3)


class HomographyFilter:
    def __init__(self):
        self.history = []

    def update(self, new_H):
        if new_H is None: return None
        self.history.append(new_H)
        if len(self.history) > 5: self.history.pop(0)
        s_H = np.mean(self.history, axis=0)
        return s_H / s_H[2, 2]


# =====================================================================
# 3. Offline Scoring System (Pure Score Version)
# =====================================================================
def score_and_select_players(tracks_db):
    TOP_BASELINE, BOTTOM_BASELINE = np.array([0.0, 11.885]), np.array([0.0, -11.885])
    scored_tracks = []

    for t_id, frame_data in tracks_db.items():
        coords = np.array([pt["real"] for pt in frame_data.values()])
        avg_y = np.mean(coords[:, 1])
        is_top = avg_y > 0
        anchor = TOP_BASELINE if is_top else BOTTOM_BASELINE

        # Accumulate score points purely based on the distance to the baseline, disregarding trajectory lifetime duration
        distances = np.linalg.norm(coords - anchor, axis=1)
        # Valid within a 10-meter radius; the closer, the higher the score
        total_score = np.sum(np.maximum(0, 10.0 - distances))

        scored_tracks.append({"id": t_id, "side": "top" if is_top else "bottom", "score": total_score})

    # Sort strictly by the highest score
    top_cands = sorted([t for t in scored_tracks if t["side"] == "top"], key=lambda x: x["score"], reverse=True)
    bot_cands = sorted([t for t in scored_tracks if t["side"] == "bottom"], key=lambda x: x["score"], reverse=True)

    res = {}
    if top_cands:
        res["top"] = top_cands[0]["id"]
        print(f"Top half winner -> ID: {res['top']} (Total Score: {top_cands[0]['score']:.1f})")
    if bot_cands:
        res["bottom"] = bot_cands[0]["id"]
        print(f"Bottom half winner -> ID: {res['bottom']} (Total Score: {bot_cands[0]['score']:.1f})")
    return res


# =====================================================================
# 4. Radar UI Renderer
# =====================================================================
class RadarDrawer:
    def __init__(self):
        self.scale = 12
        self.cx, self.cy = int(10.97 * self.scale), int(23.77 * self.scale * 0.75)
        self.w, self.h = self.cx * 2, self.cy * 2

    def draw(self, frame, top_pt, bot_pt, top_trail, bot_trail):
        overlay = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        cv2.rectangle(overlay, (0, 0), (self.w, self.h), (80, 120, 80), -1)
        sx, sy = int(5.485 * self.scale), int(11.885 * self.scale)
        cv2.rectangle(overlay, (self.cx - sx, self.cy - sy), (self.cx + sx, self.cy + sy), (255, 255, 255), 2)
        cv2.line(overlay, (self.cx - sx, self.cy), (self.cx + sx, self.cy), (255, 255, 255), 2)

        def draw_trail(trail, color_bgr):
            for i, p in enumerate(trail):
                alpha = (i + 1) / len(trail)
                c = (int(color_bgr[0] * alpha), int(color_bgr[1] * alpha), int(color_bgr[2] * alpha))
                cv2.circle(overlay, (int(p[0] * self.scale) + self.cx, int(-p[1] * self.scale) + self.cy), 2, c, -1)

        draw_trail(top_trail, (255, 150, 50))
        draw_trail(bot_trail, (50, 150, 255))

        if top_pt is not None:
            cv2.circle(overlay, (int(top_pt[0] * self.scale) + self.cx, int(-top_pt[1] * self.scale) + self.cy), 6,
                       (255, 0, 0), -1)
        if bot_pt is not None:
            cv2.circle(overlay, (int(bot_pt[0] * self.scale) + self.cx, int(-bot_pt[1] * self.scale) + self.cy), 6,
                       (0, 0, 255), -1)

        alpha = 0.7
        roi = frame[20:20 + self.h, 20:20 + self.w]
        cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)
        return frame


# =====================================================================
# 5. Main Control System
# =====================================================================
def main():
    court_model = YOLO(COURT_MODEL_PATH)
    pose_model = YOLO(POSE_MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tracks_db = defaultdict(dict)
    h_db = {}
    h_filter = HomographyFilter()

    print("\n[Pass 1] Extracting features (no display window)...")
    frame_idx = 0
    prev_H = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        court_res = court_model.predict(frame, conf=0.3, half=True, verbose=False)[0]
        v_px, v_ph, v_w = [], [], []
        if court_res.keypoints is not None and len(court_res.keypoints.data) > 0:
            for i, (x, y, conf) in enumerate(court_res.keypoints.data[0].cpu().numpy()):
                if conf > 0.4:
                    v_px.append([x, y])
                    v_ph.append(COURT_14_PTS_PHYSICAL[i])
                    v_w.append(BASE_WEIGHTS[i] * conf)
            if len(v_px) >= 4:
                raw_H = get_weighted_homography(np.array(v_ph, dtype=np.float32), np.array(v_px, dtype=np.float32),
                                                np.array(v_w, dtype=np.float32))
                if raw_H is not None: prev_H = h_filter.update(raw_H)

        h_db[frame_idx] = prev_H

        if prev_H is not None:
            # Use occlusion-resistant BoT-SORT and scale up imgsz
            pose_res = pose_model.track(frame, persist=True, tracker="botsort.yaml", imgsz=1280, verbose=False)[0]
            if pose_res.boxes is not None and pose_res.boxes.id is not None:
                track_ids = pose_res.boxes.id.int().cpu().tolist()
                poses = pose_res.keypoints.data.cpu().numpy()
                boxes = pose_res.boxes.xyxy.cpu().numpy()

                H_inv = np.linalg.inv(prev_H)

                for i, t_id in enumerate(track_ids):
                    l_foot, r_foot = poses[i][15], poses[i][16]
                    box = boxes[i]

                    # When ankle keypoint confidence is low, fall back to using the bottom-center of the bounding box as the placement point
                    if l_foot[2] > 0.2 and r_foot[2] > 0.2:
                        feet_px = (l_foot[:2] + r_foot[:2]) / 2.0
                    else:
                        # Grab the center of the bounding box bottom edge as the feet position
                        feet_px = np.array([(box[0] + box[2]) / 2.0, box[3]])

                    pt = np.array([[[feet_px[0], feet_px[1]]]], dtype=np.float32)
                    real_coord = cv2.perspectiveTransform(pt, H_inv)[0][0]

                    tracks_db[t_id][frame_idx] = {"real": real_coord, "box": box}
        frame_idx += 1

    final_ids = score_and_select_players(tracks_db)
    id_top, id_bot = final_ids.get("top"), final_ids.get("bottom")

    print("\n[Pass 2] Rendering display...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    radar = RadarDrawer()

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        top_data = tracks_db.get(id_top, {}).get(frame_idx)
        bot_data = tracks_db.get(id_bot, {}).get(frame_idx)

        if h_db.get(frame_idx) is not None:
            H = h_db[frame_idx]
            for line_meters in COURT_LINES_PHYSICAL:
                pts_meters = np.array([line_meters[0], line_meters[1]], dtype=np.float32).reshape(-1, 1, 2)
                pts_transformed = cv2.perspectiveTransform(pts_meters, H)
                pt1 = (int(pts_transformed[0][0][0]), int(pts_transformed[0][0][1]))
                pt2 = (int(pts_transformed[1][0][0]), int(pts_transformed[1][0][1]))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2, cv2.LINE_AA)

        if top_data:
            x1, y1, x2, y2 = top_data["box"]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 100, 50), 2)
            cv2.putText(frame, f"P1 ({id_top})", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 50),
                        2)

        if bot_data:
            x1, y1, x2, y2 = bot_data["box"]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (50, 100, 255), 2)
            cv2.putText(frame, f"P2 ({id_bot})", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 100, 255),
                        2)

        top_trail = [tracks_db[id_top][i]["real"] for i in range(max(0, frame_idx - 10), frame_idx) if
                     i in tracks_db.get(id_top, {})]
        bot_trail = [tracks_db[id_bot][i]["real"] for i in range(max(0, frame_idx - 10), frame_idx) if
                     i in tracks_db.get(id_bot, {})]
        frame = radar.draw(frame, top_data["real"] if top_data else None, bot_data["real"] if bot_data else None,
                           top_trail, bot_trail)

        cv2.imshow("Tennis God Mode", frame)
        out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        frame_idx += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Rendering complete! Video saved.")


if __name__ == "__main__":
    main()
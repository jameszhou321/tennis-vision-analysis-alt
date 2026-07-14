translate all chinese documentation into english: """generate_trajectory.py — Player Movement Trajectory Generation Module

Function: Extract player coordinate sequences from tracking results to generate trajectory features for action recognition.
"""
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from scipy.optimize import least_squares

# =====================================================================
# 1. Global Configuration Area
# =====================================================================
VIDEO_PATH = "your_video.mp4"  # [MODIFY HERE] Test video path
COURT_MODEL_PATH = "court_model.pt"  # [MODIFY HERE] 14-point court model path
POSE_MODEL_PATH = "yolo11x-pose.pt"  # [MODIFY HERE] Body pose model path
OUTPUT_PATH = "output_radar_ultimate.mp4"  # Output video path

# Physical Coordinates Library (14 keypoints, with the net center as the origin)
COURT_14_PTS_PHYSICAL = np.array([
    [-5.485, -11.885], [5.485, -11.885], [5.485, 11.885], [-5.485, 11.885],  # 0-3: Core outer corner points
    [0.000, -11.885], [0.000, 11.885],  # 4-5: Baseline midpoints
    [-4.115, -6.400], [4.115, -6.400], [0.000, -6.400],  # Intersections of service line with singles sideline/centerline
    [-4.115, 6.400], [4.115, 6.400], [0.000, 6.400],  # Intersections of service line with singles sideline/centerline
    [-5.485, 0.000], [5.485, 0.000]  # Intersections of net with doubles sideline
], dtype=np.float32)

# Prior Weight System: Core corners have stronger pulling force, internal segments have weaker pulling force
BASE_WEIGHTS = np.array([7, 7, 7, 7, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], dtype=np.float32)


# =====================================================================
# 2. Mathematical Optimization Engine: SciPy Weighted Least Squares (L-M)
# =====================================================================
def reprojection_residuals(h_elements, src_pts, dst_pts, weights):
    """Calculate the residual function of weighted projection errors"""
    H = np.append(h_elements, 1.0).reshape(3, 3)
    src_pts_3d = np.concatenate([src_pts, np.ones((len(src_pts), 1))], axis=1)
    proj_pts_3d = (H @ src_pts_3d.T).T

    # Prevent division by zero
    proj_pts_3d[:, 2] = np.where(proj_pts_3d[:, 2] == 0, 1e-7, proj_pts_3d[:, 2])
    proj_pts_2d = proj_pts_3d[:, :2] / proj_pts_3d[:, 2:]

    errors = proj_pts_2d - dst_pts
    weighted_errors = errors * weights[:, np.newaxis]
    return weighted_errors.flatten()


def get_weighted_homography(phys_pts, pixel_pts, weights):
    """Use the weighted L-M algorithm to compute the precise homography matrix"""
    # First compute a rough initial value using OpenCV
    H_init, _ = cv2.findHomography(phys_pts, pixel_pts, cv2.RANSAC, 5.0)
    if H_init is None: return None

    h_initial_guess = (H_init / H_init[2, 2]).flatten()[:8]
    res = least_squares(
        reprojection_residuals, x0=h_initial_guess,
        args=(phys_pts, pixel_pts, weights), method='lm'
    )
    return np.append(res.x, 1.0).reshape(3, 3)


# =====================================================================
# 3. Dual Sliding Window Filter (Matrix + Trajectory)
# =====================================================================
class HomographyFilter:
    """First-layer filtering: Sliding window smoother for the court homography matrix H"""

    def __init__(self, window_size=5):
        self.window_size = window_size
        self.h_history = []

    def update(self, new_H):
        if new_H is None:
            return None
        self.h_history.append(new_H)
        if len(self.h_history) > self.window_size:
            self.h_history.pop(0)

        smoothed_H = np.mean(self.h_history, axis=0)
        smoothed_H = smoothed_H / smoothed_H[2, 2]  # Re-normalize
        return smoothed_H


class RadarDrawer:
    """Second-layer filtering: Smoother and renderer for 2D red dot coordinates"""

    def __init__(self, window_size=6):
        self.history = []
        self.window_size = window_size
        self.trail = []

        # Radar UI settings (Scale up by 12 times for drawing)
        self.map_scale = 12
        self.map_w = int(10.97 * self.map_scale * 2)
        self.map_h = int(23.77 * self.map_scale * 1.5)
        self.center_x = self.map_w // 2
        self.center_y = self.map_h // 2

    def smooth_point(self, real_coord):
        """Sliding average of trajectory points"""
        self.history.append(real_coord)
        if len(self.history) > self.window_size:
            self.history.pop(0)
        return np.mean(self.history, axis=0)

    def draw_minimap(self, frame, current_real_coord):
        """Render the UI layer"""
        overlay = np.zeros((self.map_h, self.map_w, 3), dtype=np.uint8)
        # Draw the green court background
        cv2.rectangle(overlay, (0, 0), (self.map_w, self.map_h), (80, 120, 80), -1)

        # Draw outer boundary and the net simply
        scale_x, scale_y = int(5.485 * self.map_scale), int(11.885 * self.map_scale)
        top_left = (self.center_x - scale_x, self.center_y - scale_y)
        bottom_right = (self.center_x + scale_x, self.center_y + scale_y)
        cv2.rectangle(overlay, top_left, bottom_right, (255, 255, 255), 2)
        cv2.line(overlay, (top_left[0], self.center_y), (bottom_right[0], self.center_y), (255, 255, 255), 2)

        # Trail fading logic
        if current_real_coord is not None:
            self.trail.append({'x': current_real_coord[0], 'y': current_real_coord[1], 'alpha': 1.0})

        for p in self.trail: p['alpha'] -= 0.03
        self.trail = [p for p in self.trail if p['alpha'] > 0]

        # Render trajectory
        for p in self.trail:
            draw_x = int(p['x'] * self.map_scale) + self.center_x
            draw_y = int(-p['y'] * self.map_scale) + self.center_y  # Invert Y-axis to match bird's-eye view intuition
            color = (0, int(255 * p['alpha']), int(255 * p['alpha']))  # BGR Yellow trail
            cv2.circle(overlay, (draw_x, draw_y), 3, color, -1)

        # Draw a large red dot at the latest position
        if self.trail:
            latest = self.trail[-1]
            draw_x = int(latest['x'] * self.map_scale) + self.center_x
            draw_y = int(-latest['y'] * self.map_scale) + self.center_y
            cv2.circle(overlay, (draw_x, draw_y), 6, (0, 0, 255), -1)

        # Overlay blending
        alpha = 0.7
        roi = frame[20:20 + self.map_h, 20:20 + self.map_w]
        cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, roi)
        return frame


# =====================================================================
# 4. Main Control Engine
# =====================================================================
def main():
    print("⏳ Loading YOLO vision models...")
    court_model = YOLO(COURT_MODEL_PATH)
    pose_model = YOLO(POSE_MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # Initialize dual filters
    radar = RadarDrawer(window_size=6)
    h_filter = HomographyFilter(window_size=5)

    prev_smoothed_H = None

    print("Starting trajectory generation...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # ---------------------------------------------------------
        # [A] Court Line Detection and Dynamic Matrix Update
        # ---------------------------------------------------------
        court_res = court_model.predict(frame, conf=0.3, half=True, verbose=False)[0]

        valid_pixel_pts, valid_phys_pts, valid_weights = [], [], []
        corner_count = 0

        if court_res.keypoints is not None and len(court_res.keypoints.data) > 0:
            kpts = court_res.keypoints.data[0].cpu().numpy()
            for i, pt in enumerate(kpts):
                x, y, conf = pt
                if conf > 0.4:
                    valid_pixel_pts.append([x, y])
                    valid_phys_pts.append(COURT_14_PTS_PHYSICAL[i])
                    valid_weights.append(BASE_WEIGHTS[i] * conf)
                    if i < 4: corner_count += 1

            # Compute new H matrix only when there are enough valid points
            if corner_count >= 2 and len(valid_pixel_pts) >= 4:
                raw_H = get_weighted_homography(
                    np.array(valid_phys_pts, dtype=np.float32),
                    np.array(valid_pixel_pts, dtype=np.float32),
                    np.array(valid_weights, dtype=np.float32)
                )

                # Core filtering: Pass the raw matrix into the filter to get the smoothed matrix
                if raw_H is not None:
                    prev_smoothed_H = h_filter.update(raw_H)

        # ---------------------------------------------------------
        # [B] Human Pose and Mapping
        # ---------------------------------------------------------
        real_coord_smoothed = None
        if prev_smoothed_H is not None:
            pose_res = pose_model.predict(frame, verbose=False)[0]

            if pose_res.keypoints is not None and len(pose_res.keypoints.data) > 0:
                poses = pose_res.keypoints.data.cpu().numpy()

                # [Placeholder]: Currently using a heuristic rule of maximum Y coordinate to lock onto the player
                # When you work on the "designated player" logic, focus on modifying the best_person filtering criteria here
                best_person = None
                max_y = -1

                for pose in poses:
                    l_foot, r_foot = pose[15], pose[16]
                    if l_foot[2] > 0.3 and r_foot[2] > 0.3:
                        avg_y = (l_foot[1] + r_foot[1]) / 2
                        if avg_y > max_y:
                            max_y = avg_y
                            best_person = pose

                if best_person is not None:
                    feet_center_px = (best_person[15][:2] + best_person[16][:2]) / 2.0
                    pt = np.array([[[feet_center_px[0], feet_center_px[1]]]], dtype=np.float32)

                    # Transform coordinates using the smoothed matrix
                    real_coord = cv2.perspectiveTransform(pt, prev_smoothed_H)[0][0]

                    # Smooth the coordinate point using sliding window
                    real_coord_smoothed = radar.smooth_point(real_coord)

        # ---------------------------------------------------------
        # [C] Radar Rendering
        # ---------------------------------------------------------
        frame = radar.draw_minimap(frame, real_coord_smoothed)
        cv2.imshow("Tennis Pro Radar (Ultimate)", frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Ultimate rendering complete! Video saved.")


if __name__ == "__main__":
    main()
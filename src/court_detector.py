"""court_detector.py — Court Detector (for use by main.py)

Function: Detects the tennis court's 14 standard keypoints per frame with a YOLO
keypoint model and fits a weighted pixel<->real-world (meters) homography from them.

This is what pose_tracker.py uses to project player positions into real-world court
coordinates, which is what lets it tell an actual far-side player apart from a chair
umpire/line judge standing near the net -- see PoseTracker.select_players for why.

Replaces the older Hough-line "ROI split" approach. That approach worked entirely in
pixel space: the far-side ROI necessarily extended down to the net (to keep a far
player near their own baseline in frame), and the scoring in pose_tracker.py rewarded
candidates near the *bottom* of that ROI to avoid picking up crowd/stands near the
top -- which is exactly backwards for a stationary official who always sits at the
bottom of that box. Real-world coordinates sidestep this: a court-side official is
simply too far from either baseline to ever win the far/near slot, without needing a
referee-specific exception, and it's robust to camera zoom/angle changes since
nothing is measured in raw pixels.
"""
import cv2
import numpy as np
from scipy.optimize import least_squares

import config_legacy as config

# Standard physical coordinates (meters, net center = origin) for the court model's 14
# keypoints, in the index order the model was trained/annotated with. Same table as
# src/pipeline/offline_tennis_tracker.py / generate_trajectory.py.
COURT_14_PTS_PHYSICAL = np.array([
    [-5.485, -11.885], [5.485, -11.885], [5.485, 11.885], [-5.485, 11.885],
    [0.000, -11.885], [0.000, 11.885],
    [-4.115, -6.400], [4.115, -6.400], [0.000, -6.400],
    [-4.115, 6.400], [4.115, 6.400], [0.000, 6.400],
    [-5.485, 0.000], [5.485, 0.000]
], dtype=np.float32)

# Physical line segments, for optional debug/visual overlay of the projected court lines.
COURT_LINES_PHYSICAL = [
    ([-5.485, -11.885], [5.485, -11.885]), ([-5.485, 11.885], [5.485, 11.885]),
    ([-5.485, -11.885], [-5.485, 11.885]), ([5.485, -11.885], [5.485, 11.885]),
    ([-4.115, -11.885], [-4.115, 11.885]), ([4.115, -11.885], [4.115, 11.885]),
    ([-4.115, -6.400], [4.115, -6.400]), ([-4.115, 6.400], [4.115, 6.400]),
    ([0.000, -6.400], [0.000, 6.400]), ([-5.485, 0.000], [5.485, 0.000])
]

# Corner/T-points weighted higher than net-strap/service-line points: they anchor the
# homography's extremities and tend to be detected more reliably.
BASE_WEIGHTS = np.array([7, 7, 7, 7, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], dtype=np.float32)


def _reprojection_residuals(h_elements, src_pts, dst_pts, weights):
    H = np.append(h_elements, 1.0).reshape(3, 3)
    src_pts_3d = np.concatenate([src_pts, np.ones((len(src_pts), 1))], axis=1)
    proj_pts_3d = (H @ src_pts_3d.T).T
    proj_pts_3d[:, 2] = np.where(proj_pts_3d[:, 2] == 0, 1e-7, proj_pts_3d[:, 2])
    return (((proj_pts_3d[:, :2] / proj_pts_3d[:, 2:]) - dst_pts) * weights[:, np.newaxis]).flatten()


def get_weighted_homography(phys_pts, pixel_pts, weights):
    """RANSAC seed + Levenberg-Marquardt refinement, weighted by keypoint confidence and
    structural importance (BASE_WEIGHTS). Returns None if too few points to fit."""
    H_init, _ = cv2.findHomography(phys_pts, pixel_pts, cv2.RANSAC, 5.0)
    if H_init is None:
        return None
    res = least_squares(_reprojection_residuals, x0=(H_init / H_init[2, 2]).flatten()[:8],
                         args=(phys_pts, pixel_pts, weights), method='lm')
    return np.append(res.x, 1.0).reshape(3, 3)


class HomographyFilter:
    """Smooths the homography matrix across frames (rolling mean) so a single noisy
    keypoint detection doesn't cause a jump in projected player coordinates."""

    def __init__(self, history_len=None):
        self.history_len = history_len or config.HOMOGRAPHY_HISTORY
        self.history = []

    def update(self, new_H):
        if new_H is None:
            return None
        self.history.append(new_H)
        if len(self.history) > self.history_len:
            self.history.pop(0)
        s_H = np.mean(self.history, axis=0)
        return s_H / s_H[2, 2]


class CourtDetector:
    def __init__(self, model):
        """model: a pre-loaded YOLO court-keypoint model (see config.COURT_MODEL_PATH).
        Loading is left to the caller (main.py lazy-loads it once, same pattern as the
        pose model) so a new CourtDetector per clip doesn't reload weights each time."""
        self.model = model
        self.filter = HomographyFilter()
        self._last_H = None

    def estimate_homography(self, frame):
        """Detects the 14 court keypoints in `frame` and returns a temporally-smoothed
        pixel->real-world (meters) homography H, or None if the court has never been
        detected yet this clip.

        If this particular frame's court isn't detected (e.g. a broadcast close-up or
        replay), the last known smoothed H is returned unchanged rather than dropping
        straight to None -- matches the "gaps degrade gracefully" behavior the rest of
        this pipeline already relies on, instead of losing player tracking on every
        single missed frame.
        """
        res = self.model.predict(frame, conf=0.3, verbose=False)[0]

        if res.keypoints is not None and len(res.keypoints.data) > 0:
            v_px, v_ph, v_w = [], [], []
            for i, (x, y, conf) in enumerate(res.keypoints.data[0].cpu().numpy()):
                if conf > config.COURT_KPT_CONF:
                    v_px.append([x, y])
                    v_ph.append(COURT_14_PTS_PHYSICAL[i])
                    v_w.append(BASE_WEIGHTS[i] * conf)

            if len(v_px) >= 4:
                raw_H = get_weighted_homography(np.array(v_ph, dtype=np.float32),
                                                 np.array(v_px, dtype=np.float32),
                                                 np.array(v_w, dtype=np.float32))
                if raw_H is not None:
                    self._last_H = self.filter.update(raw_H)

        return self._last_H

    def draw_lines(self, frame, H, color=(0, 255, 0)):
        """Optional debug overlay: projects the physical court lines through H onto frame.
        Handy for visually sanity-checking a fitted homography while tuning."""
        if H is None:
            return
        for p1, p2 in COURT_LINES_PHYSICAL:
            pts = np.array([p1, p2], dtype=np.float32).reshape(-1, 1, 2)
            proj = cv2.perspectiveTransform(pts, H)
            pt1 = (int(proj[0][0][0]), int(proj[0][0][1]))
            pt2 = (int(proj[1][0][0]), int(proj[1][0][1]))
            cv2.line(frame, pt1, pt2, color, 1, cv2.LINE_AA)
"""ball_tracker.py — Lightweight Ball Tracker (classical CV, no pretrained weights required)

Function: Detects the tennis ball frame-to-frame using background subtraction + circular-blob
filtering, smooths the trajectory with a constant-velocity Kalman filter, and exposes both a
screen position (for drawing a trail) and a normalized `ball_activity_score` describing how
fast/confidently the ball is currently moving in flight.

INTENDED USE: `ball_activity_score` is designed as a future third fusion signal alongside audio
impact detection and player-motion score for rally/not-rally segmentation (see the audio-video
fusion project doc). That fusion/state-machine layer doesn't exist in main.py yet — for now this
module is wired into `annotate_rally_clip()` to draw a ball trail on already-cut rally clips.

NOTE ON ACCURACY: this is classical computer vision (MOG2 background subtraction + contour
filtering), not a learned model. It works reasonably well on **static, fixed-camera** footage
where the background is stable frame-to-frame. On broadcast footage with camera pans/zooms/cuts,
background subtraction is far less reliable, since the "background" itself is constantly moving —
expect more dropped/false detections there. The standard higher-accuracy approach in tennis-vision
research is a trained model such as TrackNet, which needs pretrained weights or training data this
environment doesn't have access to. Swap TrackNet in later behind the same `update()` interface
without touching any calling code.
"""
import cv2
import numpy as np
from collections import deque


class BallTracker:
    def __init__(self, trail_length=15, min_radius=2, max_radius=12):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=120, varThreshold=25, detectShadows=False)
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.trail = deque(maxlen=trail_length)

        # Constant-velocity Kalman filter: state = [x, y, vx, vy], measurement = [x, y]
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5

        self._initialized = False
        self.miss_count = 0
        self.max_miss = 10  # frames to coast on prediction alone before giving up the track

    def _detect_candidates(self, frame):
        """Returns a list of (x, y, radius) candidate ball blobs from foreground motion."""
        fg_mask = self.bg_subtractor.apply(frame)
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for c in contours:
            (x, y), radius = cv2.minEnclosingCircle(c)
            if self.min_radius <= radius <= self.max_radius:
                area = cv2.contourArea(c)
                circle_area = np.pi * radius * radius
                if circle_area > 0 and (area / circle_area) > 0.5:  # roughly circular, filters out limbs/rackets
                    candidates.append((x, y, radius))
        return candidates

    def update(self, frame):
        """Processes one frame. Returns {"position": (x,y) or None, "speed": float, "ball_activity_score": float}."""
        candidates = self._detect_candidates(frame)

        measurement = None
        if candidates:
            if self._initialized:
                pred = self.kalman.predict()
                px, py = float(pred[0]), float(pred[1])
                candidates.sort(key=lambda c: (c[0] - px) ** 2 + (c[1] - py) ** 2)
            else:
                # No prior track yet — just take the first plausible candidate.
                pass
            best = candidates[0]
            measurement = np.array([[np.float32(best[0])], [np.float32(best[1])]])

        estimate = None
        if measurement is not None:
            if not self._initialized:
                self.kalman.statePre = np.array([[measurement[0, 0]], [measurement[1, 0]], [0], [0]], dtype=np.float32)
                self.kalman.statePost = self.kalman.statePre.copy()
                self._initialized = True
            self.kalman.predict()
            estimate = self.kalman.correct(measurement)
            self.miss_count = 0
        elif self._initialized:
            estimate = self.kalman.predict()
            self.miss_count += 1
            if self.miss_count > self.max_miss:
                self._initialized = False

        if estimate is None:
            self.trail.clear()
            return {"position": None, "speed": 0.0, "ball_activity_score": 0.0}

        x, y, vx, vy = [float(v) for v in estimate.flatten()]
        speed = (vx ** 2 + vy ** 2) ** 0.5
        self.trail.append((int(x), int(y)))

        if measurement is not None:
            # Normalize speed into a 0-1 activity score. Divisor is a tunable heuristic —
            # calibrate against real footage (pixels/frame at typical rally-ball speed).
            ball_activity_score = min(1.0, speed / 25.0)
        else:
            # Coasting on prediction only (no fresh detection this frame) — decay confidence.
            ball_activity_score = max(0.0, 0.3 - 0.05 * self.miss_count)

        return {"position": (int(x), int(y)), "speed": speed, "ball_activity_score": ball_activity_score}

    def draw_trail(self, annotated_frame, color=(0, 255, 255)):
        """Draws a fading trail of recent ball positions onto annotated_frame in-place."""
        pts = list(self.trail)
        n = len(pts)
        for i in range(1, n):
            thickness = max(1, int(3 * (i / n)))
            cv2.line(annotated_frame, pts[i - 1], pts[i], color, thickness)
        if pts:
            cv2.circle(annotated_frame, pts[-1], 4, color, -1)
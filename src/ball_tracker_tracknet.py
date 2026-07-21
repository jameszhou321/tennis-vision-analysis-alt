"""ball_tracker_tracknet.py — TrackNet-backed Ball Tracker (wraps yastrebksv/TrackNet)

Function: Loads the pretrained TrackNet model (https://github.com/yastrebksv/TrackNet) and
exposes the same interface as ball_tracker.BallTracker (`update(frame)`, `draw_trail(...)`),
so it's a drop-in replacement for the classical heuristic tracker in main.py.

SETUP REQUIRED before this module will import successfully:
  1. Download model.py and general.py from https://github.com/yastrebksv/TrackNet and place
     them at src/tracknet/model.py and src/tracknet/general.py (plus an empty
     src/tracknet/__init__.py so the folder is importable).
  2. Download the pretrained weights linked in that repo's README and place them at
     src/models/tracknet/model_best.pt (or point TRACKNET_WEIGHTS_PATH below elsewhere).

TrackNet takes 3 consecutive frames (current + previous 2) stacked as a 9-channel input at a
fixed 640x360 resolution, and outputs a heatmap indicating ball position. This wrapper keeps a
rolling 3-frame buffer, runs inference each call, and converts the heatmap back to a (x, y)
position in the original frame's coordinate space.
"""
import os
from collections import deque

import cv2
import numpy as np
import torch

try:
    from tracknet.model import BallTrackerNet
    from tracknet.general import postprocess as tracknet_postprocess
    TRACKNET_AVAILABLE = True
except ImportError:
    TRACKNET_AVAILABLE = False

TRACKNET_WEIGHTS_PATH = os.path.join(os.path.dirname(__file__), "models", "tracknet", "model_best.pt")
TRACKNET_INPUT_WIDTH = 640
TRACKNET_INPUT_HEIGHT = 360


class TrackNetBallTracker:
    def __init__(self, weights_path=TRACKNET_WEIGHTS_PATH, device="cpu", trail_length=15):
        if not TRACKNET_AVAILABLE:
            raise ImportError(
                "TrackNet source files not found. Download model.py and general.py from "
                "https://github.com/yastrebksv/TrackNet into src/tracknet/ (see this file's "
                "module docstring for the full setup steps)."
            )
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"TrackNet weights not found at {weights_path}. Download them from the Google "
                "Drive link in https://github.com/yastrebksv/TrackNet's README."
            )

        self.device = device
        self.model = BallTrackerNet()
        self.model.load_state_dict(torch.load(weights_path, map_location=device))
        self.model.to(device)
        self.model.eval()

        self.frame_buffer = deque(maxlen=3)  # [current, previous, previous-previous]
        self.trail = deque(maxlen=trail_length)
        self.last_position = None
        self.miss_count = 0
        self.max_miss = 10

    def _preprocess(self, frame):
        """Resizes and stacks the last 3 frames (current, prev, prev-prev) into model input."""
        resized = [cv2.resize(f, (TRACKNET_INPUT_WIDTH, TRACKNET_INPUT_HEIGHT)) for f in self.frame_buffer]
        # frame_buffer[0] is the most recent (current) frame since we appendleft below.
        imgs = np.concatenate(resized, axis=2).astype(np.float32) / 255.0
        imgs = np.rollaxis(imgs, 2, 0)
        inp = np.expand_dims(imgs, axis=0)
        return torch.from_numpy(inp).float().to(self.device)

    def update(self, frame):
        """Processes one frame. Returns {"position": (x,y) or None, "speed": float, "ball_activity_score": float}."""
        self.frame_buffer.appendleft(frame)

        if len(self.frame_buffer) < 3:
            # Not enough history yet (first 2 frames of the clip) — no prediction possible.
            return {"position": None, "speed": 0.0, "ball_activity_score": 0.0}

        frame_h, frame_w = frame.shape[:2]
        scale_x = frame_w / TRACKNET_INPUT_WIDTH
        scale_y = frame_h / TRACKNET_INPUT_HEIGHT

        with torch.no_grad():
            inp = self._preprocess(frame)
            out = self.model(inp)
            output = out.argmax(dim=1).detach().cpu().numpy()[0]

        x_pred, y_pred = tracknet_postprocess(output, scale=1)  # unscale ourselves below (non-square scale)

        if x_pred is None or y_pred is None:
            self.miss_count += 1
            position = None
            ball_activity_score = max(0.0, 0.3 - 0.05 * self.miss_count)
            speed = 0.0
        else:
            gx, gy = float(x_pred) * scale_x, float(y_pred) * scale_y
            position = (int(gx), int(gy))
            self.miss_count = 0

            if self.last_position is not None:
                dx = position[0] - self.last_position[0]
                dy = position[1] - self.last_position[1]
                speed = (dx ** 2 + dy ** 2) ** 0.5
            else:
                speed = 0.0

            self.last_position = position
            self.trail.append(position)
            # Divisor is an unvalidated placeholder — calibrate against real footage.
            ball_activity_score = min(1.0, speed / 25.0)

        if position is None and self.miss_count > self.max_miss:
            self.trail.clear()
            self.last_position = None

        return {"position": position, "speed": speed, "ball_activity_score": ball_activity_score}

    def draw_trail(self, annotated_frame, color=(0, 255, 255)):
        """Draws a fading trail of recent ball positions onto annotated_frame in-place."""
        pts = list(self.trail)
        n = len(pts)
        for i in range(1, n):
            thickness = max(1, int(3 * (i / n)))
            cv2.line(annotated_frame, pts[i - 1], pts[i], color, thickness)
        if pts:
            cv2.circle(annotated_frame, pts[-1], 4, color, -1)
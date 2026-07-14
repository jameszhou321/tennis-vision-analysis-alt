"""court_detector.py — Court Detector (for use by main.py)

Function: Encapsulates YOLO tennis court keypoint detection, providing the CourtDetector class interface.
Fix: Added a robust Hough Transform line unpacking guard to prevent Unpack errors caused by anomalous data dimensions.
"""
import cv2
import numpy as np


class CourtDetector:
    def __init__(self, scale=0.5):
        self.scale = scale

    def get_rois(self, frame, width, height):
        """
        Extracts court edges and returns the ROI coordinates for the far end and near end.
        """
        small_frame = cv2.resize(frame, (0, 0), fx=self.scale, fy=self.scale)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        edges = cv2.Canny(thresh, 50, 150)

        h, w = small_frame.shape[:2]
        min_line_len = int(w * 0.15)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=min_line_len, maxLineGap=20)

        if lines is None:
            return None, None

        horizontals, left_lines, right_lines = [], [], []

        for line in lines:
            # =====================================================================
            # Enhanced Unpacking Guard: Ensure only valid line segments matching 
            # the [x1, y1, x2, y2] format are parsed.
            # =====================================================================
            try:
                if len(line.shape) == 1 and len(line) == 4:
                    x1, y1, x2, y2 = line
                elif len(line) > 0 and len(line[0]) == 4:
                    x1, y1, x2, y2 = line[0]
                else:
                    continue  # Skip structural noise data anomalies
            except (TypeError, IndexError, ValueError):
                continue  # Catch any potential unpacking exceptions to ensure multithreading does not crash
            # =====================================================================

            if y1 < h * 0.3 or y2 < h * 0.3 or y1 > h * 0.9 or y2 > h * 0.9:
                continue
            if y1 < y2:
                x1, y1, x2, y2 = x2, y2, x1, y1

            dx, dy = x2 - x1, y2 - y1
            if dx == 0:
                continue

            angle = np.degrees(np.arctan2(dy, dx))
            if -15 <= angle <= 5 or -180 <= angle <= -165:
                horizontals.append((x1, y1, x2, y2))
            elif -80 <= angle <= -35:
                left_lines.append((x1, y1, x2, y2))
            elif -145 <= angle <= -100:
                right_lines.append((x1, y1, x2, y2))

        if len(horizontals) >= 1 and len(left_lines) >= 1 and len(right_lines) >= 1:
            all_y = [p[1] for line in horizontals + left_lines + right_lines for p in
                     ((line[0], line[1]), (line[2], line[3]))]
            c_y_min = max(int(h * 0.35), min(all_y))
            c_y_max = min(int(h * 0.85), max(all_y))
            c_h = c_y_max - c_y_min

            if c_h > h * 0.1:
                tl_x, tr_x = self._get_x(left_lines, c_y_min), self._get_x(right_lines, c_y_min)
                bl_x, br_x = self._get_x(left_lines, c_y_max), self._get_x(right_lines, c_y_max)

                if None not in (tl_x, tr_x, bl_x, br_x):
                    net_candidates = []
                    for x1, y1, x2, y2 in horizontals:
                        avg_y = (y1 + y2) / 2
                        if c_y_min + c_h * 0.35 < avg_y < c_y_min + c_h * 0.55:
                            net_candidates.append((avg_y, abs(x2 - x1)))

                    if net_candidates:
                        net_candidates.sort(key=lambda x: x[1], reverse=True)
                        net_y = net_candidates[0][0]
                    else:
                        net_y = c_y_min + c_h * 0.45

                    # Extend upwards by 80% of the court height to ensure the far player fits completely into the ROI
                    f_y1 = max(0, c_y_min - c_h * 0.8)

                    f_y2 = net_y + 15 * self.scale
                    f_x1_b = self._get_x(left_lines, net_y) or tl_x
                    f_x2_b = self._get_x(right_lines, net_y) or tr_x
                    f_x1, f_x2 = f_x1_b, f_x2_b

                    # Near end ROI
                    n_y1 = net_y - 15 * self.scale
                    n_y2 = min(h, c_y_max + c_h * 0.3)
                    n_x1_b = self._get_x(left_lines, c_y_max) or bl_x
                    n_x2_b = self._get_x(right_lines, c_y_max) or br_x
                    nw = n_x2_b - n_x1_b
                    n_x1, n_x2 = n_x1_b - nw * 0.1, n_x2_b + nw * 0.1

                    def scale_box(box):
                        return [max(0, int(v / self.scale)) for v in box]

                    far_roi = scale_box([f_x1, f_y1, f_x2, f_y2])
                    near_roi = scale_box([n_x1, n_y1, n_x2, n_y2])

                    far_roi = [far_roi[0], far_roi[1], min(width, far_roi[2]), min(height, far_roi[3])]
                    near_roi = [near_roi[0], near_roi[1], min(width, near_roi[2]), min(height, near_roi[3])]

                    return far_roi, near_roi
        return None, None

    def _get_x(self, lines_list, ty):
        xs = [x1 + (ty - y1) * (x2 - x1) / (y2 - y1) for x1, y1, x2, y2 in lines_list if y1 != y2]
        return np.median(xs) if xs else None
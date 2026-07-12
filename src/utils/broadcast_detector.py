import cv2
import numpy as np

class BroadcastViewClassifier:
    def __init__(self, target_hue_range=(85, 130), min_line_count=6):
        """
        Classifies broadcast frames into 'Rally View' vs 'Non-Playing Views'.
        :param target_hue_range: HSV Hue range for the court surface (85-130 covers most blue/green courts)
        :param min_line_count: Minimum structural lines required to confirm a court layout
        """
        self.target_hue_range = target_hue_range
        self.min_line_count = min_line_count
        
        # State smoothing buffer (prevents rapid flickering between states)
        self.state_history = []
        self.buffer_size = 10 

    def is_rally_view(self, frame):
        """
        Analyzes frame features to detect a classic wide-angle tennis court view.
        """
        # 1. Downscale frame for speed optimization
        small_frame = cv2.resize(frame, (640, 360))
        hsv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

        # 2. Check for Court Color Presence
        lower_bound = np.array([self.target_hue_range[0], 40, 40])
        upper_bound = np.array([self.target_hue_range[1], 255, 255])
        court_mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        # Calculate what percentage of the screen matches the court color
        court_pixel_ratio = np.sum(court_mask > 0) / court_mask.size

        # 3. Structural Line Detection (Hough Lines)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=80, maxLineGap=10)
        
        line_count = len(lines) if lines is not None else 0

        # 4. Binary Decision Heuristic
        # A rally view has ample court color visible AND structured geometric lines
        is_current_frame_rally = (court_pixel_ratio > 0.15) and (line_count >= self.min_line_count)

        # 5. Apply Temporal Smoothing Window
        self.state_history.append(is_current_frame_rally)
        if len(self.state_history) > self.buffer_size:
            self.state_history.pop(0)

        # Return the majority vote from the buffer
        return sum(self.state_history) > (len(self.state_history) / 2)
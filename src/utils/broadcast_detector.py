"""
broadcast_detector.py — Spatial Rally Detection Engine

Function: Detects tennis rallies based on the spatial physics of the ball 
          and positional distribution of players.
"""
import cv2
import numpy as np

class SpatialRallyDetector:
    def __init__(self, fps=30, buffer_seconds=2.0, speed_threshold=12.0):
        """
        Detects rallies based on the physics of the ball and player positions.
        :param speed_threshold: Minimum pixel movement per frame to consider the ball 'active'
        """
        self.fps = fps
        self.buffer_frames = int(fps * buffer_seconds)
        self.speed_threshold = speed_threshold
        
        # Tracking buffers
        self.ball_history = []
        self.frames_since_active_play = self.buffer_frames
        self.is_playing = False

    def update(self, ball_xy, player_boxes):
        """
        Core logic engine. Call this every frame.
        :param ball_xy: (x, y) tuple of the ball, or None if not detected
        :param player_boxes: List of bounding boxes [[x1,y1,x2,y2], ...] for detected players
        """
        is_frame_active = False

        # 1. Analyze Ball Physics
        if ball_xy is not None:
            self.ball_history.append(ball_xy)
            if len(self.ball_history) > 5:
                self.ball_history.pop(0)

            if len(self.ball_history) >= 2:
                # Calculate velocity between the last two frames
                p1 = np.array(self.ball_history[-2])
                p2 = np.array(self.ball_history[-1])
                velocity = np.linalg.norm(p2 - p1)

                # If the ball is flying fast, it's a strong indicator of live play
                if velocity > self.speed_threshold:
                    is_frame_active = True

        # 2. Analyze Player Context (Spatial separation)
        # If players are too close to each other (e.g., walking to change sides or high-fiving), 
        # it's likely a break period.
        if player_boxes is not None and len(player_boxes) >= 2:
            p1_center_y = (player_boxes[0][1] + player_boxes[0][3]) / 2
            p2_center_y = (player_boxes[1][1] + player_boxes[1][3]) / 2
            
            # If they are vertically separated (one far court, one near court), confirm layout
            vertical_distance = abs(p1_center_y - p2_center_y)
            if vertical_distance < 150:  # Adjust based on video resolution
                is_frame_active = False

        # 3. State Management Window
        if is_frame_active:
            self.is_playing = True
            self.frames_since_active_play = 0
        else:
            self.frames_since_active_play += 1

        # If the ball goes dead or players stop moving for too long, kill the rally state
        if self.frames_since_active_play >= self.buffer_frames:
            self.is_playing = False
            self.ball_history.clear()

        return self.is_playing
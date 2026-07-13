import cv2
import os
from broadcast_detector import SpatialRallyDetector  # Uses our updated tracking logic

def main():
    input_video_path = "input_videos/match.mp4"
    output_video_path = "output_spatial_analysis.mp4"

    if not os.path.exists(input_video_path):
        print(f"File missing at: {input_video_path}")
        return

    cap = cv2.VideoCapture(input_video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Initialize the spatial tracker
    tracker = SpatialRallyDetector(fps=fps, buffer_seconds=2.0)

    print("Running spatial tracking analysis...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # =====================================================================
        # HOOK YOUR EXISTING TRACKERS HERE
        # Swap out these mock variables for your project's real tracking logic!
        # =====================================================================
        # Example: 
        # ball_xy = your_ball_detector.get_xy(frame) 
        # player_boxes = your_player_detector.get_boxes(frame)
        
        ball_xy = (int(width/2), int(height/2))  # Placeholder center coords
        player_boxes = [[100, 200, 150, 350], [100, 500, 150, 650]] # Placeholder bounds
        # =====================================================================

        # Calculate live state using the tracking data
        in_play = tracker.update(ball_xy, player_boxes)

        # UI Visuals
        status_text = "PLAYING" if in_play else "NOT PLAYING"
        color = (0, 255, 0) if in_play else (0, 0, 255)

        cv2.rectangle(frame, (30, 30), (400, 110), (0, 0, 0), -1)
        cv2.putText(frame, f"STATE: {status_text}", (50, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3, cv2.LINE_AA)

        out.write(frame)
        cv2.imshow("Spatial Rally Analysis", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Processing complete!")

if __name__ == "__main__":
    main()
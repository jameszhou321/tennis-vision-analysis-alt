import cv2
import os
from broadcast_detector import BroadcastViewClassifier

def main():
    # Setup paths - Change these to match your workspace video names
    input_video_path = "input_videos/shortenedos.mp4"
    output_video_path = "output_broadcast_analyzed.mp4"

    if not os.path.exists(input_video_path):
        print(f"Error: Please place a video file at '{input_video_path}' before running.")
        return

    # Initialize video capture engine
    cap = cv2.VideoCapture(input_video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize video output writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Instantiate our broadcast classifier tool
    # (Adjust target_hue_range if working with a unique clay red court: e.g., 0 to 20)
    classifier = BroadcastViewClassifier(target_hue_range=(85, 130))

    print("Processing broadcast video streams...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame state
        is_rally = classifier.is_rally_view(frame)

        # Determine UI attributes based on state
        if is_rally:
            status_text = "STATE: PLAYING (RALLY VIEW)"
            color = (0, 255, 0)  # Green
        else:
            status_text = "STATE: NOT PLAYING (NON-RALLY)"
            color = (0, 0, 255)  # Red

        # Render a sleek UI header bar on top of the frame
        cv2.rectangle(frame, (20, 20), (620, 90), (15, 15, 15), -1)
        cv2.putText(frame, status_text, (40, 65), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3, cv2.LINE_AA)

        # Save and display frame live
        out.write(frame)
        cv2.imshow("Broadcast View Classification Pipeline", frame)

        # Safe interrupt option
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release hardware holds cleanly
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Analysis Complete! Video exported to: {output_video_path}")

if __name__ == "__main__":
    main()
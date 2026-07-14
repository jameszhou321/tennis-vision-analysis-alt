"""inference_viewer.py — Person classification model inference visualization tool"""
import os
import cv2
from pathlib import Path
from ultralytics import YOLO

_UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(os.path.dirname(_UTILS_DIR))

# ================= Configuration =================
MODEL_PATH = os.path.join(_PROJECT_DIR, "models", "person", "best.pt")
DATA_DIR = os.path.join(_PROJECT_DIR, "data", "rallies_new")
CONFIDENCE_THRESHOLD = 0.4


def main():
    print("⏳ Loading YOLO26x model into GPU memory...")
    model = YOLO(MODEL_PATH)

    # 1. Scan and build the video list
    base_path = Path(DATA_DIR)
    video_files = list(base_path.rglob("raw_clip.mp4"))

    if not video_files:
        print("No video files found, please check the DATA_DIR path!")
        return

    print(f"Successfully loaded {len(video_files)} video clips")
    print("-" * 30)
    print("Video list index:")
    for idx, vf in enumerate(video_files):
        # Extract the parent folder / clip folder names for display
        print(f"[{idx:02d}] {vf.parent.parent.name} / {vf.parent.name}")
    print("-" * 30)

    # 2. Global state control
    cap = None
    is_paused = False
    update_trackbar_auto = False  # lock to prevent an infinite loop in the progress-bar callback

    # Initialize the window
    window_name = "YOLO Tennis Tracker (Graduation Project)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)  # initial window size

    # 3. Core interaction callback functions
    def load_video(idx):
        """Switch to a different video clip"""
        nonlocal cap, is_paused
        if cap is not None:
            cap.release()

        vid_path = str(video_files[idx])
        cap = cv2.VideoCapture(vid_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Dynamically update the progress bar's max value
        cv2.setTrackbarMax("Progress", window_name, max(1, total_frames - 1))
        cv2.setTrackbarPos("Progress", window_name, 0)
        is_paused = False
        print(f"\n▶ Now playing: {video_files[idx].parent.parent.name} / {video_files[idx].parent.name}")

    def on_video_trackbar(val):
        """Triggered when dragging the playlist bar"""
        load_video(val)

    def on_progress_trackbar(val):
        """Triggered when dragging the progress bar"""
        nonlocal update_trackbar_auto
        # If the trackbar update was triggered programmatically, do nothing
        if not update_trackbar_auto and cap is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, val)

            # Even while paused, run inference immediately and refresh the current frame
            ret, frame = cap.read()
            if ret:
                results = model.predict(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)
                annotated_frame = results[0].plot()
                cv2.imshow(window_name, annotated_frame)
                # After reading a frame, reset the pointer back to the dragged position to preserve state
                cap.set(cv2.CAP_PROP_POS_FRAMES, val)

    # 4. Create UI controls (progress bar and playlist bar)
    cv2.createTrackbar("Playlist", window_name, 0, len(video_files) - 1, on_video_trackbar)
    cv2.createTrackbar("Progress", window_name, 0, 100, on_progress_trackbar)

    # Load the first video on startup
    load_video(0)

    # 5. Main loop
    while True:
        if not is_paused:
            ret, frame = cap.read()
            if not ret:
                # Video finished playing, loop back to the start automatically
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # Get the current frame number and sync it to the progress bar UI
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            update_trackbar_auto = True  # lock
            cv2.setTrackbarPos("Progress", window_name, current_frame)
            update_trackbar_auto = False  # unlock

            # ================= YOLO inference core =================
            # verbose=False prevents the terminal from being flooded with detection logs
            results = model.predict(frame, verbose=False, conf=CONFIDENCE_THRESHOLD)

            # Get the image with detection boxes drawn (yellow box for near player,
            # purple box for far player in dark clothing)
            annotated_frame = results[0].plot()

            # Draw the keyboard shortcut hint
            ui_text = "Space: Pause/Play | Drag to Seek | Q: Quit"
            cv2.putText(annotated_frame, ui_text, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow(window_name, annotated_frame)

        # Keyboard event listener (wait 30ms, roughly equal to 33 FPS)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q') or key == 27:  # Q key or ESC
            break
        elif key == 32:  # Space bar: toggle pause/play state
            is_paused = not is_paused
            state_str = "⏸ Paused" if is_paused else "▶ Playing"
            print(f"[{state_str}]")

    # Final cleanup
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
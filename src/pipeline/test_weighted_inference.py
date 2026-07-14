"""test_weighted_inference.py — Weighted Inference Test Script

Function: Test target tracking effects in videos using an OpenCV tracker to validate the inference pipeline.
"""
import os
import cv2
import numpy as np

_PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(os.path.dirname(_PIPELINE_DIR))

# Video capture (replace with your video file)
cap = cv2.VideoCapture(os.path.join(_PROJECT_DIR, "data", "rallies_annotated", "rally_003_24.0s", "raw_clip.mp4"))

# BBox used to store the target when first detected and initialize the tracker
bbox = None
init_tracking = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # If the tracker is not yet initialized, attempt to initialize via color detection
    if not init_tracking:
        # 1. Preprocessing: Blur to reduce noise (Optional)
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        # 2. Convert color space to HSV
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # 3. Define the upper and lower HSV bounds for black (requires repeated tuning based on specific lighting)
        # H: Hue, S: Saturation, V: Value (Brightness)
        # Extremely low H, S, V values typically represent black. This is a rough range:
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 30])  # S can be set larger to handle blended colors, V controls brightness and must be minimal

        # 4. Create mask
        mask = cv2.inRange(hsv, lower_black, upper_black)

        # 5. Morphological operations (Optional, e.g., erosion, dilation) to further denoise and connect small blobs
        # kernel = np.ones((3,3), np.uint8)
        # mask = cv2.erode(mask, kernel, iterations=1)
        # mask = cv2.dilate(mask, kernel, iterations=1)

        # 6. Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the largest black blob (assuming it is the far-end player)
        max_contour = None
        max_area = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Filter out tiny noise blobs, but it shouldn't be too large either (since the player is quite small at the far end)
            if 5 < area < 100 and area > max_area:  # This needs adjustment based on actual pixel dimensions
                max_area = area
                max_contour = cnt

        if max_contour is not None:
            # 7. Get initial BBox
            x, y, w, h = cv2.boundingRect(max_contour)
            bbox = (x, y, w, h)

            # 8. Initialize the tracker
            success = tracker.init(frame, bbox)
            if success:
                init_tracking = True
                print("Tracking initialized, black blob detected:", bbox)
            else:
                print("Tracker initialization failed")

    # If already initialized, update the tracker position
    else:
        success, bbox = tracker.update(frame)
        if success:
            # Draw tracking box
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Tracking", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            # Tracking lost, can attempt to re-initialize using color detection (logic similar to above)
            print("Tracking lost, attempting re-detection")
            init_tracking = False  # Reset state to re-trigger detection and initialization next time

    # Display results
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
"""label_tool.py — Player Bounding Box Manual Annotation Tool (YOLO format)

Function: Draw bounding boxes on images by dragging the mouse, label near/far
players, and save annotations in YOLO txt format.
"""
import cv2
import os
import glob
import numpy as np

# ================= Configuration =================
# Assumes the script and the image folder are in the same directory
IMAGE_DIR = "data/image"  # Update this path to match your image folder
LABEL_DIR = "data/labels"  # Corresponding labels output folder

# YOLO class definitions
CLASSES = ["player_near", "player_far"]

# ================= Global State =================
drawing = False
start_x, start_y = -1, -1
current_boxes = []  # Boxes for the current image: [(class_id, x1, y1, x2, y2)]
img_list = []
current_img_index = 0
img_copy = None  # Image copy used while drawing


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def convert_to_yolo_format(img_w, img_h, box):
    """Convert (x1, y1, x2, y2) to normalized YOLO format (x_center, y_center, w, h)"""
    cls_id, x1, y1, x2, y2 = box
    dw = 1. / img_w
    dh = 1. / img_h
    x_center = (x1 + x2) / 2.0
    y_center = (y1 + y2) / 2.0
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    return (cls_id, x_center * dw, y_center * dh, w * dw, h * dh)


def save_annotations(img_path, boxes, img_w, img_h):
    """Save the current image's annotations to a txt file (Python's built-in
    open() supports non-ASCII paths)"""
    if not boxes:
        return

    img_name = os.path.basename(img_path)
    txt_name = os.path.splitext(img_name)[0] + ".txt"
    txt_path = os.path.join(LABEL_DIR, txt_name)

    # Make sure the txt file is saved with utf-8 encoding
    with open(txt_path, 'w', encoding='utf-8') as f:
        for box in boxes:
            yolo_box = convert_to_yolo_format(img_w, img_h, box)
            f.write(f"{yolo_box[0]} {yolo_box[1]:.6f} {yolo_box[2]:.6f} {yolo_box[3]:.6f} {yolo_box[4]:.6f}\n")
    print(f"Saved annotation: {txt_name}")


def load_annotations(img_path, img_w, img_h):
    """Try to load existing annotations if this image was labeled before"""
    global current_boxes
    current_boxes = []
    img_name = os.path.basename(img_path)
    txt_name = os.path.splitext(img_name)[0] + ".txt"
    txt_path = os.path.join(LABEL_DIR, txt_name)

    if os.path.exists(txt_path):
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_id = int(parts[0])
                    x_c, y_c, w_norm, h_norm = map(float, parts[1:])
                    # Denormalize back to pixel coordinates
                    w = w_norm * img_w
                    h = h_norm * img_h
                    x1 = int((x_c * img_w) - (w / 2))
                    y1 = int((y_c * img_h) - (h / 2))
                    x2 = int((x_c * img_w) + (w / 2))
                    y2 = int((y_c * img_h) + (h / 2))
                    current_boxes.append((cls_id, x1, y1, x2, y2))


def draw_boxes(img):
    """Draw the existing boxes on the image"""
    for box in current_boxes:
        cls_id, x1, y1, x2, y2 = box
        color = (0, 255, 255) if cls_id == 0 else (255, 0, 255)  # near = yellow, far = magenta
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, CLASSES[cls_id], (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def mouse_callback(event, x, y, flags, param):
    global drawing, start_x, start_y, current_boxes, img_copy

    img_original = param

    if event == cv2.EVENT_LBUTTONDOWN:
        # If two boxes have already been drawn, stop responding
        if len(current_boxes) >= 2:
            print("Two players have already been labeled! Press 'C' to relabel.")
            return

        drawing = True
        start_x, start_y = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            img_copy = img_original.copy()
            draw_boxes(img_copy)
            # Draw the box currently being dragged
            cls_id = len(current_boxes)  # 0 = near, 1 = far
            color = (0, 255, 255) if cls_id == 0 else (255, 0, 255)
            cv2.rectangle(img_copy, (start_x, start_y), (x, y), color, 2)
            cv2.imshow("YOLO Annotator", img_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        if drawing:
            drawing = False
            # Normalize coordinates in case the mouse was dragged backwards
            x1, x2 = min(start_x, x), max(start_x, x)
            y1, y2 = min(start_y, y), max(start_y, y)

            # Filter out accidental tiny boxes
            if x2 - x1 > 5 and y2 - y1 > 5:
                cls_id = len(current_boxes)
                current_boxes.append((cls_id, x1, y1, x2, y2))

            img_copy = img_original.copy()
            draw_boxes(img_copy)
            cv2.imshow("YOLO Annotator", img_copy)


def main():
    global current_img_index, current_boxes, img_copy

    ensure_dir(LABEL_DIR)

    # Get all images
    valid_exts = ('*.jpg', '*.jpeg', '*.png')
    img_paths = []
    for ext in valid_exts:
        img_paths.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))
    img_paths = sorted(img_paths)

    if not img_paths:
        print(f"Error: No images found in {IMAGE_DIR}. Please check the path.")
        return

    cv2.namedWindow("YOLO Annotator", cv2.WINDOW_AUTOSIZE)

    while current_img_index < len(img_paths):
        img_path = img_paths[current_img_index]

        # !!! KEY FIX !!!
        # Instead of using cv2.imread directly, read the raw bytes with numpy
        # and decode with cv2.imdecode (handles non-ASCII paths correctly)
        try:
            img_data = np.fromfile(img_path, dtype=np.uint8)
            img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"Exception while reading file: {img_path}, error: {e}")
            img = None

        if img is None:
            print(f"Failed to decode image, skipping: {os.path.basename(img_path)}")
            current_img_index += 1
            continue

        img_h, img_w = img.shape[:2]

        # Load existing annotations, if any
        load_annotations(img_path, img_w, img_h)

        img_copy = img.copy()
        draw_boxes(img_copy)

        # Bind the mouse callback
        cv2.setMouseCallback("YOLO Annotator", mouse_callback, param=img)

        while True:
            # Add UI hint text
            display_img = img_copy.copy()
            text = f"Img {current_img_index + 1}/{len(img_paths)} | [Space/D]: Next | [A]: Prev | [C]: Clear Box | [X]: Del Img | [Q]: Quit"
            cv2.putText(display_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow("YOLO Annotator", display_img)

            key = cv2.waitKey(10) & 0xFF

            if key == ord('d') or key == 32:  # D or Space: next image (and save)
                save_annotations(img_path, current_boxes, img_w, img_h)
                current_img_index += 1
                break

            elif key == ord('a'):  # A: previous image (and save current)
                save_annotations(img_path, current_boxes, img_w, img_h)
                current_img_index = max(0, current_img_index - 1)
                break

            elif key == ord('c'):  # C: clear all boxes for the current image
                current_boxes = []
                img_copy = img.copy()
                print("Cleared all annotation boxes for the current image")

            elif key == ord('x'):  # X: delete the current image and its label file
                try:
                    os.remove(img_path)
                    txt_path = os.path.join(LABEL_DIR, os.path.splitext(os.path.basename(img_path))[0] + ".txt")
                    if os.path.exists(txt_path):
                        os.remove(txt_path)
                    print(f"Permanently deleted bad image: {os.path.basename(img_path)}")
                except Exception as e:
                    print(f"Failed to delete file: {e}")

                img_paths.pop(current_img_index)
                if current_img_index >= len(img_paths):
                    current_img_index = len(img_paths) - 1
                break

            elif key == ord('q') or key == 27:  # Q or ESC: quit
                save_annotations(img_path, current_boxes, img_w, img_h)
                print("Exiting annotation tool")
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()
    print("All images have been reviewed!")


if __name__ == "__main__":
    main()
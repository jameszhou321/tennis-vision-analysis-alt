"""pose_tracker.py — Pose Tracker (for use by main.py)

Function: Encapsulates YOLO pose estimation, providing the PoseTracker class interface, inclusive of EMA smoothing and frame drop compensation.
"""
import cv2
import numpy as np
import config_legacy as config


class PoseTracker:
    def __init__(self, model):
        self.model = model
        self.alpha = config.POSE_ALPHA
        self.max_gap = config.POSE_MAX_GAP

    def process_and_smooth(self, crop_img, offset_x, offset_y, is_far, history_state, annotated_frame):
        """
        Performs model inference, multi-dimensional scoring (Y-axis bottom priority + X-axis inertia), 
        data smoothing, and keypoint rendering.
        """
        new_box, new_kpts = None, None

        if crop_img.shape[0] >= 10 and crop_img.shape[1] >= 10:
            conf_threshold = config.CONF_FAR if is_far else config.CONF_NEAR
            res = self.model.predict(crop_img, imgsz=config.YOLO_IMGSZ, conf=conf_threshold,
                                     classes=[0], verbose=False)[0]

            if res.boxes is not None and len(res.boxes) > 0:
                best_idx = -1
                max_score = -1.0

                roi_h, roi_w = crop_img.shape[:2]

                # ==========================================
                # Decoupled X-axis and Y-axis Decision Logic
                # ==========================================
                if history_state['box'] is not None:
                    # Tracking Mode: Inherit only the inertia expectation of the X-axis (left/right movement)
                    prev_bx1, prev_by1, prev_bx2, prev_by2 = history_state['box']
                    expected_cx = ((prev_bx1 + prev_bx2) / 2.0) - offset_x
                    max_x_tolerance = roi_w * 0.25
                else:
                    # Initialization Mode: Search along the court centerline by default
                    expected_cx = roi_w / 2.0
                    max_x_tolerance = roi_w * 0.6

                for i, box in enumerate(res.boxes):
                    bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf.item()

                    person_cx = (bx1 + bx2) / 2.0

                    # 1. X-axis Inertia Score (Prevents misdetecting ball kids on either side)
                    x_dist = abs(person_cx - expected_cx)
                    x_score = max(0, 1.0 - (x_dist / max_x_tolerance))

                    # 2. Y-axis Bottom Priority Score (Prevents misdetecting audience in the back stands)
                    # by2 is the bottom edge of the person's bounding box (feet). 
                    # The closer the feet are to the bottom of the ROI (roi_h), the closer y_score is to 1.0.
                    y_score = by2 / roi_h

                    # ==========================================
                    # Customized Weight Assignment
                    # ==========================================
                    if is_far:
                        # Far end strategy: 50% on bottom proximity, 30% on tracking inertia, and 20% on YOLO confidence.
                        score = conf * 0.2 + y_score * 0.5 + x_score * 0.3
                    else:
                        # Near end strategy: Distinct features, 50% on confidence and 50% on tracking inertia.
                        score = conf * 0.5 + x_score * 0.5

                    if score > max_score:
                        max_score = score
                        best_idx = i

                if best_idx != -1 and max_score > 0.1:
                    bx1, by1, bx2, by2 = res.boxes.xyxy[best_idx].cpu().numpy()
                    new_box = [float(bx1 + offset_x), float(by1 + offset_y),
                               float(bx2 + offset_x), float(by2 + offset_y)]

                    if res.keypoints is not None:
                        kpts = res.keypoints.data[best_idx].cpu().numpy()
                        new_kpts = []
                        for kp in kpts:
                            kx, ky, kconf = kp
                            g_kx = float(kx + offset_x) if kx > 0 else 0.0
                            g_ky = float(ky + offset_y) if ky > 0 else 0.0
                            new_kpts.append([g_kx, g_ky, float(kconf)])

        # ==========================================
        # State Machine Update and EMA Debounce Smoothing
        # ==========================================
        final_box, final_kpts = None, None

        if new_box is not None:
            if history_state['box'] is not None:
                final_box = [self.alpha * n + (1 - self.alpha) * o for n, o in zip(new_box, history_state['box'])]
                final_kpts = []
                for nk, ok in zip(new_kpts, history_state['kpts']):
                    final_kpts.append([
                        self.alpha * nk[0] + (1 - self.alpha) * ok[0],
                        self.alpha * nk[1] + (1 - self.alpha) * ok[1],
                        nk[2]
                    ])
            else:
                final_box = new_box
                final_kpts = new_kpts

            history_state['box'] = final_box
            history_state['kpts'] = final_kpts
            history_state['miss'] = 0
        else:
            history_state['miss'] += 1
            if history_state['miss'] <= self.max_gap and history_state['box'] is not None:
                final_box = history_state['box']
                final_kpts = history_state['kpts']
            else:
                history_state['box'] = None
                history_state['kpts'] = None

        # ==========================================
        # Visual Rendering and Drawing
        # ==========================================
        if final_box is not None:
            cv2.rectangle(annotated_frame, (int(final_box[0]), int(final_box[1])),
                          (int(final_box[2]), int(final_box[3])), (0, 0, 255), 2)

            pt_color = (0, 255, 0) if is_far else (0, 255, 255)
            for kp in final_kpts:
                kx, ky, kconf = kp
                if kconf > 0.3:
                    cv2.circle(annotated_frame, (int(kx), int(ky)), 4, pt_color, -1)

            return {"bbox": final_box, "keypoints": final_kpts}

        return None
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

    def process_and_smooth(self, crop_img, offset_x, offset_y, is_far, history_state, annotated_frame,
                            prev_gray_frame=None, curr_gray_frame=None):
        """
        Performs model inference, multi-dimensional scoring (Y-axis bottom priority + X-axis inertia +
        court proximity + local motion), data smoothing, and keypoint rendering.

        prev_gray_frame / curr_gray_frame: optional full-frame grayscale images (previous and current
        frame of the source clip) used to compute a per-candidate local motion score. Pass both to
        enable it; omit to fall back to a neutral prior (e.g. on the very first frame).
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

                    # 3. Court Proximity Score (soft, NOT a hard cutoff): players routinely leave the
                    # sideline area to chase wide balls, so this decays gradually with distance from the
                    # ROI's court centerline rather than rejecting anything outside a fixed boundary.
                    # Unlike x_score, this is independent of tracking history, so it doesn't compound
                    # drift if the tracker has already locked onto the wrong person.
                    court_cx = roi_w / 2.0
                    court_dist = abs(person_cx - court_cx)
                    prox_score = max(0.0, 1.0 - (court_dist / (roi_w * 0.6)))

                    # 4. Local Motion Score: players are almost always moving mid-rally (running,
                    # swinging, adjusting footwork); officials and ball kids are comparatively static
                    # frame-to-frame. Measures how much this candidate's own patch of the frame changed
                    # since the previous frame. Neutral (0.5) if no previous frame is available yet.
                    gx1 = int(bx1 + offset_x)
                    gy1 = int(by1 + offset_y)
                    gx2 = int(bx2 + offset_x)
                    gy2 = int(by2 + offset_y)
                    motion_score = self._local_motion_score(prev_gray_frame, curr_gray_frame, gx1, gy1, gx2, gy2)

                    # ==========================================
                    # Customized Weight Assignment
                    # ==========================================
                    if is_far:
                        # Far end strategy: bottom proximity still weighted highest (perspective makes
                        # this reliable), with confidence, tracking inertia, court proximity, and motion
                        # rounding out the score.
                        score = conf * 0.15 + y_score * 0.25 + x_score * 0.20 + prox_score * 0.20 + motion_score * 0.20
                    else:
                        # Near end strategy: confidence and tracking inertia remain primary (distinct
                        # features up close), with court proximity and motion as secondary tie-breakers.
                        score = conf * 0.30 + x_score * 0.30 + prox_score * 0.20 + motion_score * 0.20

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

    def _local_motion_score(self, prev_gray_frame, curr_gray_frame, gx1, gy1, gx2, gy2):
        """Returns ~0-1: how much this candidate box's own patch of the frame changed since the
        previous frame. Used to down-weight static bystanders (officials, ball kids) relative to
        players, who are almost always moving during an active rally clip."""
        if prev_gray_frame is None or curr_gray_frame is None:
            return 0.5  # neutral prior when no previous frame is available (e.g. first frame)

        h, w = curr_gray_frame.shape[:2]
        gx1, gy1 = max(0, gx1), max(0, gy1)
        gx2, gy2 = min(w, gx2), min(h, gy2)
        if gx2 <= gx1 or gy2 <= gy1:
            return 0.0

        prev_patch = prev_gray_frame[gy1:gy2, gx1:gx2]
        curr_patch = curr_gray_frame[gy1:gy2, gx1:gx2]
        if prev_patch.size == 0 or curr_patch.size == 0 or prev_patch.shape != curr_patch.shape:
            return 0.0

        diff = cv2.absdiff(curr_patch, prev_patch)
        raw = float(np.mean(diff)) / 255.0
        # Scale factor is a tunable heuristic: typical player motion produces a mean pixel diff of only
        # a few percent, so this stretches that into a more usable 0-1 range. Tune against real footage.
        return min(1.0, raw * 6.0)
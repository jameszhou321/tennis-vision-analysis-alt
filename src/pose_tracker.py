"""pose_tracker.py — Pose Tracker (for use by main.py)

Function: Full-frame person+pose tracking (persistent IDs via BoT-SORT), projected into
real-world court coordinates via CourtDetector's homography, then selects the far/near
player by which track spends the most time near a baseline over the clip.

This replaces the old per-ROI-crop, single-slot scoring approach. That approach worked in
pixel space and needed a hand-tuned score (confidence + inertia + court proximity + local
motion) to try to reject non-players; a chair umpire sitting near the net, at the bottom
edge of the far-side ROI, ended up systematically favored by that score (the y-axis
"bottom priority" term) whenever the actual far player was deep near their own baseline.
Working in real-world court meters sidesteps this: a stationary official positioned near
the net is simply too far from either baseline to ever accumulate a competitive score.

Runs in two passes (this file + main.py's annotate_rally_clip), the same shape as the
proven approach in src/pipeline/offline_tennis_tracker.py, just adapted to run per rally
clip instead of a single long broadcast feed:
  Pass 1 (track_frame, called once per frame): run court homography + person tracking,
    record every track's projected real-world position for that frame.
  Pass 2 (select_players, then build_render_track + draw): pick the far/near track IDs,
    EMA-smooth + gap-fill each one exactly like the old single-slot state machine did,
    and draw them.
"""
import cv2
import numpy as np

import config_legacy as config

# Real-world baseline anchors (meters, net center = origin) -- must match
# court_detector.COURT_14_PTS_PHYSICAL's convention.
_TOP_BASELINE = np.array([0.0, 11.885])
_BOTTOM_BASELINE = np.array([0.0, -11.885])


class PoseTracker:
    def __init__(self, pose_model, court_detector):
        self.pose_model = pose_model
        self.court_detector = court_detector
        self.alpha = config.POSE_ALPHA
        self.max_gap = config.POSE_MAX_GAP

        # track_id -> {frame_idx: {"box": [x1,y1,x2,y2], "kpts": [[x,y,conf],...] or None,
        #                          "real": (x_m, y_m)}}
        self.tracks_db = {}
        self._first_track_call = True

    # ==========================================
    # Pass 1: per-frame collection
    # ==========================================
    def track_frame(self, frame, frame_idx, H):
        """Runs person+pose tracking on the full frame and records each track's projected
        real-world (meters) position. No-ops if the court homography isn't available yet
        this clip (tracking picks back up as soon as estimate_homography finds one)."""
        if H is None:
            return

        # persist=False on this clip's first tracked frame forces BoT-SORT to reset its
        # internal state -- required because main.py reuses the same pose_model instance
        # across every rally clip in a match, and without this the tracker would otherwise
        # carry track IDs/history over from the *previous* clip's last frame.
        res = self.pose_model.track(frame, persist=not self._first_track_call,
                                     tracker="botsort.yaml", classes=[0],
                                     imgsz=config.YOLO_IMGSZ, conf=config.POSE_TRACK_CONF,
                                     verbose=False)[0]
        self._first_track_call = False

        if res.boxes is None or res.boxes.id is None:
            return

        track_ids = res.boxes.id.int().cpu().tolist()
        boxes = res.boxes.xyxy.cpu().numpy()
        kpts_all = res.keypoints.data.cpu().numpy() if res.keypoints is not None else None
        H_inv = np.linalg.inv(H)

        for i, t_id in enumerate(track_ids):
            box = boxes[i]
            kpts = kpts_all[i] if kpts_all is not None else None

            # Ankles (COCO indices 15/16), falling back to bbox bottom-center when they
            # weren't detected confidently -- same fallback offline_tennis_tracker.py uses.
            if kpts is not None and kpts[15][2] > 0.2 and kpts[16][2] > 0.2:
                feet_px = (kpts[15][:2] + kpts[16][:2]) / 2.0
            else:
                feet_px = np.array([(box[0] + box[2]) / 2.0, box[3]])

            pt = np.array([[[feet_px[0], feet_px[1]]]], dtype=np.float32)
            real_xy = cv2.perspectiveTransform(pt, H_inv)[0][0]

            self.tracks_db.setdefault(t_id, {})[frame_idx] = {
                "box": [float(v) for v in box],
                "kpts": [[float(x), float(y), float(c)] for x, y, c in kpts] if kpts is not None else None,
                "real": (float(real_xy[0]), float(real_xy[1])),
            }

    # ==========================================
    # Pass 2a: player selection
    # ==========================================
    def select_players(self, frame_height=None):
        """Picks the far-side and near-side track IDs.

        Selection score per track = sum over every frame it was seen of
        max(0, BASELINE_SEARCH_RADIUS_M - distance_to_nearest_baseline). A chair umpire or
        line judge near the net accumulates ~0 score all clip, since they're ~11-12m from
        either baseline; an actual player racks up score every frame they spend near their
        own baseline. This also naturally favors a track that's both long-lived AND
        consistently close to a baseline over one lucky close frame.

        Once the winning track is found for each real-world side, far/near is assigned by
        comparing average on-screen pixel y (whichever sits higher in frame -- smaller
        pixel y -- is the far side), so this doesn't depend on which physical corner the
        court model happened to index first. If only one side had a qualifying track,
        frame_height is used as a fallback to decide far vs. near from absolute position.

        Returns (far_id, near_id); either may be None if nothing qualified.
        """
        radius = config.BASELINE_SEARCH_RADIUS_M
        scored = []

        for t_id, frames in self.tracks_db.items():
            if len(frames) < 3:  # ignore one-off blips (a ball kid crossing frame, etc.)
                continue

            coords = np.array([f["real"] for f in frames.values()])
            avg_y = float(np.mean(coords[:, 1]))
            side = "top" if avg_y > 0 else "bottom"
            anchor = _TOP_BASELINE if side == "top" else _BOTTOM_BASELINE

            distances = np.linalg.norm(coords - anchor, axis=1)
            score = float(np.sum(np.maximum(0.0, radius - distances)))
            avg_px_y = float(np.mean([f["box"][3] for f in frames.values()]))  # feet, on screen

            scored.append({"id": t_id, "side": side, "score": score, "avg_px_y": avg_px_y})

        top_cands = sorted([t for t in scored if t["side"] == "top"], key=lambda x: x["score"], reverse=True)
        bot_cands = sorted([t for t in scored if t["side"] == "bottom"], key=lambda x: x["score"], reverse=True)

        chosen = []
        if top_cands and top_cands[0]["score"] > 0:
            chosen.append(top_cands[0])
        if bot_cands and bot_cands[0]["score"] > 0:
            chosen.append(bot_cands[0])

        if not chosen:
            return None, None

        if len(chosen) == 1:
            only = chosen[0]
            is_far = frame_height is not None and only["avg_px_y"] < frame_height / 2.0
            return (only["id"], None) if is_far else (None, only["id"])

        chosen.sort(key=lambda x: x["avg_px_y"])
        far_id, near_id = chosen[0]["id"], chosen[1]["id"]
        return far_id, near_id

    # ==========================================
    # Pass 2b: EMA smoothing + gap-fill for a selected track
    # ==========================================
    def build_render_track(self, track_id):
        """Pre-computes a frame_idx -> {"bbox", "keypoints"} dict for one selected track,
        with EMA smoothing and short-gap hold-through, so playback doesn't jitter or blink
        out on a single missed detection. Same state machine the old single-slot version
        used, just applied to a chosen track's already-collected per-frame data."""
        if track_id is None or track_id not in self.tracks_db:
            return {}

        frames = self.tracks_db[track_id]
        last_frame = max(frames.keys())

        rendered = {}
        state = {"box": None, "kpts": None, "miss": 0}

        for frame_idx in range(last_frame + 1):
            entry = frames.get(frame_idx)

            if entry is not None:
                new_box, new_kpts = entry["box"], entry["kpts"]
                if state["box"] is not None:
                    final_box = [self.alpha * n + (1 - self.alpha) * o
                                 for n, o in zip(new_box, state["box"])]
                    if new_kpts is not None and state["kpts"] is not None:
                        final_kpts = [[self.alpha * nk[0] + (1 - self.alpha) * ok[0],
                                       self.alpha * nk[1] + (1 - self.alpha) * ok[1],
                                       nk[2]] for nk, ok in zip(new_kpts, state["kpts"])]
                    else:
                        final_kpts = new_kpts
                else:
                    final_box, final_kpts = new_box, new_kpts

                state["box"], state["kpts"], state["miss"] = final_box, final_kpts, 0
            else:
                state["miss"] += 1
                if state["miss"] <= self.max_gap and state["box"] is not None:
                    final_box, final_kpts = state["box"], state["kpts"]
                else:
                    state["box"], state["kpts"] = None, None
                    final_box, final_kpts = None, None

            if final_box is not None:
                rendered[frame_idx] = {"bbox": final_box, "keypoints": final_kpts}

        return rendered

    @staticmethod
    def draw(annotated_frame, render_entry, is_far):
        """Draws one already-smoothed render entry (see build_render_track) onto a frame."""
        if render_entry is None:
            return

        box, kpts = render_entry["bbox"], render_entry["keypoints"]
        cv2.rectangle(annotated_frame, (int(box[0]), int(box[1])),
                      (int(box[2]), int(box[3])), (0, 0, 255), 2)

        pt_color = (0, 255, 0) if is_far else (0, 255, 255)
        if kpts:
            for kx, ky, kconf in kpts:
                if kconf > 0.3:
                    cv2.circle(annotated_frame, (int(kx), int(ky)), 4, pt_color, -1)
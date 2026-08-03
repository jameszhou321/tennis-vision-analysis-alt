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
import court_detector

# Real-world baseline anchors (meters, net center = origin) -- must match
# court_detector.COURT_14_PTS_PHYSICAL's convention.
_TOP_BASELINE = np.array([0.0, 11.885])
_BOTTOM_BASELINE = np.array([0.0, -11.885])


class PoseTracker:
    def __init__(self, pose_model, court_detector, pose_model_crop=None):
        self.pose_model = pose_model
        self.court_detector = court_detector
        # Separate model instance for the supplemental far-half crop pass (see _track_far_crop).
        # Must be a genuinely different YOLO() instance than pose_model: Ultralytics registers
        # its tracker callbacks once per model instance, and they fire on every subsequent
        # predict()/track() call from that instance regardless of mode -- running this pass on
        # the same instance already used for .track() would silently feed crop-local detections
        # into the full-frame BoT-SORT tracker's association state.
        self.pose_model_crop = pose_model_crop
        self.alpha = config.POSE_ALPHA
        self.max_gap = config.POSE_MAX_GAP

        # track_id -> {frame_idx: {"box": [x1,y1,x2,y2], "kpts": [[x,y,conf],...] or None,
        #                          "real": (x_m, y_m)}}
        self.tracks_db = {}
        self._first_track_call = True

        # Far-crop pass pseudo-tracks (no BoT-SORT involved): "crop_N" -> {"last_real", "last_frame"}
        self._crop_tracks = {}
        self._next_crop_id = 0

    def reset_for_cut(self):
        """Call this at a detected hard camera cut (see main.py's scene-cut detection). Forces
        the next track_frame() call to reset BoT-SORT's internal state (persist=False), since
        continuing to match post-cut detections against pre-cut track history is meaningless --
        a track ID "continuing" across a cut would just be a coincidental nearest-match, not the
        same physical person. tracks_db itself is left alone: pre- and post-cut segments of the
        same physical player simply accumulate under separate IDs, same as any other BoT-SORT ID
        switch already produces."""
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

        H_inv = np.linalg.inv(H)

        # persist=False on this clip's first tracked frame forces BoT-SORT to reset its
        # internal state -- required because main.py reuses the same pose_model instance
        # across every rally clip in a match, and without this the tracker would otherwise
        # carry track IDs/history over from the *previous* clip's last frame.
        res = self.pose_model.track(frame, persist=not self._first_track_call,
                                     tracker="botsort.yaml", classes=[0],
                                     imgsz=config.YOLO_IMGSZ, conf=config.POSE_TRACK_CONF,
                                     verbose=False)[0]
        self._first_track_call = False

        frame_real_positions = []

        if res.boxes is not None and res.boxes.id is not None:
            track_ids = res.boxes.id.int().cpu().tolist()
            boxes = res.boxes.xyxy.cpu().numpy()
            kpts_all = res.keypoints.data.cpu().numpy() if res.keypoints is not None else None

            for i, t_id in enumerate(track_ids):
                box = boxes[i]
                kpts = kpts_all[i] if kpts_all is not None else None
                feet_px = self._feet_px(box, kpts)

                pt = np.array([[[feet_px[0], feet_px[1]]]], dtype=np.float32)
                real_xy = cv2.perspectiveTransform(pt, H_inv)[0][0]
                real_xy = (float(real_xy[0]), float(real_xy[1]))

                self.tracks_db.setdefault(t_id, {})[frame_idx] = {
                    "box": [float(v) for v in box],
                    "kpts": [[float(x), float(y), float(c)] for x, y, c in kpts] if kpts is not None else None,
                    "real": real_xy,
                }
                frame_real_positions.append(real_xy)

        if self.pose_model_crop is not None:
            far_bbox = court_detector.project_far_half_pixel_bbox(H, frame.shape, config.FAR_CROP_MARGIN_FRAC)
            if far_bbox is not None:
                self._track_far_crop(frame, frame_idx, H_inv, far_bbox, frame_real_positions)

    @staticmethod
    def _feet_px(box, kpts):
        """Ankles (COCO indices 15/16), falling back to bbox bottom-center when they weren't
        detected confidently -- same fallback offline_tennis_tracker.py uses."""
        if kpts is not None and kpts[15][2] > 0.2 and kpts[16][2] > 0.2:
            return (kpts[15][:2] + kpts[16][:2]) / 2.0
        return np.array([(box[0] + box[2]) / 2.0, box[3]])

    def _track_far_crop(self, frame, frame_idx, H_inv, far_bbox, frame_real_positions):
        """Supplemental far-half detection pass: runs a second, plain (untracked) YOLO-pose call
        on a crop of just the far half of the frame, at a lower confidence floor
        (POSE_TRACK_CONF_FAR), to recover far-player detections the full-frame pass misses due to
        their smaller effective on-screen size in wide/broadcast framing. Restores the old
        per-side sensitivity advantage without reintroducing the old pixel-space scoring bug --
        select_players' geometric baseline-distance scoring remains the only thing that decides
        who's a real player; this pass just supplies additional candidate detections.

        Detections that duplicate a full-frame track already seen this same frame are skipped
        (the full-frame pass already caught that person; a duplicate would split one player's
        score across two track IDs in select_players). Surviving detections are associated
        frame-to-frame into "crop_N" pseudo-tracks via simple nearest-position gating -- no
        BoT-SORT involved here (see __init__ for why this pass must stay untracked).
        """
        x1, y1, x2, y2 = far_bbox
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return

        res = self.pose_model_crop.predict(crop, imgsz=config.YOLO_IMGSZ,
                                            conf=config.POSE_TRACK_CONF_FAR, classes=[0],
                                            verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0:
            return

        boxes = res.boxes.xyxy.cpu().numpy()
        kpts_all = res.keypoints.data.cpu().numpy() if res.keypoints is not None else None
        match_dist = config.FAR_CROP_MAX_MATCH_DIST_M
        offset = np.array([x1, y1, x1, y1], dtype=np.float32)

        for i in range(len(boxes)):
            box = boxes[i] + offset
            kpts = kpts_all[i].copy() if kpts_all is not None else None
            if kpts is not None:
                kpts[:, 0] += x1
                kpts[:, 1] += y1

            feet_px = self._feet_px(box, kpts)
            pt = np.array([[[feet_px[0], feet_px[1]]]], dtype=np.float32)
            real_xy = cv2.perspectiveTransform(pt, H_inv)[0][0]
            real_xy = (float(real_xy[0]), float(real_xy[1]))

            is_redundant = any(np.hypot(real_xy[0] - rx, real_xy[1] - ry) < match_dist
                               for rx, ry in frame_real_positions)

            # Always keep the matched pseudo-track's bookkeeping alive, even when this
            # particular frame's detection is redundant with a full-frame track -- otherwise a
            # far player who's caught alternately by the full-frame pass and this crop pass gets
            # fragmented across several short-lived crop_N tracks instead of one continuous one
            # (once the gap since the pseudo-track's last *write* exceeds FAR_CROP_TRACK_MAX_GAP,
            # a stretch of frames where the full-frame pass "took over" would otherwise sever it),
            # none of which individually accumulates enough baseline-proximity score to win
            # select_players() over a false candidate.
            crop_id = self._match_crop_track(real_xy, frame_idx)
            self._crop_tracks[crop_id] = {"last_real": real_xy, "last_frame": frame_idx}

            if is_redundant:
                continue  # full-frame pass already recorded this person this frame; don't duplicate

            self.tracks_db.setdefault(crop_id, {})[frame_idx] = {
                "box": [float(v) for v in box],
                "kpts": [[float(x), float(y), float(c)] for x, y, c in kpts] if kpts is not None else None,
                "real": real_xy,
            }

    def _match_crop_track(self, real_xy, frame_idx):
        """Greedy nearest-position association for the far-crop pass's pseudo-tracks. Returns an
        existing "crop_N" id if a live pseudo-track is within FAR_CROP_MAX_MATCH_DIST_M and hasn't
        gone unmatched for more than FAR_CROP_TRACK_MAX_GAP frames, otherwise starts a new one."""
        match_dist = config.FAR_CROP_MAX_MATCH_DIST_M
        max_gap = config.FAR_CROP_TRACK_MAX_GAP

        best_id, best_dist = None, match_dist
        for cid, state in self._crop_tracks.items():
            if frame_idx - state["last_frame"] > max_gap:
                continue
            rx, ry = state["last_real"]
            dist = float(np.hypot(real_xy[0] - rx, real_xy[1] - ry))
            if dist < best_dist:
                best_id, best_dist = cid, dist

        if best_id is not None:
            return best_id

        crop_id = f"crop_{self._next_crop_id}"
        self._next_crop_id += 1
        return crop_id

    def _stitch_tracks(self):
        """Merges tracks -- both full-frame BoT-SORT ints and far-crop "crop_N" pseudo-tracks --
        that are spatiotemporally continuous into single logical chains, before scoring in
        select_players(). Both sources independently re-issue a new ID whenever they briefly
        lose and reacquire the same physical player (marginal confidence flicker, brief
        occlusion, a camera-cut reset), fragmenting what should be one long track across many
        short-lived IDs -- none of which individually accumulates enough baseline-proximity
        score to reliably beat a false candidate. This is a superset of the far-crop pass's own
        internal continuity tracking (_match_crop_track): that only reconnects crop_N segments
        to each other, this also reconnects full-frame segments to each other and across the two
        sources, using the same greedy nearest-position-within-a-time-window logic.

        A track is greedily chained onto the best still-open chain on the same real-world side
        (top/bottom by real-world y sign) if the gap since that chain's last frame is within
        TRACK_STITCH_MAX_GAP_FRAMES and the position at that point is within
        TRACK_STITCH_MAX_DIST_M of where the new track starts -- a real player can't teleport,
        but two different simultaneous people (or noise) generally won't satisfy both bounds at
        once. Returns {chain_id: {"side": "top"|"bottom", "frames": {frame_idx: entry}}}.
        """
        max_gap = config.TRACK_STITCH_MAX_GAP_FRAMES
        max_dist = config.TRACK_STITCH_MAX_DIST_M

        entries = []
        for t_id, frames in self.tracks_db.items():
            if not frames:
                continue
            sorted_idxs = sorted(frames.keys())
            coords = np.array([frames[i]["real"] for i in sorted_idxs])
            side = "top" if float(np.mean(coords[:, 1])) > 0 else "bottom"
            entries.append({
                "side": side, "start": sorted_idxs[0], "end": sorted_idxs[-1],
                "start_real": frames[sorted_idxs[0]]["real"], "end_real": frames[sorted_idxs[-1]]["real"],
                "frames": frames,
            })

        stitched = {}
        next_id = 0
        for side in ("top", "bottom"):
            side_entries = sorted((e for e in entries if e["side"] == side), key=lambda e: e["start"])
            chains = []  # {"end", "end_real", "frames": merged dict, "id"}

            for e in side_entries:
                best_chain, best_dist = None, max_dist
                for chain in chains:
                    gap = e["start"] - chain["end"]
                    if gap < 0 or gap > max_gap:
                        continue
                    dist = float(np.hypot(e["start_real"][0] - chain["end_real"][0],
                                           e["start_real"][1] - chain["end_real"][1]))
                    if dist < best_dist:
                        best_chain, best_dist = chain, dist

                if best_chain is not None:
                    best_chain["frames"].update(e["frames"])
                    best_chain["end"] = e["end"]
                    best_chain["end_real"] = e["end_real"]
                else:
                    chain_id = f"chain_{next_id}"
                    next_id += 1
                    chains.append({"end": e["end"], "end_real": e["end_real"],
                                    "frames": dict(e["frames"]), "id": chain_id})

            for chain in chains:
                stitched[chain["id"]] = {"side": side, "frames": chain["frames"]}

        return stitched

    # ==========================================
    # Pass 2a: player selection
    # ==========================================
    def select_players(self, frame_height=None):
        """Picks the far-side and near-side track IDs.

        Tracks are first stitched together across ID fragmentation (_stitch_tracks) -- both the
        full-frame and far-crop passes independently re-issue a new ID whenever they briefly
        lose and reacquire the same physical player, and scoring needs one continuous chain per
        real player to reliably beat a false candidate, not many small fragments. The winning
        chains are written back into tracks_db under their chain_id so build_render_track can
        read them exactly like any other track.

        Selection score per chain = sum over every frame it was seen of
        max(0, BASELINE_SEARCH_RADIUS_M - distance_to_nearest_baseline). A chair umpire or
        line judge near the net accumulates ~0 score all clip, since they're ~11-12m from
        either baseline; an actual player racks up score every frame they spend near their
        own baseline. This also naturally favors a chain that's both long-lived AND
        consistently close to a baseline over one lucky close frame.

        Once the winning chain is found for each real-world side, far/near is assigned by
        comparing average on-screen pixel y (whichever sits higher in frame -- smaller
        pixel y -- is the far side), so this doesn't depend on which physical corner the
        court model happened to index first. If only one side had a qualifying chain,
        frame_height is used as a fallback to decide far vs. near from absolute position.

        Returns (far_id, near_id); either may be None if nothing qualified.
        """
        radius = config.BASELINE_SEARCH_RADIUS_M
        stitched = self._stitch_tracks()
        self.tracks_db.update({chain_id: chain["frames"] for chain_id, chain in stitched.items()})

        scored = []

        for chain_id, chain in stitched.items():
            frames = chain["frames"]
            if len(frames) < 3:  # ignore one-off blips (a ball kid crossing frame, etc.)
                continue

            coords = np.array([f["real"] for f in frames.values()])
            side = chain["side"]
            anchor = _TOP_BASELINE if side == "top" else _BOTTOM_BASELINE

            distances = np.linalg.norm(coords - anchor, axis=1)
            score = float(np.sum(np.maximum(0.0, radius - distances)))
            avg_px_y = float(np.mean([f["box"][3] for f in frames.values()]))  # feet, on screen

            scored.append({"id": chain_id, "side": side, "score": score, "avg_px_y": avg_px_y})

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
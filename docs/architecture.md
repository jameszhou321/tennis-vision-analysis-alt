# Tennis Match Analysis Project — File Manifest

> Graduation Project: Computer Vision-Based Automatic Tennis Match Analysis System
> Updated: 2026-08-03 — `court_detector.py`/`pose_tracker.py` rewritten from the old Hough-line ROI approach to homography-based real-world tracking; `main.py` gained scene-cut detection and off-court region rejection. Supersedes the 2026-07-25 version.

---

## Directory Structure Overview

```
tennis-vision-analysis/
├── src/                          Source code
│   ├── main.py                   Batch pipeline entry point (rally segmentation + cutting + overlay rendering)
│   ├── config_legacy.py          Configuration for main.py (paths, thresholds, EMA parameters)
│   ├── court_detector.py         14-keypoint YOLO model → weighted homography (pixel↔real-world meters)
│   ├── pose_tracker.py           Full-frame BoT-SORT + pose, projected into real-world court coords,
│   │                              picks far/near by baseline proximity; EMA smoothing + gap filling
│   ├── ball_tracker.py           Classical CV ball tracker (background subtraction + Kalman filter)
│   ├── ball_tracker_tracknet.py  TrackNet-backed ball tracker (higher accuracy, optional setup)
│   ├── audio_video_fusion.py     Audio impact detection + WAITING/POINT_ACTIVE hysteresis state machine
│   ├── tracknet/                 TrackNet model files (from upstream repo) + pretrained weights
│   ├── train_court_pipeline.py   Court 14-keypoint model training entry point
│   ├── test_person_detector.py   Full-scale evaluation script for the person detection model
│   ├── pipeline/                 Offline precision tracking + court annotation tooling
│   ├── model/
│   │   ├── mst/                  MSTFormer action-recognition model (core contribution)
│   │   └── yolo/                 Single-frame YOLO action-classification baseline
│   ├── demo/                     PyQt5 desktop demo (playback + timeline + live inference)
│   ├── utils/                    Annotation, data-processing, evaluation, and thesis-figure scripts
│   └── training/                 Person detector training + hard-negative mining
├── configs/                      YAML training configs (court/person YOLO + MSTFormer)
├── docs/                         Architecture notes, style guide, figures
├── requirements.txt / pyproject.toml / uv.lock   Dependency manifests
├── LICENSE
└── README.md / README_zh.md

Not checked into the repo (create locally; paths follow config_legacy.py / configs/):
├── models/                       Model weights (yolo/, court/, action/, person/, tracknet/)
├── data/                         Datasets (rallies_new/, rallies_annotated/, rallies_annotating/,
│                                  rallies_train/, court_finetune/, person_sorter/)
├── videos/                       Raw match videos
├── runs/                         Training run records
├── results/                      Analysis outputs & demo videos
└── logs/                         Pipeline run logs
```

All of `models/`, `data/`, `videos/`, `runs/`, `logs/`, and `results/` are excluded via `.gitignore` (see `docs/style_guide.md` §4) — this is a source/config/docs-only repository; large assets must be prepared separately.

---

## 1. Source Code (`src/`)

### Top-Level Files (Batch Processing Pipeline)

| File | Purpose |
| --- | --- |
| `src/main.py` | Batch video-processing entry point. Segments each match video into rallies using one of three interchangeable modes — `"static"` (fixed fence-cam, player-box velocity), `"broadcast"` (TV footage, PySceneDetect + CLIP scene classification), or `"fusion"` (weighted combination of audio impact detection, player motion, and ball activity via a hysteresis state machine) — then cuts/concatenates the clips with FFmpeg into `data/rallies_new/<video_name>/`, optionally re-rendering each clip with pose skeleton + ball trail overlays into an `annotated/` subfolder. Both `annotate_rally_clip()` and `process_fusion_clip()` also detect hard scene cuts (`detect_scene_cut_frames`, PySceneDetect `ContentDetector`) and reset the court/pose tracker state at each cut; `process_fusion_clip()` additionally rejects off-court detections (`is_within_court_region`) from the motion/ball fusion signals |
| `src/config_legacy.py` | Configuration for `main.py`: video/output/model paths, scout skip-frame/scale, EMA smoothing coefficient, pose/homography confidence thresholds, far-crop supplemental-pass and cross-source track-stitching parameters, scene-cut and off-court-region thresholds, checkpoint/control file locations |
| `src/court_detector.py` | Court homography detector (`CourtDetector`). Detects the 14 standard court keypoints with a YOLO keypoint model and fits a weighted, temporally-smoothed pixel↔real-world (meters) homography (RANSAC seed + Levenberg-Marquardt refinement); tolerates brief missed detections but drops back to `None` past `MAX_HOMOGRAPHY_STALE_FRAMES`. Replaces the old Hough-line ROI split; same approach as `src/pipeline/offline_tennis_tracker.py`. Called by `main.py` |
| `src/pose_tracker.py` | Pose tracker (`PoseTracker`). Full-frame person+pose tracking (persistent IDs via BoT-SORT), projected into real-world court coordinates via `CourtDetector`'s homography, plus a supplemental far-half crop pass and cross-source track stitching; selects the far/near player by which (stitched) track spends the most time near a baseline over the clip, then EMA-smooths and gap-fills each. Replaces the old per-ROI-crop, multi-term-scoring approach (confidence + inertia + court proximity + local motion), which a chair umpire near the net could win against a far player deep near their own baseline; called by `main.py` |
| `src/ball_tracker.py` | Classical CV ball tracker: background subtraction + circular-blob filtering + Kalman-filtered trajectory. No setup required; the default fallback backend |
| `src/ball_tracker_tracknet.py` | TrackNet-backed ball tracker wrapper — same interface as `ball_tracker.py` but higher accuracy; requires `tracknet/` model files + pretrained weights to be downloaded manually (see root `README.md`) |
| `src/audio_video_fusion.py` | Band-pass filtered audio impact (onset) detection + a WAITING/POINT_ACTIVE hysteresis state machine; fuses audio, player-motion, and ball-activity scores into rally boundaries. Used by `main.py`'s `process_fusion_clip()` (`mode="fusion"`) |
| `src/tracknet/` | TrackNet model architecture files (`model.py`, `general.py`), downloaded from the [upstream TrackNet repo](https://github.com/yastrebksv/TrackNet) rather than authored here; imported by `ball_tracker_tracknet.py`. If missing, `main.py` automatically falls back to the classical CV tracker with a console warning |
| `src/train_court_pipeline.py` | Court keypoint model training entry point. Prepares the dataset YAML → launches YOLO-pose fine-tuning → exports bad cases for iterative improvement |
| `src/test_person_detector.py` | Full-scale evaluation of the player/person classification model: runs inference across all train + val images and produces accuracy reports and verification summaries |

### Core Tracking Pipeline (`src/pipeline/`)

| File | Purpose |
| --- | --- |
| `offline_tennis_tracker.py` | **Core module.** Two-pass offline tracking, finer-grained than `main.py`'s batch scanning: Pass 1 computes a weighted homography matrix frame by frame using the court keypoint model and tracks players with BoT-SORT, projecting them into court coordinates; Pass 2 renders court lines, player boxes, and an overhead radar-view plot into an annotated output video |
| `generate_trajectory.py` | Extracts player coordinate sequences from tracking results, generating temporal trajectories for action recognition |
| `debug_vision.py` | Visualization debugging tool: overlays court detection and player tracking results onto video frames to verify pipeline output |
| `smart_extract_14pts.py` | Smart sampling annotation tool. Intelligently samples frames from match videos and pre-labels 14 keypoints using the existing court model, generating training candidates |
| `corner_driven_refine_tool.py` | Interactive GUI for court annotation refinement: drag the 4 corner points and the other 10 keypoints are computed automatically; supports negative-sample annotation |
| `prepare_weighted_dataset.py` | Dataset merging tool: merges newly annotated data with the old dataset and splits into train/val by ratio, for court model fine-tuning |
| `test_weighted_inference.py` | Inference test script; uses an OpenCV tracker to test object tracking on video, verifying the inference pipeline |

The court production workflow: `smart_extract_14pts.py` → `corner_driven_refine_tool.py` → `prepare_weighted_dataset.py` → `../train_court_pipeline.py`. The physical coordinates (meters) for the 14 court points are defined in `COURT_14_PTS_PHYSICAL` at the top of `offline_tennis_tracker.py` and `train_court_pipeline.py`.

### Action Recognition Model — MSTFormer (`src/model/mst/`)

MSTFormer (Multi-Stream Transformer) is the project's core contribution. It fuses **player pose sequences**, **court geometry**, and **multi-stream visual crops** into a single Transformer with a **dual-head output**: 5-class action classification (Idle/Forehand/Backhand/Serve/Movement) + binary keyframe detection.

| File | Purpose |
| --- | --- |
| `model_main.py` | Model definition, `MSTFormer`. Three visual streams (full frame + player1 + player2) → optional merge (`merge_visual_tokens`, shared cross-attention resampler) → concatenated with pose tokens → Transformer → dual-head output. Toggles: `use_pose` / `use_player_crops` / `use_visual` / `merge_visual_tokens` / `parallel_backbones` |
| `dataset.py` | Dataset class `TennisActionDataset`. Reads `pose_data.json` + `annotations.json`, slices via sliding window, builds the 125-dimensional pose vector and the three visual frame streams (uint8, normalized on GPU); supports image augmentation (color jitter / Gaussian noise / blur / random erasing / semi-transparent overlay) |
| `train.py` | Training entry point. Jointly trains action classification + keyframe detection with AMP + gradient accumulation, splits train/val by video, prints action Accuracy and keyframe Precision/Recall per epoch. Each run saves `best.pth`, `final.pth`, `train_log.csv`, and a `config.yaml` snapshot under `models/action/<config>/<timestamp>/` |
| `config.py` | YAML config parser; resolves relative paths to absolute, fills in device and gradient-accumulation steps |
| `augment.py` | Asynchronous image-augmentation buffer; moves augmentation off the DataLoader worker process and into a separate thread pool |
| `extract_frames.py` | Pre-extracts full frames from rally videos into `frames/` to speed up image reading during training |
| `extract_crops.py` | Pre-extracts player1/player2 crop images into `player1/`, `player2/` |
| `run_ablation.py` | Batch-runs the experiments under `configs/ablation`, `configs/components`, `configs/hyperparams` |
| `modules/` | Model submodules (see below) |
| `tests/` | `eval_optimal.py` (evaluation + confusion matrix, `--config`/`--weights` args), `test_matrix.py`, `test_dataset.py` |

**`modules/` submodules**

| File | Purpose |
| --- | --- |
| `backbone_factory.py` | Visual backbone factory; builds one of the four backbones below from the `visual_backbone` config |
| `yolo_extractor.py` | YOLO11 backbone (primary): taps P3/P4/P5 and applies cross-scale attention fusion |
| `resnet_extractor.py` | ResNet18 backbone (comparison) |
| `vit_extractor.py` | Lightweight ViT patch-embedding backbone (comparison) |
| `raw_extractor.py` | Raw pixel projection backbone (comparison) |
| `token_resampler.py` | Perceiver-style cross-attention resampler; compresses an arbitrary number of tokens down to a fixed count |
| `pos_encoding.py` | Sinusoidal positional encoding (disabled by default via `use_pos_encoding=false`) |
| `action_head.py` | `ActionClassificationHead` (5-class) + `KeyframeDetectionHead` (binary), both LN → Linear → GELU → Dropout → Linear |

125-dim pose vector breakdown: 17×3 absolute keypoint coords+conf, 17×2 relative to person center, 2 person-center-relative-to-court, 2 velocity, 2 acceleration, 6 ball (reserved), 28 (14 court points × 2).

Full field-by-field config documentation lives in [`configs/CONFIG_REFERENCE.md`](../configs/CONFIG_REFERENCE.md).

### Single-Frame YOLO Baseline (`src/model/yolo/`)

A comparison baseline that treats action recognition as single-frame image classification (YOLO11n backbone + global average pooling + classification head, 5 classes), used to illustrate the shortcomings of frame-by-frame classification versus MSTFormer's temporal modeling.

| File | Purpose |
| --- | --- |
| `model.py` | `YoloFrameClassifier` — hooks into the final YOLO backbone output, followed by a classification head |
| `dataset.py` | Single-frame dataset: pulls the per-frame action label from rally frames + `annotations.json` |
| `train.py` | Training script (shares `annotations.json` and action-class definitions with `model/mst/`) |

### Demo Application (`src/demo/`)

PyQt5 desktop app combining video playback, a three-row timeline, and real-time MSTFormer inference visualization.

| File | Purpose |
| --- | --- |
| `main.py` | Entry point. Parses `--rally`/`--config`/`--weights`/`--person`/`--pose`; handles CUDA DLL and Qt-plugin load-order on Windows |
| `app.py` | Main window: video playback, three-row timeline, file/model selection, inference trigger, action legend |
| `player.py` | Video player: QTimer + OpenCV frame-by-frame reading, handles paths containing Chinese characters (short-path conversion on Windows) |
| `timeline.py` | Three-row timeline panel: GT annotation bar / prediction bar / frame grid bar, with a cursor following playback position |
| `inference.py` | Inference thread (QThread). Two modes: ① `--person`/`--pose` supplied → real-time detection with bbox + skeleton overlay drawn on frame; ② not supplied → falls back to reading pre-extracted `pose_data.json` + crop images. Both modes feed the whole sequence into MSTFormer at once |
| `seq_len_sweep.py` | Sequence-length sweep script: iterates over different `seq_len` values and outputs an accuracy CSV |

### Person-Detector Training (`src/training/`)

Trains the near-side/far-side player identification model, with accompanying hard-negative mining.

| File | Purpose |
| --- | --- |
| `train_person_detector.py` | Fine-tunes YOLO on `data/person_sorter/` to distinguish `player_near` / `player_far` |
| `merge_hard_negatives.py` | Merges mined hard examples (misdetected ball kids/spectators, etc.) into the training set |
| `yolo-train-legacy.py` | Superseded training script, kept for reference only |

### Utility Scripts (`src/utils/`)

Organized into six functional groups; most run from the repository root.

**Annotation tools**

| File | Purpose |
| --- | --- |
| `action_annotator.py` | **Action temporal annotation** (Flask web app, `http://localhost:5000`). Extracts clips from `data/rallies_new/` in rotation by video source into `data/rallies_annotating/`, lets you label 5 action-type time segments in the browser, saves as `annotations.json`. Supports deleting clips (auto-refills) and progress persistence (`_progress.json`) |
| `label_tool.py` | Player bounding-box annotation (OpenCV GUI): drag to draw a box, label near/far player, save in YOLO txt format |

**Data production and processing**

| File | Purpose |
| --- | --- |
| `data-batch-extractor.py` | Batch rally data extraction: iterates over `data/rallies_new/`, runs court + pose detection on each rally, outputs `tracking_data.json`, with progress logging and resume support |
| `data-creater.py` | Person-classification sampling: randomly samples frames from `data/rallies_new/` into `data/person_sorter/image/` |
| `dataset_splitter.py` | Splits `data/person_sorter/` images into train/val by ratio |
| `prepare_train_dataset.py` | Copies files needed for training from `rallies_annotated/` to `rallies_train/` (`raw_clip.mp4`, `pose_data.json`, `annotations.json`), with resume support |
| `merge_annotating_data.py` | Merges newly annotated data from `rallies_annotating/` into `rallies_annotated/`, converting `tracking_data.json` → `pose_data.json` (including the `court` field) |
| `add_court_to_pose.py` | Adds court keypoints to older data: runs the court model frame by frame, writes the 14 points into the `court` field of `pose_data.json`, with resume support |
| `rerun_pose_detection.py` | Reruns pose detection on player crop images at a low threshold (0.1), maps coordinates back to the original frame and filters using the person bbox |
| `trim_waiting_segments.py` | Trims overly long "Idle" segments to alleviate class imbalance |

**Inference visualization and testing**

| File | Purpose |
| --- | --- |
| `inference_viewer.py` | Person-classification model inference visualization |
| `test_person_on_video.py` / `visualize_person_test.py` | Tests person detection on video and visualizes results |
| `visualize_data_quality.py` | Data-quality visualization (whether annotations/poses look normal) |
| `side_by_side_viewer.py` | Side-by-side comparison view of multiple results |
| `broadcast_detector.py` | Detects/classifies broadcast-style footage (used by `main.py`'s `"broadcast"` segmentation mode) |

**Evaluation and reporting**

| File | Purpose |
| --- | --- |
| `batch_eval_all.py` | Batch-evaluates all trained models |
| `generate_model_report.py` | Aggregates metrics across models into a report |
| `analyze_class_distribution.py` | Computes the distribution of action classes |

**Hard-example mining**

| File | Purpose |
| --- | --- |
| `hard_negative_extractor.py` | Mines hard negative samples from false detections |
| `hard_negative_reviewer.py` | Manual review of hard examples |

**Thesis figures** (writing-only, unrelated to system operation)

`generate_thesis_figures.py`, `generate_ch3_figures.py`, `generate_confusion_figures.py`, `generate_confusion_matrices.py`, `create_thesis_figure_N.py`, `extract_forehand_frame.py`, `unify_citations.py` — generate training curves, confusion matrices, and other figures for the thesis, output to `docs/figures/`. These require the local (unpublished) dataset to reproduce.

---

## 2. Model Weights (`models/`, not in repo)

| Path | Purpose |
| --- | --- |
| `models/yolo/yolo11x-pose.pt` | YOLO11x pose estimation model (primary, highest accuracy) |
| `models/yolo/*-pose.pt` | Additional/backup YOLO pose variants |
| `models/court/best.pt` | Best court 14-point keypoint detection weights (YOLO-pose fine-tuned) |
| `models/person/best.pt` | Best person classification weights (near/far player) |
| `models/action/<config>/<timestamp>/best.pth` | Action recognition weights, produced by `model/mst/train.py` |
| `models/tracknet/model_best.pt` | TrackNet pretrained weights (manual download, optional ball-tracking backend) |

---

## 3. Datasets (`data/`, not in repo)

| Path | Contents | Notes |
| --- | --- | --- |
| `data/rallies_new/` | Rally clips cut by `main.py` (`raw_clip.mp4` + optional `annotated/` overlay), organized by source video | Produced by the batch segmentation pipeline; source for annotation |
| `data/rallies_annotating/` | Annotation workspace | Staging directory populated by `action_annotator.py` from `rallies_new/`, organized into subfolders by video source; contains `_progress.json` |
| `data/rallies_annotated/` | Rally clips with `annotations.json` | Manually annotated rallies used for training/evaluating the action model. Each rally contains `raw_clip.mp4`, `pose_data.json` (including the `court` field), and `annotations.json` |
| `data/rallies_train/` | Training data copied from `rallies_annotated/` | `raw_clip.mp4` + `pose_data.json` + `annotations.json`, produced by `prepare_train_dataset.py`; may also hold pre-extracted `frames/`, `player1/`, `player2/` from `extract_frames.py`/`extract_crops.py` |
| `data/court_finetune/` | Images + YOLO labels + `bad_cases/` | Court 14-keypoint fine-tuning dataset, train/val split |
| `data/person_sorter/` | Images + YOLO labels | Person classification (near/far player) training data |

### `annotations.json` format

```json
[
  {"start_time": 0.0, "end_time": 4.837, "action_name": "Idle", "action_id": 0},
  {"start_time": 4.837, "end_time": 12.78, "action_name": "Serve", "action_id": 3}
]
```

Action categories: `Idle(0)` `Forehand(1)` `Backhand(2)` `Serve(3)` `Movement(4)`

### `pose_data.json` format

```json
[
  {
    "frame": 0,
    "court": [[x, y, conf], ...],
    "near_player": {"bbox": [x1, y1, x2, y2], "keypoints": [[x, y, conf], ...]},
    "far_player":  {"bbox": [x1, y1, x2, y2], "keypoints": [[x, y, conf], ...]}
  }
]
```

- `court`: 14 court keypoints; when conf < 0.3, the point is zeroed out in the feature vector
- `near_player` / `far_player`: 17 COCO skeleton keypoints, detected within the person bbox (low threshold 0.1)
- `_pose_rerun: true`: marks that this frame has already been processed by `rerun_pose_detection.py`

> `src/main.py`'s direct output is video clips (`rally_XXX.mp4`, `all_rallies_combined.mp4`, plus `_annotated` counterparts if enabled) and, in `"fusion"` mode, internally computed audio/motion/ball scores. The `pose_data.json`/`annotations.json` formats above are produced/consumed by the offline pipeline (`src/pipeline/`) and annotation tools (`src/utils/`) for model training, not by `main.py` directly.

---

## 4. Other Directories (not in repo)

| Path | Contents |
| --- | --- |
| `videos/` | Raw match videos (MP4 + ASS subtitles) |
| `configs/` | YAML training configs (tracked in the repo — see §5 below) |
| `runs/court_finetune/` | Court model training records (`weights/best.pt` per version) |
| `runs/yolo/` | YOLO detection/pose training records |
| `results/` | Analysis result charts (confusion matrices), demo videos |
| `logs/` | Pipeline run logs (progress, errors, statistics) |

---

## 5. Training Configuration Files (`configs/`)

Configs split into two groups — YOLO-related (court/person) and MSTFormer action recognition. Full item-by-item field documentation lives in [`configs/CONFIG_REFERENCE.md`](../configs/CONFIG_REFERENCE.md); a shorter overview is in [`configs/README.md`](../configs/README.md).

### Court / Person Classification (Ultralytics YOLO training)

| File | Purpose |
| --- | --- |
| `court_keypoints.yaml` | First-version court keypoint dataset config, 4-corner-point annotation (historical) |
| `court_keypoints_weighted.yaml` | Weighted court keypoint dataset config, built on the first version with merged annotation data (historical) |
| `court_keypoints_ultimate.yaml` | Merges all annotation rounds (historical) |
| `court_14pts_weighted.yaml` | Current primary court config: 14-point annotation format (4 corners + 10 auxiliary points) |
| `person_sorter_dataset.yaml` | Person classification dataset config. 2 classes: `player_near` / `player_far`, data in `data/person_sorter/`, used by `src/training/train_person_detector.py` |

### MSTFormer Action Recognition Configs

Unified baseline across configs: `embed_dim=128`, `depth=8`, `use_pos_encoding=false`.

| File | Description |
| --- | --- |
| `main.yaml` | Current best baseline. Three-stream visual merge (`merge_visual_tokens=true`) + pose + Focal Loss; used directly for official training |
| `main_shared.yaml` | Variant sharing the YOLO backbone across streams |
| `single_frame/sf_main.yaml` | Config for the single-frame YOLO baseline (`src/model/yolo/`) |

**`hyperparams/` — hyperparameter tuning**

| File | Variable | Description |
| --- | --- | --- |
| `hp_embed96.yaml` | `embed_dim=96` | Smaller embedding dimension |
| `hp_embed256.yaml` | `embed_dim=256` | Larger embedding dimension |
| `hp_depth4.yaml` | `depth=4` | Shallow Transformer |
| `hp_depth12.yaml` | `depth=12` | Deep Transformer |
| `hp_vtokens8.yaml` | `visual_tokens=8` | Stronger visual compression |
| `hp_vtokens32.yaml` | `visual_tokens=32` | More visual detail |

**`ablation/` — ablation experiments**

| File | Variable | Description |
| --- | --- | --- |
| `abl_no_pose.yaml` | `use_pose=false` | Remove pose input |
| `abl_no_crops.yaml` | `use_player_crops=false` | Remove player crop images |
| `abl_no_visual.yaml` | `use_visual=false` | Pose only, no visual stream |
| `abl_global_only.yaml` | `use_pose=false` + `use_player_crops=false` | Full-frame visual only |

**`components/` — component comparison**

| File | Variable | Description |
| --- | --- | --- |
| `cmp_focal_loss.yaml` | `loss=focal` | Focal Loss baseline |
| `cmp_ce_loss.yaml` | `loss=cross_entropy` | Cross-entropy loss comparison |
| `cmp_no_merge.yaml` | `merge_visual_tokens=false` | Independent three-stream tokens (longer sequence) |
| `cmp_resnet_backbone.yaml` | `visual_backbone=resnet18` | ResNet18 backbone (ImageNet pretrained) |
| `cmp_frozen_backbone.yaml` | `unfreeze_backbone=false` | Frozen backbone, only Transformer is trained |

There are also `optimal.yaml` and `optimal_full.yaml` at the configs root, used for final-model training sweeps referenced by the thesis figure scripts.

```bash
python src/model/mst/train.py --config configs/main.yaml
python src/model/mst/run_ablation.py        # batch-run ablation/components/hyperparams
```

---

## 6. Module Dependencies

```
main.py
  ├── config_legacy.py           (path/parameter configuration)
  ├── court_detector.py          (court homography via 14-keypoint model)
  ├── pose_tracker.py            (BoT-SORT + pose, projected into real-world court coords)
  ├── ball_tracker.py            (classical CV ball tracking, default)
  ├── ball_tracker_tracknet.py   (TrackNet ball tracking, optional backend)
  │     └── tracknet/            (upstream model architecture + weights)
  └── audio_video_fusion.py      (audio impact + hysteresis state machine, mode="fusion")

train_court_pipeline.py
  └── data/court_finetune/       (training data)
  └── configs/*.yaml             (training configuration)

src/pipeline/offline_tennis_tracker.py
  └── models/yolo/               (YOLO pose/detection models)
  └── models/court/              (court keypoint model)

src/model/mst/train.py
  ├── model_main.py              (model definition)
  ├── modules/backbone_factory.py, action_head.py, pos_encoding.py, token_resampler.py
  ├── dataset.py                 (data loading)
  ├── augment.py                 (async image augmentation)
  └── config.py                  (config parsing)

src/model/yolo/train.py
  ├── model.py                   (YoloFrameClassifier)
  └── dataset.py                 (single-frame dataset)

src/training/train_person_detector.py
  └── data/person_sorter/        (near/far player classification data)

src/utils/action_annotator.py
  └── data/rallies_new/          (source data, organized into subfolders by video source)
  └── data/rallies_annotating/   (workspace, contains _progress.json progress record)

src/demo/main.py
  ├── app.py, player.py, timeline.py, inference.py
  ├── models/action/<config>/<timestamp>/best.pth   (MSTFormer weights)
  └── models/person/best.pt, models/yolo/*-pose.pt  (optional real-time detection mode)
```

---

## 7. Two Main Pipelines at a Glance

```
A. Data production line (video → training samples)
   videos/ ──main.py──▶ data/rallies_new/ (rally clips, + annotated/ overlays if enabled)
                          │
                          ├─ utils/action_annotator.py ─▶ label actions → annotations.json
                          └─ model/mst/extract_crops.py ─▶ player crop images player1/ player2/

B. Model line
   Court keypoints:        train_court_pipeline.py       ─▶ models/court/best.pt
   Person classification:  training/train_person_detector.py ─▶ models/person/best.pt
   Action recognition:     model/mst/train.py             ─▶ models/action/<config>/<timestamp>/best.pth
```

## 8. Annotation Workflow

```
1. Collect videos
   Place raw match videos into videos/

2. Segment into rallies
   main.py (mode="static" | "broadcast" | "fusion") → data/rallies_new/{match_name}/rally_xxx/

3. Annotate court keypoints (for court model fine-tuning)
   pipeline/smart_extract_14pts.py        → pre-label frames
   pipeline/corner_driven_refine_tool.py  → manual refinement
   pipeline/prepare_weighted_dataset.py   → merge into data/court_finetune/
   train_court_pipeline.py                → train new model

4. Annotate player actions (for action recognition model training)
   utils/action_annotator.py → extract clips from rallies_new/ in rotation by source into rallies_annotating/
                                label annotations.json in the browser (supports deletion, progress persistence)
   utils/merge_annotating_data.py → merge into data/rallies_annotated/
   utils/prepare_train_dataset.py → copy into data/rallies_train/
   model/mst/extract_frames.py, model/mst/extract_crops.py → pre-extract visual data (optional, speeds up training)
   model/mst/train.py --config configs/main.yaml → train MSTFormer

5. Annotate person classification (for near/far player identification)
   utils/data-creater.py       → sample images
   utils/label_tool.py         → label bounding boxes
   utils/dataset_splitter.py   → split train/val
   training/train_person_detector.py → train classification model
   training/merge_hard_negatives.py  → fold in mined hard negatives (utils/hard_negative_extractor.py)
```

---

## 9. Not Currently in This Snapshot

The previous version of this document described a gitignored `_archive/` directory (legacy Hough-transform experiments, superseded annotation tools, archived training data/configs). That directory is excluded from version control (`.gitignore`) and is not present in this repository snapshot, so it has been removed from this manifest. If you maintain a local `_archive/` for old experiments, document it separately rather than in this file, since its contents aren't part of the tracked codebase.
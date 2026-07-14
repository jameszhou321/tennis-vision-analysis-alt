# Tennis Match Analysis Project — File Manifest

> Graduation Project: Computer Vision-Based Automatic Tennis Match Analysis System
> Updated: 2026-04-24 (session18: config system reorganization, positional encoding toggle, inference.py fix)

---

## Directory Structure Overview

```
Directory Structure Overview
Project_Annotation_and_Testing/
├── src/                    Source code
│   ├── pipeline/           Core tracking pipeline
│   ├── model/              Action recognition model
│   │   └── mst/            MSTFormer model code (standalone directory)
│   ├── demo/                PyQt5 Demo application (video playback + inference visualization)
│   ├── utils/               Utility scripts (annotation, data processing)
│   ├── main.py              Batch processing main entry point
│   ├── train_court_pipeline.py  Court model training entry point
│   ├── court_detector.py    Court detector module
│   ├── pose_tracker.py      Pose tracker module
│   └── config_legacy.py     Batch processing configuration
├── models/                 Model weights
│   ├── yolo/                General-purpose YOLO series models
│   ├── court/                Court keypoint detection model
│   ├── action/                Action recognition model
│   └── person/                Person classification model
├── data/                    Datasets
│   ├── rallies_annotated/   Annotated rallies (with annotations.json)
│   ├── rallies_new/          Newly collected match rally data
│   ├── court_finetune/       Court fine-tuning dataset
│   └── person_sorter/         Person classification dataset
├── videos/                  Raw match videos
├── configs/                  YAML training config files
├── runs/                     Training run records
├── results/                  Analysis results & demo videos
├── logs/                      Pipeline run logs
└── _archive/                  Archive area (old code/old data)
    ├── legacy_src/            Legacy source code
    └── trainData/              Legacy training data
```

---

## 1. Source Code (`src/`)

### Main Entry Points

| File | Purpose |
| --- | --- |
| `src/main.py` | Batch video processing main entry point. Iterates over the `videos/` directory and runs the full tracking pipeline on each match video, with resume support (depends on `config_legacy.py`, `court_detector.py`, `pose_tracker.py`) |
| `src/train_court_pipeline.py` | Court keypoint model training entry point. Prepares the dataset YAML → launches YOLO fine-tuning → exports bad cases for iterative improvement |
| `src/config_legacy.py` | Configuration file for `main.py`. Defines video input/output paths, model paths, and processing parameters (frame skip count, confidence threshold, etc.) |
| `src/court_detector.py` | Court detector class (`CourtDetector`). Uses Hough line detection for a quick scan of the court and frames the far/near ROI (used during the patrol/scanning stage, not the keypoint model), called by `main.py` |
| `src/pose_tracker.py` | Pose tracker class (`PoseTracker`). Wraps YOLO pose estimation, with EMA smoothing and dropped-frame compensation, called by `main.py` |

### Core Tracking Pipeline (`src/pipeline/`)

| File | Purpose |
| --- | --- |
| `offline_tennis_tracker.py` | **Core module.** Offline tennis tracking: reads rally videos, computes a homography matrix using the court keypoint model, tracks players using the pose model, and outputs annotated video |
| `generate_trajectory.py` | Trajectory generation module. Extracts player coordinate sequences from tracking results and generates temporal features for action recognition |
| `debug_vision.py` | Visualization debugging tool. Overlays court detection and player tracking results onto video frames to verify pipeline output |
| `smart_extract_14pts.py` | Smart sampling annotation tool. Intelligently samples frames from match videos and pre-labels 14 keypoints using the court model to generate training data |
| `corner_driven_refine_tool.py` | Court annotation refinement tool (interactive GUI). Drag the 4 corner points and the other 10 keypoints are computed automatically; supports negative-sample annotation |
| `prepare_weighted_dataset.py` | Dataset merging tool. Merges newly annotated data with the old dataset, splits into train/val by ratio, for court model fine-tuning |
| `test_weighted_inference.py` | Inference test script. Uses an OpenCV tracker to test object tracking on video, verifying the inference pipeline |

### Action Recognition Model (`src/model/mst/`)

| File | Purpose |
| --- | --- |
| `model_main.py` | MSTFormer model definition. Dual-head output: 5-class action classification + binary keyframe classification. Supports `merge_visual_tokens` (three-stream merge resampler), `use_player_crops` (whether to use p1/p2 crop images), `use_pose` (whether pose tokens are zeroed out) toggles. The physics_extractor input is 125-dimensional |
| `modules/action_head.py` | Classification head module. `ActionClassificationHead` (5-class action) + `KeyframeDetectionHead` (binary keyframe), both with the same structure: LN → Linear → GELU → Dropout → Linear |
| `modules/pos_encoding.py` | Positional encoding module. Provides sinusoidal positional encoding for the Transformer |
| `modules/backbone_factory.py` | Visual backbone factory. Builds a YOLO11 / ViT feature extractor based on the `visual_backbone` config |
| `modules/token_resampler.py` | Token resampling module. Compresses an arbitrary number of tokens down to a fixed number (cross-attention resampler) |
| `tokenizer_pseudo.py` | Pseudo-tokenizer documentation. Describes how pose sequences are converted into tokens |
| `dataset.py` | Action recognition dataset loader. Reads `annotations.json`, builds sliding-window sequences, returns `(pose, packed_frames, action_labels, keyframe_labels)`. The pose vector is 125-dimensional (includes the 14 court keypoint coordinates and the person's position relative to the court center). Supports `image_augment` image augmentation: color jitter / Gaussian noise / blur / random erasing / semi-transparent overlay |
| `config.py` | Model training hyperparameter configuration (legacy version, superseded by YAML) |
| `train.py` | MSTFormer training script. Jointly trains action classification + keyframe detection, printing action Accuracy and keyframe Precision/Recall each epoch. Each training run saves `best.pth`, `final.pth`, `train_log.csv` (per-epoch metrics), and `config.yaml` (config snapshot) under `models/action/<config>/<timestamp>/` |
| `test_matrix.py` | Confusion matrix evaluation script |
| `test_dataset.py` | Dataset loading validation script |
| `tests/eval_optimal.py` | Model evaluation script, supports `--config`/`--weights` arguments, generates a confusion matrix image + CSV + classification report |

### Demo Application (`src/demo/`)

| File | Purpose |
| --- | --- |
| `main.py` | Demo entry point. Parses command-line arguments (`--rally`, `--config`, `--weights`, `--person`, `--pose`), fixes a Windows CUDA DLL load-order issue |
| `app.py` | Main window (PyQt5). Video playback, three-row timeline, file selection, YOLO model path input, inference trigger, action legend |
| `player.py` | Video player. QTimer + OpenCV frame-by-frame reading, handles Chinese-character paths (short-path conversion) |
| `timeline.py` | Timeline panel. Three rows: GT annotation bar / prediction bar / frame grid bar, with a cursor that follows the playback position |
| `inference.py` | Inference thread (QThread). Supports two modes: ① when person/pose YOLO paths are supplied, real-time detection with bbox + skeleton overlay drawn onto the frame; ② when not supplied, falls back to reading `pose_data.json` + pre-extracted crop images. Both modes feed the entire sequence into MSTFormer at once |
| `seq_len_sweep.py` | Sequence-length sweep script. Iterates over different seq_len values and outputs an accuracy CSV |

### Utility Scripts (`src/utils/`)

| File | Purpose |
| --- | --- |
| `action_annotator.py` | **Action temporal annotation tool** (Flask web app). Extracts clips from `data/rallies_new/` in rotation by video source into `data/rallies_annotating/`, lets you label 5 action-type time segments in the browser, and saves them as `annotations.json`. Supports deleting clips (auto-refills with new clips) and progress persistence (`_progress.json` records deletions so a restart doesn't re-extract) |
| `label_tool.py` | Player bounding box annotation tool (OpenCV GUI). Drag to draw a bounding box on an image, label near/far player, save in YOLO txt format |
| `data-batch-extractor.py` | Batch rally data extraction pipeline. Iterates over `data/rallies_new/`, runs court detection + pose tracking on each rally video, outputs `tracking_data.json`, with progress logging and resume support |
| `data-creater.py` | Person classification training data collection tool. Randomly samples frames from `data/rallies_new/` and stores them in `data/person_sorter/image/` for labeling |
| `dataset_splitter.py` | Dataset train/val split tool. Randomly splits images in `data/person_sorter/` into training and validation sets by ratio |
| `yolo-train.py` | YOLO person classification model training script. Fine-tunes a YOLO model based on the `data/person_sorter/` dataset |
| `inference_viewer.py` | Person classification model inference visualization tool. Runs person detection on `data/rallies_new/` and visualizes the results |
| `src/utils/prepare_train_dataset.py` | Copies files needed for training from `rallies_annotated/` to `rallies_train/` (raw_clip.mp4, pose_data.json, annotations.json), with resume support |
| `src/utils/merge_annotating_data.py` | Merges newly annotated data from `rallies_annotating/` into `rallies_annotated/`. Converts tracking_data.json → pose_data.json (including the court field), continuing numbering from rally_127 |
| `src/utils/add_court_to_pose.py` | Adds court keypoints to older data (rallies_annotated/). Runs the court model frame by frame, writes the 14 points into the `court` field of pose_data.json, with resume support |
| `src/utils/rerun_pose_detection.py` | Reruns pose detection on player1/player2 crop images (threshold 0.1), maps coordinates back to the original frame and filters using the person bbox, writes empty-detection frame stats to logs/pose_rerun_stats.json |

---

## 2. Model Weights (`models/`)

| Path | Purpose |
| --- | --- |
| `models/yolo/yolo11x-pose.pt` | YOLO11x pose estimation model (primary, highest accuracy) |
| `models/yolo/yolo26x-pose.pt` | YOLO26x pose estimation model (backup) |
| `models/yolo/yolov8n-pose.pt` | YOLOv8n pose model (base for court training) |
| `models/yolo/yolo26n.pt` | YOLO26n detection model |
| `models/yolo/yolo26x.pt` | YOLO26x detection model |
| `models/yolo/yoloe-26l-seg.pt` | YOLOe 26L segmentation model |
| `models/yolo/yoloe-26x-seg.pt` | YOLOe 26X segmentation model |
| `models/court/best.pt` | Best court 14-point keypoint detection weights (YOLO fine-tuned) |
| `models/person/best.pt` | Best person classification weights (near/far player) |
| `models/action/` | Action recognition weights (produced after training; currently empty — old weights archived to `_archive/models/action_backup_20260424/`) |

---

## 3. Datasets (`data/`)

| Path | Contents | Notes |
| --- | --- | --- |
| `data/rallies_annotated/` | 199 rallies (rally_001~104 + rally_127~221), all with `annotations.json` | Manually annotated rallies, used for training and evaluating the action recognition model. Each rally contains `raw_clip.mp4`, `pose_data.json` (including the court field), and `annotations.json` (5-class action time-segment labels) |
| `data/rallies_train/` | 192 rallies | Training data copied from rallies_annotated/ (raw_clip.mp4 + pose_data.json + annotations.json) |
| `data/rallies_annotating/` | Annotation workspace | Staging directory populated by `action_annotator.py` from `rallies_new`, organized into subfolders by video source. Contains `_progress.json` (records deleted clips, prevents re-extraction) |
| `data/court_finetune/` | Images + YOLO labels + bad_cases/ | Court 14-keypoint fine-tuning dataset, with train/val split |
| `data/person_sorter/` | Images + YOLO labels | Person classification (near/far player) training data |

### annotations.json Format

```json
[
  {"start_time": 0.0, "end_time": 4.837, "action_name": "Idle", "action_id": 0},
  {"start_time": 4.837, "end_time": 12.78, "action_name": "Serve", "action_id": 3}
]
```

Action categories: `Idle(0)` `Forehand(1)` `Backhand(2)` `Serve(3)` `Movement(4)`

### pose_data.json Format

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
- `_pose_rerun: true`: marks that this frame has already been processed by rerun_pose_detection.py

---

## 4. Other Directories

| Path | Contents |
| --- | --- |
| `videos/` | Raw match videos (25GB, MP4 + ASS subtitles), 10 matches |
| `configs/` | YAML training configs, see detailed section below |
| `runs/court_finetune/` | Court model training records (includes `weights/best.pt` for each version) |
| `runs/yolo/` | YOLO detection/pose training records |
| `results/` | Analysis result charts (confusion matrices), demo videos (`output_god_mode.mp4`, etc.) |
| `logs/` | Pipeline run logs (progress, errors, statistics) |

---

## 4.5. Training Configuration Files (`configs/`)

### Court / Person Classification (used for YOLO training)

| File | Purpose |
| --- | --- |
| `court_keypoints.yaml` | First-version court keypoint detection dataset config. Points to the old `Court_Finetune_Workspace/dataset`, 4-corner-point annotation, superseded by later versions |
| `court_keypoints_weighted.yaml` | Weighted court keypoint dataset config. Built on the first version with more merged annotation data, used for the second round of fine-tuning |
| `court_keypoints_ultimate.yaml` | Ultimate-version court keypoint dataset config. Merges all annotation rounds, used for the final production model training |
| `court_14pts_weighted.yaml` | Weighted 14-keypoint court dataset config. Upgraded to the 14-point annotation format (4 corner points + 10 auxiliary points), points to `dataset_v2`, currently the main court training config |
| `person_sorter_dataset.yaml` | Person classification dataset config. 2 classes: `player_near` (near-side player) / `player_far` (far-side player), data in `data/person_sorter/`, used by `yolo-train.py` |

### MSTFormer Action Recognition Configs (reorganized in session18)

All configs share a unified baseline: `embed_dim=128`, `depth=8`, `use_pos_encoding=false` (positional encoding disabled).

**Main config**

| File | Description |
| --- | --- |
| `main.yaml` | Current best baseline. Three-stream visual merge (`merge_visual_tokens=true`) + pose + Focal Loss, used directly for official training |

**hyperparams/ — Hyperparameter tuning**

| File | Variable | Description |
| --- | --- | --- |
| `hp_embed96.yaml` | `embed_dim=96` | Smaller embedding dimension |
| `hp_embed256.yaml` | `embed_dim=256` | Larger embedding dimension |
| `hp_depth4.yaml` | `depth=4` | Shallow Transformer |
| `hp_depth12.yaml` | `depth=12` | Deep Transformer |
| `hp_vtokens8.yaml` | `visual_tokens=8` | Stronger visual compression |
| `hp_vtokens32.yaml` | `visual_tokens=32` | More visual detail |

**ablation/ — Ablation experiments**

| File | Variable | Description |
| --- | --- | --- |
| `abl_no_pose.yaml` | `use_pose=false` | Remove pose input |
| `abl_no_crops.yaml` | `use_player_crops=false` | Remove player crop images |
| `abl_no_visual.yaml` | `use_visual=false` | Pose only, no visual stream |
| `abl_global_only.yaml` | `use_pose=false` + `use_player_crops=false` | Full-frame visual only |

**components/ — Component comparison**

| File | Variable | Description |
| --- | --- | --- |
| `cmp_focal_loss.yaml` | `loss=focal` | Focal Loss baseline |
| `cmp_ce_loss.yaml` | `loss=cross_entropy` | Cross-entropy loss comparison |
| `cmp_no_merge.yaml` | `merge_visual_tokens=false` | Independent three-stream tokens (sequence length 5880) |
| `cmp_resnet_backbone.yaml` | `visual_backbone=resnet18` | ResNet18 backbone (ImageNet pretrained) |
| `cmp_frozen_backbone.yaml` | `unfreeze_backbone=false` | Frozen backbone, only Transformer is trained |

Old configs (`mst_v2_*.yaml`, 11 files) have been archived to `_archive/configs_backup_20260424/`.

---

| Path | Contents | Notes |
| --- | --- | --- |
| `_archive/legacy_src/Hough.py` | Hough-transform court detection | Early experimental code using Hough line detection for court boundaries |
| `_archive/legacy_src/auto_extract_and_label.py` | Automatic extraction & labeling | Old script for automatically extracting frames and pre-labeling them |
| `_archive/legacy_src/convert_videos.py` | Video format conversion | Tool for batch converting video formats |
| `_archive/legacy_src/court_filter_test.py` | Court filtering test | Test script for court detection filtering logic |
| `_archive/legacy_src/demo_video_court.py` | Court detection demo | Early demo script for court detection results |
| `_archive/legacy_src/action_annotator_20260423.py` | Old annotation tool | Version prior to 2026-04-23, reads a flat rallies_annotated directory, no rotation extraction or deletion persistence |
| `_archive/legacy_src/text.py` | Pose extraction debugging | Temporary script for debugging pose keypoint extraction |
| `_archive/legacy_src/upgrade_4_to_14.py` | 4-point to 14-point conversion tool | Migration tool converting old 4-corner-point annotations to the new 14-keypoint format |
| `_archive/trainData_backup_20260424/` | rallies_train archive (104 rallies) | Archived 2026-04-24, training data before the feature expansion |
| `_archive/unannotated_rallies/` | Rallies without action annotations (22, rally_105~126) | Have pose_data.json but lack annotations.json, staged pending further labeling |
| `_archive/Second_Train_Dataset/` | Second batch of annotated data | Raw annotation data used for the second round of court model fine-tuning |

---

## 6. Module Dependencies

```
main.py
  ├── config_legacy.py      (path/parameter configuration)
  ├── court_detector.py     (court detection)
  └── pose_tracker.py       (pose tracking)

train_court_pipeline.py
  └── data/court_finetune/  (training data)
  └── configs/*.yaml        (training configuration)

src/model/mst/train.py
  ├── model_main.py         (model definition)
  ├── action_head.py        (classification head)
  ├── pos_encoding.py       (positional encoding)
  ├── dataset.py            (data loading)
  └── config.py             (hyperparameters)

src/pipeline/offline_tennis_tracker.py
  └── models/yolo/          (YOLO models)
  └── runs/court_finetune/  (court model)

src/utils/action_annotator.py
  └── data/rallies_new/         (source data, organized into subfolders by video source)
  └── data/rallies_annotating/  (workspace, contains _progress.json progress record)
```

---

## 7. Annotation Workflow

```
1. Collect videos
   Place raw match videos into videos/

2. Extract rallies
   main.py → data/rallies_new/{match_name}/rally_xxx/

3. Annotate court keypoints (for court model fine-tuning)
   smart_extract_14pts.py → pre-label frames
   corner_driven_refine_tool.py → manual refinement
   prepare_weighted_dataset.py → merge into data/court_finetune/
   train_court_pipeline.py → train new model

4. Annotate player actions (for action recognition model training)
   action_annotator.py → extract clips from rallies_new in rotation by source into rallies_annotating/
   label annotations.json in the browser (supports deleting clips, progress persistence)
   src/model/train.py → train MSTFormer

5. Annotate person classification (for near/far player identification)
   data-creater.py → sample images
   label_tool.py → label bounding boxes
   dataset_splitter.py → split train/val
   yolo-train.py → train classification model
```
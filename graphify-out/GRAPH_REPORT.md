# Graph Report - tennis-vision-analysis-alt  (2026-08-04)

## Corpus Check
- 99 files · ~97,209 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 993 nodes · 1555 edges · 68 communities (57 shown, 11 thin omitted)
- Extraction: 90% EXTRACTED · 9% INFERRED · 0% AMBIGUOUS · INFERRED: 147 edges (avg confidence: 0.74)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `b6a38b56`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- TokenResampler
- run_ablation.py — batch runs ablation/components/hyperparams configs
- CONFIG_REFERENCE.md — MSTFormer Config Fields Reference
- RallyStateMachine
- Config Reference & Figures
- docs/architecture.md — Project File Manifest & Module Dependencies
- YoloFrameClassifier
- src/utils/README.md — annotation/data/eval tool scripts overview
- MSTFormer
- inference.py
- MainWindow
- YOLO
- PoseTracker
- Dataset
- Action Annotator Flask Routes
- Inference Viewer & Review
- Waiting Segment Trimming
- ActionBarWidget
- main
- TrackNetBallTracker
- extract_forehand_frame.py
- VideoPlayer
- generate_model_report.py
- Hard Negative Reviewer Tool
- Data Quality Visualization
- Court Corner Refinement Tool
- Batch Data Extraction Pipeline
- config.py
- Player Bbox Labeling Tool
- Classical Ball Tracker
- Offline Tennis Tracker
- Thesis Figure Generation (Frames)
- hard_negative_extractor.py
- Pose Re-detection on Crops
- Person-on-Video Test Script
- Court Annotation Tool Suite
- TennisActionDataset
- Debug Vision Overlay
- test_dataset
- Thesis Figure Generation (Main)
- Action Class Taxonomy & Imbalance
- SpatialRallyDetector
- eval_optimal.py
- Crop & Pose Data Extraction
- Court Keypoint Addition Tool
- Person Test Visualization
- MST Confusion Matrix Classes
- Person Detector Training Entry
- Citation Unification Script
- Full-Frame Extraction
- Annotation Data Merging
- Hard Negative Mining Effect
- Player Crop Figure
- Chapter 3 Figure Generator
- Train Dataset Preparation
- Weighted Dataset Merging
- Hard Negatives Merging
- Class Distribution Analysis
- Person Data Collection Tool
- Dataset Train/Val Split
- YOLO Classifier Package Init
- Weighted Inference Test
- Improved Confusion Figure
- Project Root
- Main Model Training Curve (85.37% Test Acc)

## God Nodes (most connected - your core abstractions)
1. `docs/architecture.md — Project File Manifest & Module Dependencies` - 50 edges
2. `MSTFormer` - 43 edges
3. `TennisActionDataset` - 34 edges
4. `MainWindow` - 23 edges
5. `CONFIG_REFERENCE.md — MSTFormer Config Fields Reference` - 21 edges
6. `src/utils/README.md — annotation/data/eval tool scripts overview` - 21 edges
7. `TokenResampler` - 20 edges
8. `src/model/mst/README.md — MSTFormer directory overview` - 20 edges
9. `CLAUDE.md — Project Guidance for Claude Code` - 16 edges
10. `src/README.md — Source Code Overview` - 16 edges

## Surprising Connections (you probably didn't know these)
- `TennisActionDataset` --shares_data_with--> `125-dim Pose Feature Vector (17x3 abs kpts, 17x2 relative, center/velocity/accel, 6 ball reserved, 28 court)`  [EXTRACTED]
  src/model/mst/dataset.py → docs/architecture.md
- `merge_visual_tokens toggle — Perceiver-style resampling merges 3 visual streams into shared tokens vs independent per-stream tokens (rationale: trades sequence length/VRAM for potential cross-stream information sharing)` --references--> `MSTFormer`  [EXTRACTED]
  configs/components/cmp_no_merge.yaml → src/model/mst/model_main.py
- `use_visual toggle — disables all visual streams, isolating pure-pose performance (rationale: quantify overall visual contribution)` --references--> `MSTFormer`  [EXTRACTED]
  configs/ablation/abl_no_visual.yaml → src/model/mst/model_main.py
- `YoloFrameClassifier` --semantically_similar_to--> `modules/yolo_extractor.py — YOLO11 backbone (P3/P4/P5, cross-scale attention)`  [INFERRED] [semantically similar]
  src/model/yolo/model.py → src/model/mst/README.md
- `Main Model Training Curve (85.37% Test Acc)` --references--> `configs/main.yaml — Current Optimal MSTFormer Baseline`  [INFERRED]
  docs/figures/fig1_main_training_curve.png → configs/main.yaml

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **MSTFormer Ablation Study Config Group (pose/crops/visual on-off)** — configs_config_reference_abl_no_pose, configs_config_reference_abl_no_crops, configs_config_reference_abl_no_visual, configs_config_reference_abl_global_only [EXTRACTED 1.00]
- **MSTFormer Component Comparison Config Group (loss/merge/backbone variants)** — configs_config_reference_cmp_focal_loss, configs_config_reference_cmp_ce_loss, configs_config_reference_cmp_no_merge, configs_config_reference_cmp_resnet_backbone, configs_config_reference_cmp_frozen_backbone [EXTRACTED 1.00]
- **Swappable Visual Backbone Implementations (yolo11/resnet18/vit/raw via backbone_factory)** — src_model_mst_modules_backbone_factory_module, src_model_mst_modules_yolo_extractor_module, src_model_mst_modules_resnet_extractor_module, src_model_mst_modules_vit_extractor_module, src_model_mst_modules_raw_extractor_module [EXTRACTED 1.00]
- **MSTFormer training pipeline (train.py orchestrates dataset.py + config.py + model_main.py)** — src_model_mst_train_train, src_model_mst_dataset_tennisactiondataset, src_model_mst_config_config, src_model_mst_model_main_mstformer [INFERRED 0.85]
- **Court keypoint model data production workflow: smart sampling → corner refinement → dataset prep → training** — src_pipeline_smart_extract_14pts_smart_extract_14pts, src_pipeline_corner_driven_refine_tool_corner_driven_refine_tool, src_pipeline_prepare_weighted_dataset_prepare_weighted_dataset, src_train_court_pipeline_train_court_pipeline [EXTRACTED 0.95]
- **MSTFormer input-stream ablation study family (each isolates one input stream vs main.yaml baseline)** — configs_ablation_abl_global_only, configs_ablation_abl_no_crops, configs_ablation_abl_no_pose, configs_ablation_abl_no_visual, configs_main_main [EXTRACTED 0.90]

## Communities (68 total, 11 thin omitted)

### Community 0 - "TokenResampler"
Cohesion: 0.06
Nodes (31): Component Comparison Experiments (Ablation Bar Chart), model_main.py — MSTFormer Model Definition (Dual-Head: Action Classification +…, ActionClassificationHead, KeyframeDetectionHead, action_head.py — Action Classification Head and Keyframe Detection Head…, build_visual_extractor(), backbone_factory.py — Visual backbone factory (Builds YOLO11/ResNet/ViT/Raw per…, ViTPatchExtractor + TokenResampler: Preserves fine-grained patch features while… (+23 more)

### Community 1 - "run_ablation.py — batch runs ablation/components/hyperparams configs"
Cohesion: 0.17
Nodes (21): merge_visual_tokens toggle — Perceiver-style resampling merges 3 visual streams into shared tokens vs independent per-stream tokens (rationale: trades sequence length/VRAM for potential cross-stream information sharing), use_player_crops toggle — enables/disables per-player cropped visual streams (rationale: isolate crop-stream contribution vs full-frame-only), use_pose toggle — enables/disables 125-dim pose feature token (rationale: isolate pose-stream contribution to action accuracy), use_visual toggle — disables all visual streams, isolating pure-pose performance (rationale: quantify overall visual contribution), visual_backbone selection (yolo11 default vs resnet18/vit/raw) — rationale: compare detection-pretrained vs general ImageNet-pretrained vs lightweight extraction, abl_global_only.yaml — Ablation: Full-frame Visual Only (no crops, no pose), abl_no_crops.yaml — Ablation: No Player Crops (full-frame visual + pose), abl_no_pose.yaml — Ablation: No Pose (visual + crops only) (+13 more)

### Community 2 - "CONFIG_REFERENCE.md — MSTFormer Config Fields Reference"
Cohesion: 0.12
Nodes (26): ablation/abl_global_only.yaml (use_pose=false + use_player_crops=false), ablation/abl_no_crops.yaml (use_player_crops=false), ablation/abl_no_pose.yaml (use_pose=false), ablation/abl_no_visual.yaml (use_visual=false, pose only), class_weights per-class loss weighting field, components/cmp_ce_loss.yaml (loss=cross_entropy comparison), components/cmp_focal_loss.yaml (loss=focal baseline), components/cmp_frozen_backbone.yaml (unfreeze_backbone=false) (+18 more)

### Community 3 - "RallyStateMachine"
Cohesion: 0.15
Nodes (12): _bandpass_filter(), compute_impact_score_series(), extract_audio_wav(), get_score_at(), RallyStateMachine, audio_video_fusion.py — Audio Impact Detection + Hysteresis State Machine…, WAITING / POINT_ACTIVE hysteresis state machine (per the fusion design doc's…, Feeds one new sample; returns the status string for this sample ("PLAYING… (+4 more)

### Community 4 - "Config Reference & Figures"
Cohesion: 0.07
Nodes (25): configs/CONFIG_REFERENCE.md, Fig. 2 — MSTFormer Ablation Study Bar Chart, Figure 4: MSTFormer Hyperparameter Experiments (Depth / Embedding Dim / Visual Tokens), All Model Results — Ablation Summary Chart, Keyframe Detection F1 During Training (fig8), MSTFormer Ablation Study (stream/component removal), MSTFormer Hyperparameter Sensitivity Finding (depth=8, dim=256, vt=16 near-optimal), MSTFormer (multi-stream Transformer action recognition model) (+17 more)

### Community 5 - "docs/architecture.md — Project File Manifest & Module Dependencies"
Cohesion: 0.05
Nodes (73): CLAUDE.md — Project Guidance for Claude Code, configs/court_14pts_weighted.yaml — Current Court 14-Keypoint Dataset, configs/court_keypoints.yaml — First-Version Court Keypoint Config, configs/court_keypoints_ultimate.yaml — Merged Court Keypoint Config, configs/court_keypoints_weighted.yaml — Weighted Court Keypoint Config, configs/person_sorter_dataset.yaml — Person Classifier Dataset, annotations.json format (action time-segment labels), BoT-SORT (player tracking algorithm used in offline pipeline) (+65 more)

### Community 6 - "YoloFrameClassifier"
Cohesion: 0.25
Nodes (4): Action Class Taxonomy (idle/forehand/backhand/serve/movement), YOLO Single-Frame Action Classification — Model Definition Uses YOLO11n…, YOLO11 backbone (feature extractor inclusive) + Classification head for single-…, YoloFrameClassifier

### Community 7 - "src/utils/README.md — annotation/data/eval tool scripts overview"
Cohesion: 0.09
Nodes (31): Class-imbalance handling in MSTFormer training — focal loss (gamma=2.0) + per-class weights [1.0,4.0,5.0,4.0,1.5] to offset forehand/backhand/serve underrepresentation vs idle/movement; compared against plain cross-entropy, cmp_ce_loss.yaml — Component: Cross-Entropy Loss, cmp_focal_loss.yaml — Component: Focal Loss (baseline loss), configs/person_sorter_dataset.yaml — person_sorter dataset configuration, tests/eval_optimal.py — evaluation + confusion matrix, merge_hard_negatives.py — merges mined hard negatives into training set, src/training/README.md — person detector training overview, train_person_detector.py — fine-tunes YOLO for player_near/player_far classification (+23 more)

### Community 8 - "MSTFormer"
Cohesion: 0.25
Nodes (7): parallel_backbones config toggle (run 3 backbones concurrently, VRAM-heavy), Dual-Head Output (5-class action classification + binary keyframe detection), MSTFormer, packed: uint8 [N, 3, 320, 960] → 3-way float32 each [N, 3, 320, 320],…, _unpack_and_normalize(), modules/action_head.py — action classification + keyframe detection heads, modules/pos_encoding.py — sinusoidal positional encoding (disabled by default)

### Community 9 - "inference.py"
Cohesion: 0.12
Nodes (19): QThread, _build_gt_labels(), _crop_fixed(), _draw_person(), _emit_result(), InferenceThread, _nullctx, inference.py — Inference thread: real-time person YOLO + pose YOLO detection… (+11 more)

### Community 10 - "MainWindow"
Cohesion: 0.10
Nodes (8): QMainWindow, _btn(), _label(), _load_prefs(), MainWindow, _save_prefs(), main(), main.py — Demo Entry Point

### Community 11 - "YOLO"
Cohesion: 0.06
Nodes (39): annotate_rally_clip(), BatchTennisPipeline, build_timeline_blocks(), concat_videos(), detect_scene_cut_frames(), format_timestamp(), get_acceleration_device(), get_ball_tracker() (+31 more)

### Community 12 - "PoseTracker"
Cohesion: 0.06
Nodes (24): config_legacy.py — Batch Processing Pipeline Configuration (for use by main.py)…, get_weighted_homography(), HomographyFilter, is_within_court_region(), project_far_half_pixel_bbox(), court_detector.py — Court Detector (for use by main.py) Function: Detects the…, model: a pre-loaded YOLO court-keypoint model (see config.COURT_MODEL_PATH).…, Detects the 14 court keypoints in `frame` and returns a temporally-smoothed… (+16 more)

### Community 14 - "Action Annotator Flask Routes"
Cohesion: 0.21
Nodes (18): route, delete_clip(), extract_next_rally(), extract_one(), get_deleted_set(), get_json(), get_source_folders(), get_source_rallies() (+10 more)

### Community 15 - "Inference Viewer & Review"
Cohesion: 0.14
Nodes (10): main(), inference_viewer.py — Person classification model inference visualization tool, extract_feet_pos(), get_homography(), side_by_side_viewer.py — Dual-view Comparison Player Functionality: Displays…, Core Interaction: Handles all mouse clicks, dragging, and wheel scrolling, Maps the mouse X coordinate to video frame index, Draws the left playlist panel (+2 more)

### Community 16 - "Waiting Segment Trimming"
Cohesion: 0.17
Nodes (18): copy_with_hardlinks(), count_trailing_wait(), has_leading_wait(), has_two_serves(), load_annotations(), main(), process_rally(), Trim waiting segments in the dataset to mitigate class imbalance. Reads data… (+10 more)

### Community 17 - "ActionBarWidget"
Cohesion: 0.10
Nodes (13): QWidget, _action_color(), ActionBarWidget, FrameTrackWidget, timeline.py — Timeline components: GT tracks / Prediction tracks / Frame grid…, Complete timeline panel: GT track + Prediction track + Frame grid track (with…, Parses GT intervals from annotations.json and triggers rendering paths., per_frame_preds: list[int], length == total_frames, predicted category index… (+5 more)

### Community 18 - "main"
Cohesion: 0.17
Nodes (11): get_weighted_homography(), HomographyFilter, main(), RadarDrawer, generate_trajectory.py — Player Movement Trajectory Generation Module Function:…, Sliding average of trajectory points, Calculate the residual function of weighted projection errors, Use the weighted L-M algorithm to compute the precise homography matrix (+3 more)

### Community 19 - "TrackNetBallTracker"
Cohesion: 0.10
Nodes (11): ball_tracker_tracknet.py — TrackNet-backed Ball Tracker (wraps…, Shared bookkeeping for anything treated as a miss, whether the model returned…, Processes one frame. Returns {"position": (x,y) or None, "speed": float,…, Draws a fading trail of recent ball positions onto annotated_frame in-place., Clears all tracking state. Call this after a hard camera cut, so TrackNet's…, Resizes and stacks the last 3 frames (current, prev, prev-prev) into model…, TrackNetBallTracker, postprocess() (+3 more)

### Community 20 - "extract_forehand_frame.py"
Cohesion: 0.20
Nodes (14): draw_court(), draw_player(), extract_crop(), get_short_path(), imwrite_cn(), main(), Extract a good forehand frame from rally videos for thesis figure., imwrite_cn with Chinese path support. (+6 more)

### Community 22 - "VideoPlayer"
Cohesion: 0.16
Nodes (4): QObject, _get_short_path(), player.py — Video player based on QTimer + OpenCV frame-by-frame loading., VideoPlayer

### Community 23 - "generate_model_report.py"
Cohesion: 0.22
Nodes (9): collect_all_data(), generate_report(), parse_csv(), plot_curves(), Reads all training outputs under models/action/ and generates a comprehensive…, Collect data for all models, Generate the comprehensive Markdown report, Read train_log.csv and return a list of dicts (+1 more)

### Community 24 - "Hard Negative Reviewer Tool"
Cohesion: 0.25
Nodes (13): _imread(), load_low_quality(), load_manifest(), load_reviewed(), main(), Path, hard_negative_reviewer.py — Interactive frame review + annotation tool…, Redraw the current frame: existing boxes + the box currently being drawn + UI… (+5 more)

### Community 25 - "Data Quality Visualization"
Cohesion: 0.25
Nodes (13): build_index(), draw_bbox(), draw_court(), draw_keypoints(), get_frame(), get_short_path(), load_entry(), main() (+5 more)

### Community 26 - "Court Corner Refinement Tool"
Cohesion: 0.23
Nodes (4): corner_driven_refine_tool.py — Court Corner-Driven Annotation Refinement Tool…, Core Feature 1: Initializes a standard court template in the center of the…, Core Feature 2: Supports saving empty negative samples, UltimateRefiner

### Community 27 - "Batch Data Extraction Pipeline"
Cohesion: 0.23
Nodes (7): main(), PipelineManager, data_batch_extractor.py — Batch Rally Data Extraction Pipeline Function:…, Loads breakpoint resumption progress and historical statistics., Records a video clip path as successfully processed., Logs a corrupted or unreadable video, marking it processed to prevent endless…, Updates global confidence moving targets and checks if the current video…

### Community 28 - "config.py"
Cohesion: 0.09
Nodes (25): evaluate(), main(), _nullctx, seq_len_sweep.py — Sequence Length Sweep: Fixed weights, evaluate accuracy…, load_config(), config.py — YAML Configuration Parser, evaluate_and_plot(), test_matrix.py — Confusion Matrix Evaluation Script (+17 more)

### Community 29 - "Player Bbox Labeling Tool"
Cohesion: 0.24
Nodes (12): convert_to_yolo_format(), draw_boxes(), ensure_dir(), load_annotations(), main(), mouse_callback(), label_tool.py — Player Bounding Box Manual Annotation Tool (YOLO format)…, Convert (x1, y1, x2, y2) to normalized YOLO format (x_center, y_center, w, h) (+4 more)

### Community 30 - "Classical Ball Tracker"
Cohesion: 0.18
Nodes (6): BallTracker, ball_tracker.py — Lightweight Ball Tracker (classical CV, no pretrained weights…, Draws a fading trail of recent ball positions onto annotated_frame in-place., Returns a list of (x, y, radius) candidate ball blobs from foreground motion., Clears all tracking state. Call this after a hard camera cut, so a stale pre-…, Processes one frame. Returns {"position": (x,y) or None, "speed": float,…

### Community 32 - "Offline Tennis Tracker"
Cohesion: 0.27
Nodes (7): get_weighted_homography(), HomographyFilter, main(), RadarDrawer, offline_tennis_tracker.py — Offline Tennis Tracking Main Module Function: Read…, reprojection_residuals(), score_and_select_players()

### Community 33 - "Thesis Figure Generation (Frames)"
Cohesion: 0.26
Nodes (11): draw_court(), draw_dashed_line(), draw_player(), extract_crop(), get_short_path(), imwrite_cn(), main(), Generate thesis Figure: annotated frame + square player crops with dashed… (+3 more)

### Community 34 - "hard_negative_extractor.py"
Cohesion: 0.23
Nodes (11): extract_worst_frames(), imwrite_unicode(), main(), parse_subpar_log(), hard_negative_extractor.py — Extracts low-confidence frames from…, Convert a Windows path containing Chinese characters to an 8.3 short path for…, Image writing with support for paths containing Chinese characters, Parse pipeline_subpar.txt and return the list of clips matching the given… (+3 more)

### Community 35 - "Pose Re-detection on Crops"
Cohesion: 0.26
Nodes (11): _calc_win(), _crop_to_orig(), get_short_path(), _in_bbox(), main(), process_clip(), Rerun pose detection on player1/player2 cropped images with a low threshold,…, Calculate crop window size from bbox [x1, y1, x2, y2] and base_win (consistent… (+3 more)

### Community 36 - "Person-on-Video Test Script"
Cohesion: 0.29
Nodes (10): ndarray, collect_clips(), draw_detections(), get_short_path(), main(), process_rally(), Path, test_person_on_video.py — Per-frame inference on rallies_new clips, outputting… (+2 more)

### Community 38 - "Court Annotation Tool Suite"
Cohesion: 0.31
Nodes (10): src/main.py — batch rally segmentation pipeline (referenced), corner_driven_refine_tool.py — interactive GUI to refine 4 corner points, debug_vision.py — visualization debugging overlay, generate_trajectory.py — extracts player coordinate trajectories, offline_tennis_tracker.py — two-pass homography + BoT-SORT tracking + radar-view render, prepare_weighted_dataset.py — merges/splits court annotation dataset, src/pipeline/README.md — offline precision tracking and court annotation tools overview, smart_extract_14pts.py — smart sampling + model pre-labels 14 court points (+2 more)

### Community 39 - "TennisActionDataset"
Cohesion: 0.13
Nodes (19): cmp_frozen_backbone.yaml — Component: Frozen Backbone (unfreeze_backbone=false), augment.py — asynchronous image augmentation buffer, config.py — YAML config parser, Dataset, Fixed slicing, used for test set or initialization., Called before the start of each epoch to randomly re-slice training samples., TennisActionDataset, extract_crops.py — pre-extracts player1/player2 crops (+11 more)

### Community 40 - "Debug Vision Overlay"
Cohesion: 0.29
Nodes (7): get_weighted_homography(), interpolate_track(), main(), RadarRenderer, debug_vision.py — Visual Debugging Tool Function: Overlays tracking and…, Computes weighted homography, optimizing projection residuals via the…, Fills tracking gaps via linear interpolation and applies EMA smoothing while…

### Community 41 - "test_dataset"
Cohesion: 0.27
Nodes (9): get_short_path(), iou(), load_labels(), test_person_detector.py — Full-scale Evaluation for Player Detection Model…, Converts long Windows Chinese paths to short DOS paths to prevent parsing…, Reads Ground Truth (GT) from .txt label files in YOLO format., Calculates Intersection over Union (IoU) for two YOLO-formatted bounding boxes., Evaluates the dataset across specified data splits. (+1 more)

### Community 42 - "Thesis Figure Generation (Main)"
Cohesion: 0.27
Nodes (5): add_labels(), fig2(), fig3(), fig4(), generate_thesis_figures.py — Thesis Figure Generation Tool (using real data…

### Community 43 - "Action Class Taxonomy & Imbalance"
Cohesion: 0.28
Nodes (9): Backhand Action Class (反手), Forehand Action Class (正手), Idle Action Class (待机), Movement Action Class (移动), Serve Action Class (发球), annotations.json Action Label Format, Idle-Segment Dataset Trimming Process, Action Class Distribution Chart (Original vs After Trimming) (+1 more)

### Community 45 - "SpatialRallyDetector"
Cohesion: 0.28
Nodes (5): broadcast_detector.py — Spatial Rally Detection Engine Function: Detects tennis…, Detects rallies based on the physics of the ball and player positions. :param…, Core logic engine. Call this every frame. :param ball_xy: (x, y) tuple of the…, SpatialRallyDetector, main()

### Community 46 - "eval_optimal.py"
Cohesion: 0.08
Nodes (24): 5-Class Action Taxonomy (idle/forehand/backhand/serve/movement), MSTFormer Action Classification Head, Dataset, Fig. 7 — Column-Normalized Confusion Matrix (Precision View), compute_cm(), main(), plot_confusion_matrix(), print_classification_report() (+16 more)

### Community 48 - "Crop & Pose Data Extraction"
Cohesion: 0.39
Nodes (7): crop_fixed_window(), extract_clip(), get_short_path(), main(), extract_crops.py — Pre-extract athlete cropped images + generate pose_data.json…, Crops a win×win window centered at (cx, cy). Pads out-of-bounds regions with…, save_jpg()

### Community 49 - "Court Keypoint Addition Tool"
Cohesion: 0.36
Nodes (7): _already_done(), get_short_path(), main(), process_clip(), add_court_keypoints.py — Complementing Court Keypoints for Annotated Rallies…, Retrieves the Windows short path name to prevent path parsing issues with…, Checks whether the 'court' field already exists by validating the first non-…

### Community 50 - "Person Test Visualization"
Cohesion: 0.46
Nodes (7): draw_box(), get_short_path(), load_labels(), main(), visualize_person_test.py — Visualizes person detection results. Usage: python…, read_img(), visualize_image()

### Community 51 - "MST Confusion Matrix Classes"
Cohesion: 0.38
Nodes (7): Action class: backhand, Action class: forehand, Action class: idle, Action class: movement, Action class: serve, MSTFormer 5-Class Action Classification, Confusion Matrix — MSTFormer Main Model

### Community 55 - "Person Detector Training Entry"
Cohesion: 0.38
Nodes (6): check_data(), prepare_dataset_yaml(), train_person_detector.py — Player Detection Model Training Entry Function:…, Generate absolute path version of dataset.yaml to prevent YOLO path parsing…, Verify dataset is ready before training, train()

### Community 56 - "Citation Unification Script"
Cohesion: 0.38
Nodes (6): main(), normalize_ref(), parse_references(), Global Unified Citation Numbering Script. 1. Collect references across all…, Extract reference list elements from text content., Normalize reference text strings to perform precise deduplication matching.

### Community 57 - "Full-Frame Extraction"
Cohesion: 0.53
Nodes (5): extract_clip(), get_short_path(), main(), extract_frames.py — Pre-extract full-frame JPEGs for direct reading in…, save_jpg()

### Community 58 - "Annotation Data Merging"
Cohesion: 0.47
Nodes (5): _convert_tracking_to_pose(), _get_max_rally_num(), main(), Merge new annotation data from data/rallies_annotating/ into…, Convert tracking_data.json to pose_data.json format (list, indexed by frame_id).

### Community 59 - "Hard Negative Mining Effect"
Cohesion: 0.50
Nodes (5): Hard Negative Mining Effect (F1-Score bar chart), Hard Negative Mining (HNM) technique, train_person_detector.py (near/far player classifier training), generate_ch3_figures.py (thesis chapter 3 figure generator), hard_negative_extractor.py (mines hard negatives from false detections)

### Community 62 - "Player Crop Figure"
Cohesion: 0.83
Nodes (4): Far Player Crop, MSTFormer Player-Crop Visual Stream, Near Player Crop, Player Crops Figure (Near/Far Player, 320x320)

### Community 64 - "Train Dataset Preparation"
Cohesion: 0.67
Nodes (3): is_complete(), main(), Copy annotation data from data/rallies_annotated/ to data/rallies_train/. Only…

### Community 79 - "Main Model Training Curve (85.37% Test Acc)"
Cohesion: 0.67
Nodes (4): MSTFormer Action Recognition Model, MSTFormer Main Model: 85.37% Test Accuracy, Train/Test Accuracy Gap (Overfitting Signal, ~44 epochs), Main Model Training Curve (85.37% Test Acc)

## Ambiguous Edges - Review These
- `docs/training_list.txt — Reference Video URL List` → `data/rallies_new/ — Segmented Rally Clips (main.py output)`  [AMBIGUOUS]
  docs/training_list.txt · relation: conceptually_related_to
- `configs/CONFIG_REFERENCE.md` → `MSTFormer Hyperparameter Sensitivity Finding (depth=8, dim=256, vt=16 near-optimal)`  [AMBIGUOUS]
  docs/figures/fig4_hyperparameter_comparison.png · relation: conceptually_related_to

## Knowledge Gaps
- **42 isolated node(s):** `tennis-vision-analysis`, `configs/optimal_full.yaml — Full-Dataset Optimal Training`, `configs/court_keypoints.yaml — First-Version Court Keypoint Config`, `docs/training_list.txt — Reference Video URL List`, `Three-Stage CV Pipeline (Court Detection → Pose Tracking → Action Recognition)` (+37 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **11 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `docs/training_list.txt — Reference Video URL List` and `data/rallies_new/ — Segmented Rally Clips (main.py output)`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `configs/CONFIG_REFERENCE.md` and `MSTFormer Hyperparameter Sensitivity Finding (depth=8, dim=256, vt=16 near-optimal)`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **Why does `docs/architecture.md — Project File Manifest & Module Dependencies` connect `docs/architecture.md — Project File Manifest & Module Dependencies` to `TokenResampler`, `CONFIG_REFERENCE.md — MSTFormer Config Fields Reference`, `YoloFrameClassifier`, `TennisActionDataset`, `MSTFormer`, `PoseTracker`?**
  _High betweenness centrality (0.172) - this node is a cross-community bridge._
- **Why does `MSTFormer` connect `MSTFormer` to `TokenResampler`, `run_ablation.py — batch runs ablation/components/hyperparams configs`, `CONFIG_REFERENCE.md — MSTFormer Config Fields Reference`, `Config Reference & Figures`, `docs/architecture.md — Project File Manifest & Module Dependencies`, `YoloFrameClassifier`, `TennisActionDataset`, `src/utils/README.md — annotation/data/eval tool scripts overview`, `inference.py`, `config.py`?**
  _High betweenness centrality (0.121) - this node is a cross-community bridge._
- **Why does `TokenResampler` connect `TokenResampler` to `MSTFormer`, `docs/architecture.md — Project File Manifest & Module Dependencies`?**
  _High betweenness centrality (0.091) - this node is a cross-community bridge._
- **Are the 17 inferred relationships involving `MSTFormer` (e.g. with `_emit_result()` and `InferenceThread`) actually correct?**
  _`MSTFormer` has 17 INFERRED edges - model-reasoned connections that need verification._
- **Are the 15 inferred relationships involving `TennisActionDataset` (e.g. with `main()` and `_nullctx`) actually correct?**
  _`TennisActionDataset` has 15 INFERRED edges - model-reasoned connections that need verification._
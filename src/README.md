# src/ — Source Code Overview

This directory contains all the source code for the tennis match vision analysis system, organized functionally into a "top-level entry point + 5 subpackages".

> All scripts default to being run from the **repository root** (paths are relative to the root), e.g. `python src/main.py`.

## Top-Level Files (Batch Processing Pipeline)

| File | Purpose | Usage |
| --- | --- | --- |
| `main.py` | Batch video processing main entry point: iterates over `videos/`, a CPU thread scans the court and cuts rallies while a GPU thread runs pose tracking, outputting each rally's `raw_clip.mp4` / `annotated_clip.mp4` / `pose_data.json`, with resume support | `python src/main.py` |
| `config_legacy.py` | Configuration for `main.py` (video/output/model paths, confidence, EMA parameters, etc.) — change paths and thresholds here | Imported by `main.py`, `pose_tracker.py` |
| `court_detector.py` | Court ROI detector used during the scanning stage: uses Hough line detection to quickly determine whether a court is present in the frame and frames the far/near-side regions (lightweight, not the keypoint model) | Called by `main.py` |
| `pose_tracker.py` | Pose tracker: runs YOLO-pose within the ROI, uses multi-dimensional scoring to identify the actual players, includes EMA smoothing and dropped-frame compensation | Called by `main.py` |
| `train_court_pipeline.py` | Training entry point for the court **14-keypoint** detection model (YOLO-pose fine-tuning), also exports bad cases for iteration | `python src/train_court_pipeline.py` |
| `test_person_detector.py` | Quick test script for the person detection/classification model | `python src/test_person_detector.py` |

## Subpackages

| Directory | Contents | See |
| --- | --- | --- |
| `pipeline/` | Offline precision tracking (court homography matrix + trajectories), court annotation sampling/refinement tools, dataset merging | [`pipeline/README.md`](./pipeline/README.md) |
| `model/mst/` | **MSTFormer** action recognition model (core): model definition, dataset, training, ablations, evaluation | [`model/mst/README.md`](./model/mst/README.md) |
| `model/yolo/` | Single-frame YOLO action classification model (comparison baseline) | [`model/yolo/README.md`](./model/yolo/README.md) |
| `demo/` | PyQt5 desktop demo: video playback + timeline + real-time inference visualization | [`demo/README.md`](./demo/README.md) |
| `utils/` | Annotation tools, data processing, paper figures, and evaluation scripts | [`utils/README.md`](./utils/README.md) |
| `training/` | Person detection model training and hard-example mining | [`training/README.md`](./training/README.md) |

## Two Main Pipelines at a Glance

```
A. Data production line (from video to training samples)
   videos/ ──main.py──▶ data/rallies_new/ (rally clips + pose_data.json)
                          │
                          ├─ utils/action_annotator.py ─▶ label actions annotations.json
                          └─ model/mst/extract_crops.py ─▶ player crop images player1/ player2/

B. Model line
   Court keypoints:        train_court_pipeline.py ─▶ models/court/best.pt
   Person classification:  training/train_person_detector.py ─▶ models/person/best.pt
   Action recognition:     model/mst/train.py ─▶ models/action/<config>/<timestamp>/best.pth
```

## Conventions

- Code comments are always written in Chinese; every module starts with a `"""docstring"""` describing the file's purpose.
- Handling paths containing Chinese characters: on Windows, short paths are used to avoid OpenCV encoding issues; this is automatically skipped on non-Windows systems (see `_get_short_path` in each file).
- Large files (`videos/`, `data/`, `models/`, `runs/`) are not included in the repository and must be prepared separately; path conventions are defined in `config_legacy.py` and `configs/`.
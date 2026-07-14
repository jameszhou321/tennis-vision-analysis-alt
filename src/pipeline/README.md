# pipeline/ — Offline Precision Tracking and Court Annotation Tools

This directory contains two categories of content: ① single-rally **offline precision tracking** (finer-grained than the batch scanning in `main.py`, including the court homography matrix and a radar-view plot); ② **sampling and refinement tools** for annotating the court's 14 keypoints.

## Offline Tracking

| File | Purpose | Run |
| --- | --- | --- |
| `offline_tennis_tracker.py` | **Core module.** Two-pass processing: Pass 1 uses the court keypoint model to compute a weighted homography matrix frame by frame, tracks players with botsort, and projects them into court coordinates; Pass 2 renders the court lines, player boxes, and an overhead radar-view plot, outputting an annotated video | `python src/pipeline/offline_tennis_tracker.py` |
| `generate_trajectory.py` | Extracts player coordinate sequences from the tracking results, generating temporal trajectories for action recognition | `python src/pipeline/generate_trajectory.py` |
| `debug_vision.py` | Visualization debugging: overlays court detection/player tracking onto video frames to verify the pipeline (change the input video/model paths via the constants at the top of the file) | `python src/pipeline/debug_vision.py` |
| `test_weighted_inference.py` | Tests object tracking performance using an OpenCV tracker | — |

## Court Keypoint Annotation Tools

The full data production workflow for the court model:

```
Match video ─▶ smart_extract_14pts.py (smart sampling + model pre-labels 14 points)
              │
              ▼
        corner_driven_refine_tool.py (drag the 4 corner points, the other 10 are auto-computed, manual refinement)
              │
              ▼
        prepare_weighted_dataset.py (merges new and old annotations, splits train/val by ratio)
              │
              ▼
        ../train_court_pipeline.py (trains the new court model)
```

| File | Purpose |
| --- | --- |
| `smart_extract_14pts.py` | Intelligently samples frames from video and pre-labels 14 keypoints using the existing court model, generating training candidates |
| `corner_driven_refine_tool.py` | Interactive GUI: drag the 4 corner points and the other 10 points are automatically computed; supports negative-sample annotation |
| `prepare_weighted_dataset.py` | Merges new annotations with the old dataset and splits into train/val, for court model fine-tuning |

> The physical coordinates (in meters) for the court's 14 points are defined in `COURT_14_PTS_PHYSICAL` at the top of `offline_tennis_tracker.py` and `../train_court_pipeline.py`.
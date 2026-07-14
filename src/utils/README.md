# utils/ — Annotation, Data Processing, and Evaluation Tools

This directory contains various one-stop tool scripts, organized into five categories by purpose. Most are run from the repository root.

## 1. Annotation Tools

| File | Purpose | Run |
| --- | --- | --- |
| `action_annotator.py` | **Action temporal annotation** (Flask web app). Extracts clips from `data/rallies_new/` in rotation, lets you label 5 action-type time segments in the browser, saves to `annotations.json`; supports deleting clips and progress persistence | `python src/utils/action_annotator.py`, then visit http://localhost:5000 |
| `label_tool.py` | Player **bbox annotation** (OpenCV GUI): drag to draw boxes labeling near/far-side players, saves in YOLO txt format | `python src/utils/label_tool.py` |

## 2. Data Production and Processing

| File | Purpose |
| --- | --- |
| `data-batch-extractor.py` | Batch rally data extraction pipeline: iterates over `rallies_new/`, runs court + pose detection, outputs `tracking_data.json`, with resume support |
| `data-creater.py` | Person classification sampling: randomly samples frames from rallies into `data/person_sorter/image/` |
| `dataset_splitter.py` | Splits images in `person_sorter/` into train/val by ratio |
| `prepare_train_dataset.py` | Copies files needed for training from `rallies_annotated/` to `rallies_train/`, with resume support |
| `merge_annotating_data.py` | Merges new annotations from `rallies_annotating/` into `rallies_annotated/` (converts tracking→pose format) |
| `add_court_to_pose.py` | Adds the 14 court points frame by frame to older data, writing into the `court` field of `pose_data.json` |
| `rerun_pose_detection.py` | Reruns pose detection on player crop images with a low threshold, maps coordinates back to the original frame and filters |
| `trim_waiting_segments.py` | Trims overly long "Idle" segments to alleviate class imbalance |

## 3. Inference Visualization and Testing

| File | Purpose |
| --- | --- |
| `inference_viewer.py` | Person classification model inference visualization |
| `test_person_on_video.py` / `visualize_person_test.py` | Tests person detection on video and visualizes results |
| `visualize_data_quality.py` | Data quality visualization (whether annotations/poses look normal) |
| `side_by_side_viewer.py` | Side-by-side comparison view of multiple results |

## 4. Evaluation and Reporting

| File | Purpose |
| --- | --- |
| `batch_eval_all.py` | Batch-evaluates all trained models |
| `generate_model_report.py` | Aggregates metrics across models into a report |
| `analyze_class_distribution.py` | Computes the distribution of action classes |

## 5. Hard-Example Mining

| File | Purpose |
| --- | --- |
| `hard_negative_extractor.py` | Mines hard negative samples from false detections |
| `hard_negative_reviewer.py` | Manual review of hard examples |

## 6. Thesis Figures (used only for writing the thesis, unrelated to system operation)

`generate_thesis_figures.py`, `generate_ch3_figures.py`, `generate_confusion_figures.py`, `generate_confusion_matrices.py`, `create_thesis_figure_N.py`, `extract_forehand_frame.py`, `unify_citations.py` — generate training curves, confusion matrices, and other figures for the thesis, output to `docs/figures/`.

> Note: These scripts require the local dataset to reproduce; the open-source repository does not include the dataset. See `docs/figures/` for the resulting figures.
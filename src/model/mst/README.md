# model/mst/ — MSTFormer Action Recognition Model (Core of the Project)

MSTFormer (Multi-Stream Transformer) is the core model of this project. It fuses **player pose sequences**, **court geometric position**, and **multi-stream visual crop images** into a single Transformer, with a **dual-head output**:

- Action classification: Idle / Forehand / Backhand / Serve / Movement (5 classes)
- Keyframe detection: whether each frame is a keyframe marking an action transition (binary classification)

## Input Composition

| Input | Dimensions | Source |
| --- | --- | --- |
| Pose physical features `pose` | `[B, T, 125]` | From `pose_data.json`, built by `_build_pose_vec` in `dataset.py` |
| Packed visual frames `packed_frames` | `[B, T, 3, 320, 960]` uint8 | Full frame + player1 + player2, concatenated side by side into three streams |

> 125 dimensions = 17×3 (absolute keypoint coordinates + confidence) + 17×2 (relative to person center) + 2 (person center relative to court) + 2 (velocity) + 2 (acceleration) + 6 (ball, reserved) + 28 (14 court points × 2). Visual frame normalization is done on the GPU side; the CPU side keeps them as uint8 to save bandwidth.

## File Structure

| File | Purpose |
| --- | --- |
| `model_main.py` | **Model definition**, `MSTFormer`. Three visual streams → optional merge (`merge_visual_tokens`) → concatenated with pose tokens → Transformer → dual-head output. Toggles: `use_pose` / `use_player_crops` / `use_visual` / `merge_visual_tokens` |
| `dataset.py` | Dataset class `TennisActionDataset`. Reads `pose_data.json` + `annotations.json`, slices via sliding window, builds the 125-dimensional pose vector and the three visual frame streams; includes image augmentation |
| `train.py` | **Training entry point.** Jointly trains action classification + keyframe detection, with AMP + gradient accumulation, splits train/val by video, outputs to `models/action/<config>/<timestamp>/` |
| `config.py` | YAML config parser; converts relative paths to absolute, fills in device and gradient accumulation steps |
| `augment.py` | Asynchronous image augmentation buffer; moves augmentation off the DataLoader worker and into a separate thread pool |
| `extract_frames.py` | Pre-extracts full frames from rally videos into `frames/` (speeds up image reading during training) |
| `extract_crops.py` | Pre-extracts player1/player2 crop images into `player1/`, `player2/` |
| `run_ablation.py` | Batch-runs the experiments under `configs/ablation`, `components`, `hyperparams` |
| `modules/` | Model submodules (see below) |
| `tests/` | `eval_optimal.py` (evaluation + confusion matrix), `test_matrix.py`, `test_dataset.py` |

### modules/ Submodules

| File | Purpose |
| --- | --- |
| `backbone_factory.py` | Visual backbone factory; builds one of the four backbones below based on the `visual_backbone` config |
| `yolo_extractor.py` | YOLO11 backbone, taps P3/P4/P5, cross-scale attention fusion (primary) |
| `resnet_extractor.py` | ResNet18 backbone (comparison) |
| `vit_extractor.py` | Lightweight ViT patch embedding (comparison) |
| `raw_extractor.py` | Raw pixel projection (comparison) |
| `token_resampler.py` | Perceiver-style cross-attention, compresses an arbitrary number of tokens down to a fixed count |
| `pos_encoding.py` | Sinusoidal positional encoding (disabled by default) |
| `action_head.py` | Action classification head + keyframe detection head |

## How to Train

```bash
# 0) Preparation: each rally directory needs pose_data.json + annotations.json
#    Optionally pre-extract visual data (otherwise the raw_clip.mp4 is decoded on the fly
#    during training, which is slower):
python src/model/mst/extract_frames.py
python src/model/mst/extract_crops.py

# 1) Smoke test: run 1 sample for 1 epoch to verify the pipeline works end to end
python src/model/mst/train.py --config configs/main.yaml --smoke

# 2) Full training (main.yaml is the current best baseline)
python src/model/mst/train.py --config configs/main.yaml

# 3) Evaluation + confusion matrix
python src/model/mst/tests/eval_optimal.py --config configs/main.yaml --weights <best.pth>

# 4) Batch ablation / hyperparameter / component experiments
python src/model/mst/run_ablation.py
```

For configuration details, see [`configs/CONFIG_REFERENCE.md`](../../../configs/CONFIG_REFERENCE.md).
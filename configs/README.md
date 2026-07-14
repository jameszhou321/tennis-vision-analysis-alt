# configs/ — Training Configurations

YAML configurations are split into two groups: YOLO-related (court/person) and MSTFormer action recognition. For a complete item-by-item explanation, see [`CONFIG_REFERENCE.md`](./CONFIG_REFERENCE.md).

## YOLO Group (For ultralytics training)

| File | Purpose |
| --- | --- |
| `court_14pts_weighted.yaml` | Court 14 keypoints dataset configuration (current primary config) |
| `court_keypoints*.yaml` | Historical version configurations for court keypoints |
| `person_sorter_dataset.yaml` | Person classification dataset (2 classes: near-side / far-side) |

## MSTFormer Group

Unified baseline: `embed_dim=128`, `depth=8`, `use_pos_encoding=false`.

| Path | Purpose |
| --- | --- |
| `main.yaml` | **Current optimal baseline**, used directly for formal training |
| `main_shared.yaml` | Variant sharing the same YOLO backbone |
| `hyperparams/` | Hyperparameter tuning (embed_dim, depth, visual_tokens, etc.) |
| `ablation/` | Ablation studies (no pose / no crops / pure pose / global frame only) |
| `components/` | Component comparisons (Focal vs CE, token merging vs no merge, different backbones, etc.) |
| `single_frame/` | Single-frame classification baseline configurations |

## Usage

```bash
python src/model/mst/train.py --config configs/main.yaml
python src/model/mst/run_ablation.py        # Batch run ablation/components/hyperparams
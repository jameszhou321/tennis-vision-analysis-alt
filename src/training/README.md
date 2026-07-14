# training/ — Person Detection/Classification Model Training

Trains the "near-side player / far-side player" identification model, along with accompanying hard negative mining.

| File | Purpose | Run |
| --- | --- | --- |
| `train_person_detector.py` | Fine-tunes YOLO based on `data/person_sorter/` to distinguish `player_near` / `player_far` | `python src/training/train_person_detector.py` |
| `merge_hard_negatives.py` | Merges mined hard examples (misdetected ball boys/spectators, etc.) into the training set to improve discriminative power | `python src/training/merge_hard_negatives.py` |
| `yolo-train-legacy.py` | Old training script, kept for reference | — |

## Accompanying Data Tools (in `../utils/`)

```
data-creater.py      Samples images into data/person_sorter/image/
label_tool.py         Labels bounding boxes (near/far side)
dataset_splitter.py   Splits into train/val
─────────────────────────────────────────
train_person_detector.py   Training
hard_negative_extractor.py / hard_negative_reviewer.py   Hard-example mining and review (../utils/)
```

Dataset configuration: `configs/person_sorter_dataset.yaml`.
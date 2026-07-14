# model/yolo/ — Single-Frame YOLO Action Classification (Comparison Baseline)

A baseline model that treats action recognition as a **single-frame image classification** problem: YOLO11n backbone + global average pooling + classification head, outputting 5 action classes. Used for comparison against the temporal model MSTFormer, to illustrate the shortcomings of "frame-by-frame classification" relative to "temporal modeling".

| File | Purpose |
| --- | --- |
| `model.py` | Model definition `YoloFrameClassifier`: hooks into the final output of the YOLO backbone, followed by a classification head |
| `dataset.py` | Single-frame dataset: takes the action label for each frame from the rally frames + `annotations.json` |
| `train.py` | Training script |

## Running

```bash
python src/model/yolo/train.py
```

> Shares the same `annotations.json` annotations and action category definitions (Idle/Forehand/Backhand/Serve/Movement) with `model/mst/`.
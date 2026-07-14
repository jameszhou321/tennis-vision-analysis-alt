# demo/ — PyQt5 Desktop Visualization Demo

A desktop application combining video playback, a three-row timeline, and real-time MSTFormer inference visualization, used to intuitively inspect the model's predictions on a given rally.

## Running

```bash
python src/demo/main.py \
  --rally   data/rallies_annotated/rally_001_19.8s \
  --config  configs/main.yaml \
  --weights models/action/<config>/<timestamp>/best.pth \
  --person  models/person/best.pt \   # optional: if provided, runs real-time detection
  --pose    models/yolo/yolo11x-pose.pt  # optional
```

Parameters can also be selected in the UI. The `--person`/`--pose` arguments define two working modes:

- **Provided**: runs person detection + pose estimation in real time, drawing bboxes and skeleton overlays on the frame;
- **Not provided**: falls back to reading the pre-extracted `pose_data.json` and crop images from the rally directory.

Both modes feed the entire sequence into MSTFormer for inference at once.

## Files

| File | Purpose |
| --- | --- |
| `main.py` | Entry point. Parses arguments; handles the CUDA DLL and Qt plugin load order for torch/PyQt5 on Windows |
| `app.py` | Main window: video playback, three-row timeline, file/model selection, inference trigger, action legend |
| `player.py` | Video player: QTimer + OpenCV frame-by-frame reading, handles paths containing Chinese characters |
| `timeline.py` | Three-row timeline: GT annotation bar / prediction bar / frame grid bar, with a cursor that follows playback |
| `inference.py` | Inference thread (QThread), two modes as described above |
| `seq_len_sweep.py` | Sequence-length sweep script: iterates over different `seq_len` values and outputs an accuracy CSV |

> Depends on `PyQt5`; must be run in an environment with a graphical interface (remote servers require X11/VNC).
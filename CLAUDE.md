# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A computer-vision pipeline for tennis match video analysis: court detection → player pose tracking →
action recognition (idle/forehand/backhand/serve/movement). The core contribution is **MSTFormer**
(`src/model/mst/`), a multi-stream Transformer fusing pose sequences, court geometry, and player-crop
visuals with a dual-head output (5-class action + keyframe detection).

This is a research/thesis codebase, not a packaged library: no build step, no linter/formatter config,
and no unified test runner. Ad-hoc `test_*.py` scripts exist per-module but are run directly with
`python`, not via pytest.

## Setup

```bash
pip install -r requirements.txt   # or: uv sync (uv.lock is present)
```

Install `torch`/`torchvision` matching your CUDA version first (see requirements.txt comment), then the rest.
Python 3.11+ (see `.python-version` / `pyproject.toml`).

`videos/`, `data/`, `models/`, `runs/`, `logs/`, `results/` are gitignored — this repo ships source/configs/docs
only. Path conventions for these live in `src/config_legacy.py` and `configs/`.

## Common commands

All scripts are run from the **repository root** (they use root-relative paths).

```bash
# Batch pipeline: segment match videos into rallies, cut+compile with FFmpeg, overlay pose/ball annotations
# (edit mode="static"|"broadcast"|"fusion" and other args at the bottom of src/main.py)
python src/main.py

# Court keypoint model training
python src/train_court_pipeline.py

# MSTFormer training (the core model)
python src/model/mst/train.py --config configs/main.yaml
python src/model/mst/train.py --config configs/main.yaml --smoke   # 1 sample/1 epoch sanity check
python src/model/mst/tests/eval_optimal.py --config configs/main.yaml --weights <path/to/best.pth>
python src/model/mst/run_ablation.py    # batch-runs configs/ablation, configs/components, configs/hyperparams

# Person (near/far player) classifier training
python src/training/train_person_detector.py

# Action-timeline annotation UI (Flask, http://localhost:5000)
python src/utils/action_annotator.py

# Visualization demo (PyQt5, requires a graphical environment)
python src/demo/main.py --rally <rally_dir> --config configs/main.yaml --weights <best.pth>
```

There is no single "run the tests" command — run the relevant `test_*.py` / `tests/` script directly for the
area you're changing (e.g. `python src/model/mst/tests/test_dataset.py`).

## Architecture

Three sequential stages, each independently swappable:

```
Raw match video
  ├─[1] Court detection     YOLO 14-keypoint model → homography (top-down coords)
  ├─[2] Pose tracking       YOLO-pose (near/far player) → 17-pt skeleton, EMA smoothing, gap filling
  └─[3] Action recognition  MSTFormer → 5-class action + keyframe detection (dual head)
```

**Two distinct entry points touch this pipeline differently** — don't conflate them:
- `src/main.py` — batch rally segmentation (three interchangeable modes: `static` fence-cam velocity,
  `broadcast` PySceneDetect+CLIP, `fusion` audio/motion/ball hysteresis state machine via
  `src/audio_video_fusion.py`) + FFmpeg cut/concat + optional overlay rendering. Config in
  `src/config_legacy.py`.
- `src/pipeline/offline_tennis_tracker.py` — a separate, finer-grained two-pass tracker (per-frame
  weighted homography + BoT-SORT tracking + radar-view rendering), used for precision data production
  rather than batch processing.

**MSTFormer (`src/model/mst/`)**: three visual streams (full frame + player1 crop + player2 crop) are
each run through a swappable backbone (`modules/backbone_factory.py`: yolo11/resnet18/vit/raw), optionally
merged into shared tokens (`merge_visual_tokens`, via the Perceiver-style resampler in
`modules/token_resampler.py`), concatenated with a 125-dim pose-feature token sequence, and fed through a
Transformer to two heads (`modules/action_head.py`): action classification and keyframe detection.
Architecture toggles (`use_pose`/`use_player_crops`/`use_visual`/`merge_visual_tokens`/`parallel_backbones`)
are all config-driven — see `configs/CONFIG_REFERENCE.md` for the full field reference and validity matrix
per config group (there are non-obvious field interactions, e.g. `merge_visual_tokens=true` with
`use_player_crops=false` silently does nothing useful).

**Data flow** ties training data production to two parallel tracks that converge at `pose_data.json` +
`annotations.json` per rally:
```
main.py → data/rallies_new/ → utils/action_annotator.py → annotations.json (action labels)
                             → merge_annotating_data.py  → pose_data.json  (court + player keypoints)
```
`pose_data.json` (`court` 14-pt array + `near_player`/`far_player` 17-pt COCO skeletons) and
`annotations.json` (start/end/action_id time segments, classes `idle(0) forehand(1) backhand(2) serve(3)
movement(4)`) are the two formats nearly every downstream training/eval script consumes. `main.py`'s
direct output is video clips, not these JSON formats — they're produced by `src/pipeline/` and
`src/utils/` for model training.

Full per-file responsibilities and module dependency graphs: `docs/architecture.md` (also see
per-directory `README.md`s — `src/README.md`, `src/pipeline/README.md`, `src/model/mst/README.md`,
`src/training/README.md`, `src/utils/README.md`, `src/demo/README.md`).

Ball tracking has two interchangeable backends (`src/ball_tracker.py` classical CV, default; or
`src/ball_tracker_tracknet.py` wrapping the upstream TrackNet repo) — TrackNet requires manually
downloading `src/tracknet/{model.py,general.py}` and weights at `src/models/tracknet/model_best.pt`;
`main.py` falls back to the classical tracker with a console warning if these are absent.

## Conventions (see `docs/style_guide.md` for the full guide)

- Every `.py` file opens with a module docstring (`"""filename.py — one-line responsibility"""`), not a
  `# filename — ...` comment.
- Inline comments are written in **Chinese**, explaining *why*, not restating the code. No casual/emoji
  comments or print statements; console output uses neutral tags like `print("[Training] ...")`.
- Module-private helpers/constants are prefixed `_` (e.g. `_build_pose_vec`).
- Never hardcode absolute paths — scripts assume they're run from the repo root and use relative paths.
- Training hyperparameters belong in `configs/*.yaml`, not hardcoded in scripts; adding a new config field
  means updating `configs/CONFIG_REFERENCE.md` and `configs/README.md` too.
- Paths containing Chinese characters need Windows short-path handling (`_get_short_path` helper pattern
  used in `pose_tracker.py`/`demo/player.py`) — skipped automatically on non-Windows via a `hasattr(ctypes,
  "windll")` check.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).

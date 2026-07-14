# Code Style Guide

The project's source code follows the conventions below. Please keep them consistent before submitting code.

## 1. Module File Header

Every `.py` file starts with a **module docstring** describing its purpose, in this format:

```python
"""filename.py — one-line responsibility

Optional additional explanation (multiple lines).
"""
import ...
```

Do not use a `# filename — ...` comment line as the file header (this has been standardized to a docstring).

## 2. Comments

- Comments must always be written in **Chinese**, explaining "why this was done" rather than restating the code.
- Do not write casual/emotional development-process comments (such as "core magic", "killer move", "unchanged", "modified here", etc.), and do not use emoji in comments or `print` statements.
- Console output should use neutral prefix tags, e.g. `print("[Training] ...")`, `print("[Pass 1] ...")`.
- Classes and public functions should have docstrings where possible, describing input/output dimensions and key assumptions.

## 3. Naming and Formatting

- 4-space indentation; broadly follow PEP 8 style.
- Prefix module-private functions/constants with `_` (e.g. `_build_pose_vec`, `_MEAN`).
- Import order: standard library → third-party → project modules.

## 4. Paths and Cross-Platform Compatibility

- **Do not hardcode absolute paths** (especially local machine paths). Scripts should default to being run from the repository root, using relative paths or `argparse` arguments.
- Large-file directories (`videos/`, `data/`, `models/`, `runs/`) are excluded by `.gitignore` and not checked into the repo.
- When handling paths containing Chinese characters, use short paths on Windows to avoid OpenCV encoding issues; on non-Windows systems this must be skippable:

```python
def _get_short_path(path):
    buf = ctypes.create_unicode_buffer(512)
    if not hasattr(ctypes, "windll"):   # Use the original path directly on non-Windows systems
        return path
    ctypes.windll.kernel32.GetShortPathNameW(path, buf, 512)
    return buf.value or path
```

## 5. Configuration

- Training hyperparameters go in `configs/*.yaml`; do not hardcode them in scripts.
- When adding a new config, update `configs/CONFIG_REFERENCE.md` and `configs/README.md` accordingly.

## 6. Data Format Conventions

- Action annotations: `annotations.json`, action categories `Idle(0) Forehand(1) Backhand(2) Serve(3) Movement(4)`.
- Pose data: `pose_data.json`, each frame contains `court` (14 points) / `near_player` / `far_player`.
- See the "Data Format" section of the root README for details.
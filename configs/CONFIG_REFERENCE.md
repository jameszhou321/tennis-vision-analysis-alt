# MSTFormer Configuration Fields Reference Manual

> Automatically generated based on code analysis, covering `model_main.py`, `backbone_factory.py`, `train.py`, `dataset.py`, and `config.py`.
> Updated: 2026-04-24

---

## 1. Data Fields

| Field | Required | Default Value | Role / Purpose |
|------|------|--------|------|
| `data_root` | ✅ | — | Root directory for training data (contains individual rally subdirectories) |
| `crops_root` | ✅ | — | Root directory for player crops (typically the same as data_root) |
| `seq_len` | ✅ | — | Sequence frame length per sample (e.g., 120) |
| `min_seq_len` | ❌ | `max(30, seq_len//2)` | Shortest slice frame length during reshuffling; only takes effect when `reshuffle_augment=true` |
| `train_ratio` | ✅ | — | Training set split ratio (0 to 1) |
| `num_classes` | ✅ | — | Number of action classification classes (currently 5) |
| `class_weights` | ✅ | — | List of loss weights for each class; length must equal `num_classes` |

---

## 2. Model Architecture Switches

| Field | Required | Default Value | Role / Purpose | Conflict / Dependency |
|------|------|--------|------|-----------|
| `use_visual` | ❌ | `true` | Whether to enable the visual stream; `false` sets the model to pure pose mode | When `false`, all visual-related fields become ineffective |
| `use_player_crops` | ❌ | `true` | Whether to load player cropped images (p1/p2); when `false`, only the full-frame stream is used | Only takes effect when `use_visual=true` |
| `use_pose` | ❌ | `true` | Whether to use pose features; when `false`, the pose vector is set to zero (token slot is still retained) | None |
| `use_pos_encoding` | ❌ | `false` | Whether to enable sinusoidal position encoding; when closed, the model does not rely on absolute positional info | None |
| `keyframe_only` | ❌ | `false` | When `true`, trains only the keyframe detection head and skips building the action_head | When `true`, `keyframe_loss_weight` becomes ineffective |
| `merge_visual_tokens` | ❌ | `false` | When `true`, the three-stream tokens are concatenated and passed through a shared_resampler down to `visual_tokens` | Only meaningful when `use_visual=true` and `use_player_crops=true` |
| `parallel_backbones` | ❌ | `false` | When `true`, the three-stream backbones run in parallel (requires more VRAM, prone to single-card OOM) | Only takes effect when `use_visual=true` and `use_player_crops=true` |

---

## 3. Visual Backbone Networks

### General Fields

| Field | Required | Default Value | Role / Purpose | Trigger Condition |
|------|------|--------|------|----------|
| `visual_backbone` | ❌ | `"yolo11"` | Backbone type: `yolo11` / `vit` / `resnet18` / `raw` | `use_visual=true` |
| `visual_tokens` | ❌ | `16` | Final number of visual tokens outputted per stream | Controls TokenResampler output when `visual_backbone="yolo11"`; controls total combined number when `merge_visual_tokens=true`; **ineffective in vit mode** |

### yolo11-Specific

| Field | Required | Default Value | Role / Purpose | Trigger Condition |
|------|------|--------|------|----------|
| `backbone_weights` | ✅* | — | Weights path for p1/p2 backbones (yolo11n-pose.pt) | `visual_backbone="yolo11"` |
| `global_backbone_weights` | ❌ | Same as `backbone_weights` | Weights path for full-frame backbone (yolo11n.pt); if left blank, shares the same weights as p1/p2 | `visual_backbone="yolo11"` and `use_player_crops=true` |
| `unfreeze_backbone` | ❌ | `false` | When `true`, unfreezes backbone parameters and enables gradient checkpointing | `visual_backbone="yolo11"`; **ineffective in vit mode** |
| `multi_scale_levels` | ✅* | — | Multi-scale feature levels, e.g., `[3, 4, 5]` corresponding to P3/P4/P5 | `visual_backbone="yolo11"` |
| `tokens_per_scale` | ✅* | — | Number of tokens after spatial pooling for each scale (e.g., 4 → 2×2) | `visual_backbone="yolo11"` |

> ✅* indicates that the field is required when `visual_backbone="yolo11"` (hard-coded read, missing it will throw a KeyError). In vit mode, placeholders can be used for these fields as they will not be read.

### vit-Specific

| Field | Required | Default Value | Role / Purpose | Trigger Condition |
|------|------|--------|------|----------|
| `vit_patch_grid` | ❌ | `4` | Patch grid size; patch tokens are compressed down to `visual_tokens` via TokenResampler | `visual_backbone="vit"` |

> `vit_depth` / `vit_num_heads` are deprecated (following the session15 refactor, ViT has no internal Transformer; the fields remain for compatibility but are not read).

---

## 4. Model Hyperparameters

| Field | Required | Default Value | Role / Purpose |
|------|------|--------|------|
| `embed_dim` | ✅ | — | Model embedding dimension (the uniform dimension for all tokens) |
| `depth` | ✅ | — | Number of primary TransformerEncoder layers |
| `num_heads` | ✅ | — | Number of attention heads for the main Transformer (must be divisible by `embed_dim`) |
| `dropout` | ❌ | `0.1` | Dropout ratio for the Transformer layers |

---

## 5. Training Hyperparameters

| Field | Required | Default Value | Role / Purpose | Conflict / Dependency |
|------|------|--------|------|-----------|
| `batch_size` | ✅ | — | Actual batch size | Must be evenly divisible by `virtual_batch_size` |
| `virtual_batch_size` | ✅ | — | Virtual batch size; `accumulation_steps = virtual_batch_size / batch_size` | Must be an integer multiple of `batch_size` |
| `total_epochs` | ✅ | — | Total number of training epochs | |
| `learning_rate` | ✅ | — | Initial learning rate (AdamW) | |
| `weight_decay` | ✅ | — | Weight decay coefficient (AdamW) | |
| `warmup_epochs` | ❌ | `5` | Linear warmup epochs, followed by cosine annealing down to `lr × 0.01` | |
| `loss` | ❌ | `"cross_entropy"` | Loss function: `cross_entropy` or `focal` | |
| `focal_gamma` | ❌ | `2.0` | The γ parameter for Focal Loss | Only takes effect when `loss="focal"` |
| `keyframe_loss_weight` | ❌ | `0.5` | Keyframe loss weight: `total = loss_action + weight × loss_kf` | Only takes effect when `keyframe_only=false` |

---

## 6. Hardware & Data Loading

| Field | Required | Default Value | Role / Purpose |
|------|------|--------|------|
| `num_workers` | ✅ | — | Number of DataLoader worker processes (2 is recommended for Windows) |
| `pin_memory` | ✅ | — | Whether to lock memory to accelerate GPU data transfer |
| `reshuffle_augment` | ❌ | `true` | When `true`, randomly re-slices training segments each epoch (temporal augmentation) |
| `image_augment` | ❌ | `false` | When `true`, enables image-level data augmentation on the training set: color jitter / Gaussian noise / blur / random erasing / semi-transparent overlays |
| `transformer_checkpoint` | ❌ | `true` | When `true`, enables gradient checkpointing for Transformer layers, saving roughly 50% VRAM |

---

## 7. Automatically Generated Fields (Do Not Manually Fill)

The following fields are automatically injected at runtime by `config.py` or `train.py` and should not appear in your YAML files:

| Field | Source | Explanation |
|------|------|------|
| `device` | `config.py` | Automatically detects cuda / cpu |
| `accumulation_steps` | `config.py` | Calculated as `virtual_batch_size / batch_size` |
| `_yaml_path` | `train.py` | File path of the configuration file, used for logging |
| `_smoke_clip` | `train.py` | Test clip path used under `--smoke` mode |

---

## 8. Configuration Field Validity Quick Reference Matrix

### Main Configuration

| Field | main |
|------|:----:|
| `use_visual` | ✅ |
| `use_player_crops` | ✅ |
| `use_pose` | ✅ |
| `use_pos_encoding` | ❌ false |
| `merge_visual_tokens` | ✅ true |
| `unfreeze_backbone` | ✅ |

### ablation/

| Field | abl_no_pose | abl_no_crops | abl_no_visual | abl_global_only |
|------|:-----------:|:------------:|:-------------:|:---------------:|
| `use_visual` | ✅ | ✅ | ❌ false | ✅ |
| `use_player_crops` | ✅ | ❌ false | ❌ false | ❌ false |
| `use_pose` | ❌ false (set to zero) | ✅ | ✅ | ❌ false (set to zero) |

### components/

| Field | cmp_focal_loss | cmp_ce_loss | cmp_no_merge | cmp_resnet_backbone | cmp_frozen_backbone |
|------|:--------------:|:-----------:|:------------:|:-------------------:|:-------------------:|
| `loss` | focal | cross_entropy | focal | focal | focal |
| `merge_visual_tokens` | ✅ true | ✅ true | ❌ false | ✅ true | ✅ true |
| `visual_backbone` | yolo11 | yolo11 | yolo11 | resnet18 | yolo11 |
| `unfreeze_backbone` | ✅ | ✅ | ✅ | ✅ | ❌ false |

> "Placeholder": The field exists but is not read by the code (due to hitting a different branch). Deleting it will cause a KeyError. "Ineffective": The field presence or value does not affect the program outcome.

---

## 9. Common Configuration Errors

| Error | Consequence |
|------|------|
| `num_heads` is not evenly divisible by `embed_dim` | Throws a runtime error. |
| `batch_size` cannot evenly divide `virtual_batch_size` | Gradient accumulation steps evaluate to a float; training behavior becomes abnormal. |
| Setting `visual_tokens` while in `vit` mode | Ineffective; the token count is fixed by `vit_patch_grid²`. |
| `merge_visual_tokens=true` but `use_player_crops=false` | The shared_resampler only processes a single stream; token merging is meaningless here. |
| Setting `focal_gamma` when `loss="cross_entropy"` | Ineffective; it doesn't affect execution but can be highly misleading. |
| Setting `keyframe_loss_weight` when `keyframe_only=true` | Ineffective; there is only keyframe loss, so no action loss exists to weight against it. |
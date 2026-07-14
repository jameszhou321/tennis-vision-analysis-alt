"""model_main.py — MSTFormer Model Definition (Dual-Head: Action Classification + Keyframe Detection)"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # Ensure mst/ is in the system path

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from modules.backbone_factory import build_visual_extractor
from modules.pos_encoding import SinusoidalPositionalEncoding
from modules.action_head import ActionClassificationHead, KeyframeDetectionHead
from modules.token_resampler import TokenResampler

# ImageNet normalization constants performed on the GPU side (avoids float32 PCIe bandwidth overhead from CPU)
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def _unpack_and_normalize(packed, device):
    """packed: uint8 [N, 3, 320, 960] → 3-way float32 each [N, 3, 320, 320], normalized on GPU"""
    mean = _MEAN.to(device)
    std  = _STD.to(device)
    x = packed.float().div_(255.0)
    global_f = (x[:, :, :, :320]   - mean) / std
    p1_f     = (x[:, :, :, 320:640] - mean) / std
    p2_f     = (x[:, :, :, 640:]   - mean) / std
    return global_f, p1_f, p2_f


class MSTFormer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg["embed_dim"]
        self.use_visual = cfg.get("use_visual", True)
        self.use_player_crops = cfg.get("use_player_crops", True)
        self.use_transformer_ckpt = cfg.get("transformer_checkpoint", True)
        self.merge_visual_tokens = cfg.get("merge_visual_tokens", False)
        self.parallel_backbones = cfg.get("parallel_backbones", False)
        self.share_yolo_backbone = cfg.get("share_yolo_backbone", False)

        self.physics_extractor = nn.Sequential(
            nn.Linear(125, 256),
            nn.ReLU(),
            nn.Linear(256, d)
        )

        if self.use_visual:
            self.backbone_global = build_visual_extractor(cfg, use_global_weights=True)
            if self.use_player_crops:
                # When share_yolo_backbone is active, p1/p2 reuse the global backbone weights, saving ~5.7M parameters
                _shared = self.backbone_global.backbone if self.share_yolo_backbone else None
                self.backbone_p1 = build_visual_extractor(cfg, shared_backbone=_shared)
                self.backbone_p2 = build_visual_extractor(cfg, shared_backbone=_shared)

        # Dynamically probe the actual number of output tokens from each visual stream
        if self.use_visual:
            import torch as _torch
            _dummy = _torch.zeros(1, 3, 320, 320)
            self.backbone_global.eval()
            with _torch.no_grad():
                _k = self.backbone_global(_dummy).shape[1]
            if cfg.get("unfreeze_backbone", False):
                self.backbone_global.train()
            vis_streams = 3 if self.use_player_crops else 1
            if self.merge_visual_tokens:
                # Concatenate 3 streams and pass through a shared resampler to compress down to `visual_tokens`
                visual_tokens = cfg.get("visual_tokens", 16)
                self.shared_resampler = TokenResampler(d, num_out=visual_tokens)
                self.tokens_per_frame = 1 + visual_tokens
            else:
                # Each stream independently outputs `_k` tokens
                self.tokens_per_frame = 1 + vis_streams * _k
        else:
            self.tokens_per_frame = 1

        self.pos_embed = SinusoidalPositionalEncoding(d)
        self.use_pos_encoding = cfg.get("use_pos_encoding", False)

        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d,
                nhead=cfg["num_heads"],
                dim_feedforward=d * 4,
                dropout=cfg.get("dropout", 0.1),
                batch_first=True,
                activation="gelu",
                norm_first=True,  # pre-norm, a prerequisite for Flash Attention fast path
            )
            for _ in range(cfg["depth"])
        ])

        self.keyframe_only = cfg.get("keyframe_only", False)
        self.use_pose = cfg.get("use_pose", True)
        if not self.keyframe_only:
            self.action_head = ActionClassificationHead(d, cfg["num_classes"])
        self.keyframe_head = KeyframeDetectionHead(d)

    def _run_visual(self, extractor, frames, B, T):
        flat = frames.view(B * T, *frames.shape[2:])
        tokens = extractor(flat)
        return tokens.view(B, T, -1, self.cfg["embed_dim"])

    def forward(self, pose_data, packed_frames):
        B, T = pose_data.shape[:2]
        d = self.cfg["embed_dim"]

        phys = self.physics_extractor(
            pose_data if self.use_pose else torch.zeros_like(pose_data)
        ).unsqueeze(2)  # (B, T, 1, D)
        parts = [phys]

        if self.use_visual:
            # GPU-side unpacking: [B, T, 3, 320, 960] → 3-way [B, T, 3, 320, 320]
            flat_packed = packed_frames.view(B * T, 3, 320, 960)
            global_f, p1_f, p2_f = _unpack_and_normalize(flat_packed, pose_data.device)
            global_f = global_f.view(B, T, 3, 320, 320)

            vis_parts = [self._run_visual(self.backbone_global, global_f, B, T)]
            if self.use_player_crops:
                p1_f = p1_f.view(B, T, 3, 320, 320)
                p2_f = p2_f.view(B, T, 3, 320, 320)
                if self.parallel_backbones:
                    # Run 3 streams in independent CUDA streams, then synchronize after submission
                    s1 = torch.cuda.Stream()
                    s2 = torch.cuda.Stream()
                    with torch.cuda.stream(s1):
                        tok_p1 = self._run_visual(self.backbone_p1, p1_f, B, T)
                    with torch.cuda.stream(s2):
                        tok_p2 = self._run_visual(self.backbone_p2, p2_f, B, T)
                    torch.cuda.synchronize()
                    vis_parts += [tok_p1, tok_p2]
                else:
                    vis_parts.append(self._run_visual(self.backbone_p1, p1_f, B, T))
                    vis_parts.append(self._run_visual(self.backbone_p2, p2_f, B, T))

            if self.merge_visual_tokens:
                # 3-way cat → (B*T, total_k, D) → shared_resampler → (B, T, visual_tokens, D)
                merged = torch.cat(vis_parts, dim=2).view(B * T, -1, d)
                merged = self.shared_resampler(merged).view(B, T, -1, d)
                parts.append(merged)
            else:
                parts.extend(vis_parts)

        frame_tokens = torch.cat(parts, dim=2)             # (B, T, tokens_per_frame, D)

        x = frame_tokens.view(B, T * self.tokens_per_frame, d)
        if self.use_pos_encoding:
            x = self.pos_embed(x)

        for layer in self.transformer_layers:
            if self.use_transformer_ckpt and self.training:
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)

        x = x.view(B, T, self.tokens_per_frame, d).mean(dim=2)  # (B, T, D)
        kf = self.keyframe_head(x)
        if self.keyframe_only:
            return kf                                             # (B, T, 2)
        return self.action_head(x), kf                           # (B, T, num_classes), (B, T, 2)
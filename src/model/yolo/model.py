"""
YOLO Single-Frame Action Classification — Model Definition
Uses YOLO11n backbone + Global Average Pooling + Classification Head
"""

import torch
import torch.nn as nn
from ultralytics import YOLO


class YoloFrameClassifier(nn.Module):
    """YOLO11 backbone (feature extractor inclusive) + Classification head for single-frame 5-class action classification."""

    def __init__(self, weights_path, num_classes=5, unfreeze_backbone=True, img_size=224):
        super().__init__()
        self.img_size = img_size
        yolo = YOLO(weights_path)
        self.model = yolo.model          # DetectionModel, whose forward pass handles skip connections

        # YOLO11n P5 output channel size is 256
        feat_ch = 256

        # Register a hook to capture the final backbone feature output right before the Detect head
        self._captured = None
        self._hook_handle = self.model.model[22].register_forward_hook(self._capture)

        # Classification Head
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_ch, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

        if unfreeze_backbone:
            for p in self.model.parameters():
                p.requires_grad_(True)
        else:
            for p in self.model.parameters():
                p.requires_grad_(False)

        # ImageNet normalization constants (aligned with MST config)
        self.register_buffer("_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _capture(self, module, input, output):
        self._captured = output

    def forward(self, x):
        # x: (B, 3, H, W) float32, ImageNet normalized
        # Input resizing and normalization are handled by the dataset, forward directly here
        _ = self.model(x)                     # DetectionModel.forward
        feat = self._captured                  # (B, C, fH, fW)
        pooled = self.pool(feat).flatten(1)    # (B, C)
        return self.classifier(pooled)         # (B, 5)


if __name__ == "__main__":
    model = YoloFrameClassifier("models/yolo/yolo11n.pt", num_classes=5)
    dummy = torch.randn(4, 3, 224, 224)
    out = model(dummy)
    print(f"Output: {out.shape}")
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {total:,} total, {trainable:,} trainable")
"""action_head.py — Action Classification Head and Keyframe Detection Head (Identical Architecture)"""
import torch.nn as nn


class ActionClassificationHead(nn.Module):
    def __init__(self, embed_dim=128, num_classes=5):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(embed_dim // 2, num_classes)
        )

    def forward(self, x):
        return self.head(x)


class KeyframeDetectionHead(nn.Module):
    def __init__(self, embed_dim=128):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(embed_dim // 2, 2)
        )

    def forward(self, x):
        return self.head(x)
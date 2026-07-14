"""vit_extractor.py — Lightweight ViT Patch Embedding Feature Extractor"""
import torch
import torch.nn as nn


class ViTPatchExtractor(nn.Module):
    """Lightweight patch embedding: Splits the input image into a patch_grid x patch_grid grid 

    of 16x16 patches. Each patch is flattened and projected through a two-layer FFN to embed_dim, 
    directly outputting tokens.
    
    Contains no internal Transformer layer blocks; sequence-level feature extraction is deferred 
    to the primary MSTFormer architecture.

    Input:  (B, 3, H, W)
    Output: (B, num_patches, embed_dim) where num_patches = patch_grid^2
    """

    def __init__(self, patch_grid, embed_dim, vit_depth=2, num_heads=4):
        # vit_depth and num_heads are preserved to maintain interface compatibility but are unused
        super().__init__()
        self.patch_grid = patch_grid
        patch_px = 16
        in_dim = 3 * patch_px * patch_px  # 768

        self.pool = nn.AdaptiveAvgPool2d((patch_grid * patch_px, patch_grid * patch_px))
        self.ffn = nn.Sequential(
            nn.Linear(in_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, x):
        B = x.shape[0]
        g = self.patch_grid
        patch_px = 16

        x = self.pool(x)                                          # (B, 3, g*16, g*16)
        x = x.view(B, 3, g, patch_px, g, patch_px)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()            # (B, g, g, 3, 16, 16)
        x = x.view(B, g * g, 3 * patch_px * patch_px)            # (B, num_patches, 768)
        return self.ffn(x)                                        # (B, num_patches, embed_dim)
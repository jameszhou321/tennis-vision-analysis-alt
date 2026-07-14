"""raw_extractor.py — Raw Pixel Projection Feature Extractor"""
import torch.nn as nn
from .token_resampler import TokenResampler


class RawProjectionExtractor(nn.Module):
    """Directly applies AdaptiveAvgPool to raw frame pixels, projects them linearly, 

    and passes the result through a TokenResampler to output a fixed number of visual_tokens.
    """

    def __init__(self, tokens_per_scale, embed_dim, visual_tokens=16):
        super().__init__()
        pool_size = int(tokens_per_scale ** 0.5)
        self.pool = nn.AdaptiveAvgPool2d(pool_size)
        self.proj = nn.Linear(3 * pool_size * pool_size, embed_dim)
        self.tokens_per_scale = tokens_per_scale
        self.resampler = TokenResampler(embed_dim, num_out=visual_tokens)

    def forward(self, x):
        x = self.pool(x)                                                        # (B, 3, pool, pool)
        x = x.flatten(1).unsqueeze(1).expand(-1, self.tokens_per_scale, -1)    # (B, k, 3*pool*pool)
        x = self.proj(x)                                                        # (B, k, D)
        return self.resampler(x)                                                # (B, visual_tokens, D)
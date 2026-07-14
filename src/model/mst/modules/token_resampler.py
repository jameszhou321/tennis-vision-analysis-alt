"""token_resampler.py — Token Resampler Module (Perceiver-style cross-attention)"""
import torch
import torch.nn as nn


class TokenResampler(nn.Module):
    """Resamples an arbitrary number of input tokens down to a fixed num_out token count.
    
    Uses num_out learnable queries to perform cross-attention over the inputs (Perceiver-style).

    Input:  (B, N_in, D)
    Output: (B, num_out, D)
    """

    def __init__(self, embed_dim, num_out=16, num_heads=4):
        super().__init__()
        self.queries = nn.Parameter(torch.zeros(1, num_out, embed_dim))
        nn.init.trunc_normal_(self.queries, std=0.02)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads,
                                                batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        q = self.queries.expand(x.shape[0], -1, -1)
        out, _ = self.cross_attn(q, x, x)
        return self.norm(out + q)
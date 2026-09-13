"""One transformer block: attention, then feed-forward, each wrapped in a residual.

INPUT   x  (B, T, D)
OUTPUT  x  (B, T, D)    -- same shape, which is what makes blocks stackable

THE LAYOUT (pre-normalization)

    x = x + Attention(LayerNorm(x))
    x = x + FeedForward(LayerNorm(x))

RESIDUAL CONNECTIONS  (the "x + ...")
    Each sub-layer computes a CORRECTION that is added to its input, rather than
    a replacement for it. Two consequences:

    1. At initialization the corrections are small, so the whole stack starts
       out close to the identity function -- a stable place to begin.
    2. During backpropagation the gradient of (x + f(x)) with respect to x is
       (1 + f'(x)). That "1" is an uninterrupted path from the loss straight back
       to the earliest layers. Without it, gradients shrink as they pass through
       each layer, and deep stacks stop learning at the bottom.

    The stream of vectors flowing through these additions is often called the
    residual stream. Every block reads from it and writes back into it.

WHY PRE-NORM
    The original Transformer normalized AFTER the addition (post-norm). Pre-norm
    normalizes each sub-layer's INPUT instead, which leaves the residual path
    itself untouched. It trains stably without delicate learning-rate warmup,
    which matters when the hyperparameters have not been tuned yet.

PARAMETERS  12 * D^2 + 13 * D    (small config: 1,774,464 per block)
    attention 4D^2 + 4D, feed-forward 8D^2 + 5D, two LayerNorms 2D each.
"""

from __future__ import annotations

import torch
from torch import nn

from .attention import CausalSelfAttention
from .feed_forward import FeedForward
from .normalization import LayerNorm


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        context_length: int,
        ffn_multiplier: int = 4,
        dropout: float = 0.0,
        fused: bool = True,
    ) -> None:
        super().__init__()
        self.attention_norm = LayerNorm(embedding_dim, fused=fused)
        self.attention = CausalSelfAttention(
            embedding_dim, num_heads, context_length, dropout=dropout, fused=fused
        )
        self.feed_forward_norm = LayerNorm(embedding_dim, fused=fused)
        self.feed_forward = FeedForward(embedding_dim, ffn_multiplier, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))            # (B, T, D)
        x = x + self.feed_forward(self.feed_forward_norm(x))      # (B, T, D)
        return x

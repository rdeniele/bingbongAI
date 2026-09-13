"""The transformer stack: L blocks in sequence, then a final LayerNorm.

INPUT   x  (B, T, D)    embeddings
OUTPUT  x  (B, T, D)    contextualised vectors, one per position

Each block refines the residual stream. Early blocks tend to capture local,
surface-level patterns; later blocks build on them. The vector at position t
after the final block is the model's complete summary of tokens 0..t -- and
it is the ONLY thing the language-model head sees when predicting token t+1.

The final LayerNorm exists because of pre-norm: every block normalizes its
sub-layer inputs, but the residual stream itself is never normalized along the
way, so its scale grows with depth. One normalization at the end puts it back
on a consistent scale before the output projection.
"""

from __future__ import annotations

import torch
from torch import nn

from .normalization import LayerNorm
from .transformer_block import TransformerBlock


class Transformer(nn.Module):
    def __init__(
        self,
        num_layers: int,
        embedding_dim: int,
        num_heads: int,
        context_length: int,
        ffn_multiplier: int = 4,
        dropout: float = 0.0,
        fused: bool = True,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            TransformerBlock(
                embedding_dim, num_heads, context_length,
                ffn_multiplier=ffn_multiplier, dropout=dropout, fused=fused,
            )
            for _ in range(num_layers)
        )
        self.final_norm = LayerNorm(embedding_dim, fused=fused)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)                                         # (B, T, D)
        return self.final_norm(x)                                # (B, T, D)

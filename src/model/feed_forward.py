"""The position-wise feed-forward network.

PURPOSE
    Attention moves information BETWEEN positions. The feed-forward network
    PROCESSES what has arrived at each position. It is applied to every token
    independently, with the same weights -- no information crosses positions
    here.

    Two thirds of each block's parameters live in this layer. A common
    interpretation is that attention decides what to look at, and the
    feed-forward layers hold much of what the model has learned about what
    things mean.

INPUT   x    (B, T, D)
OUTPUT  out  (B, T, D)

MATH
    hidden = GELU(x W_1 + b_1)          (B, T, 4D)   expand
    out    = hidden W_2 + b_2           (B, T, D)    contract

    The 4x expansion is the original Transformer's convention. The non-linearity
    in the middle is essential: without it, two linear layers collapse into one
    linear layer and the whole block could only ever compute linear functions.

    GELU(x) = x * Phi(x), where Phi is the standard normal CDF. Unlike ReLU it
    is smooth around zero and lets small negative values through, which tends
    to train slightly better in transformers.

PARAMETERS  8 * D^2 + 5 * D    (small config: 1,181,568 per layer)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class FeedForward(nn.Module):
    def __init__(self, embedding_dim: int, multiplier: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        hidden_dim = multiplier * embedding_dim
        self.expand = nn.Linear(embedding_dim, hidden_dim)       # W_1: D -> 4D
        self.contract = nn.Linear(hidden_dim, embedding_dim)     # W_2: 4D -> D
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.expand(x))                          # (B, T, 4D)
        return self.dropout(self.contract(hidden))               # (B, T, D)

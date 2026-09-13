"""Layer normalization.

PURPOSE
    Keep the scale of each token's vector under control as it passes through
    many layers. Without normalization, activations drift larger or smaller
    layer after layer, and training becomes unstable.

INPUT   (..., D)   any tensor whose last dimension is the embedding dimension
OUTPUT  (..., D)   same shape

MATH
    For one token vector x of length D:

        mean  = (1/D) * sum_i x_i
        var   = (1/D) * sum_i (x_i - mean)^2
        x_hat = (x - mean) / sqrt(var + eps)      <- zero mean, unit variance
        y     = gamma * x_hat + beta              <- learned scale and shift

    gamma (starts at 1) and beta (starts at 0) are trained, so the network can
    undo the normalization wherever that turns out to help.

    Note what is normalized over: the D features of ONE token at ONE position.
    Nothing is shared across the batch or across positions -- which is why
    LayerNorm, unlike BatchNorm, behaves identically at batch size 1 and
    during generation.

TWO IMPLEMENTATIONS, ONE DEFINITION
    `manual_forward` is the specification: the formula above, line for line.
    `forward` calls PyTorch's fused kernel instead. It computes the same function
    (tests/test_model.py asserts both the outputs and the gradients match) but
    keeps far fewer intermediate tensors alive for backpropagation. On a 4 GiB
    GPU that memory difference is measurable -- see scripts/check_model.py.
    Set `fused=False` to train on the manual version directly.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5, fused: bool = True) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.fused = fused
        self.weight = nn.Parameter(torch.ones(dim))   # gamma
        self.bias = nn.Parameter(torch.zeros(dim))    # beta

    def manual_forward(self, x: torch.Tensor) -> torch.Tensor:
        """The formula, written out. (..., D) -> (..., D)."""
        mean = x.mean(dim=-1, keepdim=True)                            # (..., 1)
        variance = (x - mean).pow(2).mean(dim=-1, keepdim=True)        # (..., 1)
        x_hat = (x - mean) / torch.sqrt(variance + self.eps)           # (..., D)
        return self.weight * x_hat + self.bias                         # (..., D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.fused:
            return self.manual_forward(x)
        return F.layer_norm(x, (self.dim,), self.weight, self.bias, self.eps)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}, fused={self.fused}"

"""Causal multi-head self-attention -- the mechanism that lets tokens read each other.

PURPOSE
    Every other layer in the model processes each position in isolation.
    Attention is the ONLY place information moves between positions. When the
    model predicts the word after "The color of snow is", attention is how the
    position of "is" pulls in what "snow" means.

INPUT   x    (B, T, D)
OUTPUT  out  (B, T, D)

THE MATH, STEP BY STEP  (shapes for the small config: D=384, H=6, Dh=64)

 1. Project each token three ways with learned matrices:
        q = x W_q   "what am I looking for?"      (B, T, D)
        k = x W_k   "what do I contain?"          (B, T, D)
        v = x W_v   "what do I hand over if picked?"  (B, T, D)

 2. Split D into H heads of size Dh = D / H:
        (B, T, D) -> (B, T, H, Dh) -> (B, H, T, Dh)
    Each head runs its own independent attention pattern. Because H * Dh = D,
    six heads of 64 cost the same as one head of 384.

 3. Score every query against every key:
        scores = q @ k^T / sqrt(Dh)                (B, H, T, T)
    scores[b, h, i, j] = how relevant position j is to position i.

    Why divide by sqrt(Dh): if q and k have unit-variance entries, their dot
    product has variance Dh. At Dh=64 that is a standard deviation of 8, large
    enough to push softmax into a near one-hot regime where gradients vanish.
    Dividing restores unit variance.

 4. Causal mask -- THIS is what makes it a language model:
        scores[i, j] = -inf  wherever j > i
    Position i may look at positions 0..i and never at the future. During
    training the model predicts every next token in parallel; without the mask,
    position i could simply read token i+1 and copy it. Loss would look superb
    and the model would learn nothing usable. tests/test_attention.py proves
    the mask holds by changing future tokens and checking earlier outputs do
    not move by a single bit.

 5. Softmax over each row turns scores into weights that sum to 1:
        weights = softmax(scores, dim=-1)          (B, H, T, T)
    exp(-inf) = 0, so masked positions receive exactly zero weight.

 6. Take the weighted average of the values:
        out = weights @ v                          (B, H, T, Dh)

 7. Merge heads and mix them with one more projection:
        (B, H, T, Dh) -> (B, T, H, Dh) -> (B, T, D) -> W_o -> (B, T, D)

PARAMETERS  4 * D^2 + 4 * D    (small config: 591,360 per layer)
    W_q, W_k, W_v are stored as one (3D x D) matrix for efficiency; it is
    mathematically identical to three separate D x D matrices.

MEMORY NOTE
    The (B, H, T, T) score tensor is the expensive part. For the small config
    at batch 16: 16 * 6 * 512 * 512 = 25.2M entries, about 96 MiB in fp32 --
    per layer, before softmax outputs and backward storage are counted.
    `fused=True` uses PyTorch's scaled_dot_product_attention kernel, which
    computes the SAME function without materialising that tensor. Tests assert
    equivalence; scripts/check_model.py measures the memory difference.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        context_length: int,
        dropout: float = 0.0,
        fused: bool = True,
    ) -> None:
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.context_length = context_length
        self.dropout_p = dropout
        self.fused = fused

        # W_q, W_k, W_v in one matrix: D -> 3D, then split.
        self.qkv_proj = nn.Linear(embedding_dim, 3 * embedding_dim)
        # W_o: mixes the concatenated heads back together.
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Lower-triangular boolean matrix: mask[i, j] is True where j <= i,
        # i.e. where attention is ALLOWED. Built once at max size, sliced per call.
        # persistent=False: derived from context_length, so it is not saved in
        # checkpoints and cannot disagree with the config.
        mask = torch.tril(torch.ones(context_length, context_length, dtype=torch.bool))
        self.register_buffer("causal_mask", mask.view(1, 1, context_length, context_length),
                             persistent=False)

    def _split_heads(self, t: torch.Tensor) -> torch.Tensor:
        """(B, T, D) -> (B, H, T, Dh)"""
        batch, seq_len, _ = t.shape
        return t.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, D)
            return_attention: also return the (B, H, T, T) weight matrix. Forces
                the manual path, since the fused kernel never builds that matrix.

        Returns:
            out (B, T, D), or (out, weights) when return_attention is True.
        """
        batch, seq_len, dim = x.shape
        if seq_len > self.context_length:
            raise ValueError(
                f"sequence length {seq_len} exceeds context length {self.context_length}"
            )

        # Step 1-2: project, split into q/k/v, then into heads.
        q, k, v = self.qkv_proj(x).split(self.embedding_dim, dim=-1)    # each (B, T, D)
        q = self._split_heads(q)                                        # (B, H, T, Dh)
        k = self._split_heads(k)                                        # (B, H, T, Dh)
        v = self._split_heads(v)                                        # (B, H, T, Dh)

        weights: torch.Tensor | None = None
        if self.fused and not return_attention:
            # Steps 3-6 in one kernel. Same math, no (B, H, T, T) tensor kept.
            out = F.scaled_dot_product_attention(
                q, k, v,
                is_causal=True,
                dropout_p=self.dropout_p if self.training else 0.0,
            )                                                           # (B, H, T, Dh)
        else:
            # Step 3: scaled dot-product scores.
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, T, T)
            # Step 4: causal mask. Forbidden positions -> -inf.
            allowed = self.causal_mask[:, :, :seq_len, :seq_len]        # (1, 1, T, T)
            scores = scores.masked_fill(~allowed, float("-inf"))
            # Step 5: each row becomes a probability distribution over positions.
            weights = F.softmax(scores, dim=-1)                         # (B, H, T, T)
            # Step 6: weighted sum of values.
            out = self.attn_dropout(weights) @ v                        # (B, H, T, Dh)

        # Step 7: merge heads and project.
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, dim)  # (B, T, D)
        out = self.resid_dropout(self.out_proj(out))                    # (B, T, D)

        if return_attention:
            return out, weights
        return out

    def extra_repr(self) -> str:
        return (
            f"embedding_dim={self.embedding_dim}, num_heads={self.num_heads}, "
            f"head_dim={self.head_dim}, fused={self.fused}"
        )

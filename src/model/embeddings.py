"""Token and positional embeddings: where integers become vectors.

The tokenizer turns text into ids like [261, 268, 262]. An id is just a label;
the number 268 is not "bigger" than 262 in any meaningful sense. The embedding
layer gives every id a learned vector, so the rest of the network can work with
geometry instead of labels.

Both tables start as random noise, N(0, 0.02). This is where "trained from
random initialization" literally begins: before training, token 268 (" color")
and token 404 (" snow") are unrelated random points. Training moves them.
"""

from __future__ import annotations

import torch
from torch import nn


class TokenEmbedding(nn.Module):
    """A lookup table: one learned D-dimensional vector per vocabulary entry.

    INPUT   token_ids  (B, T)     integers in [0, V)
    OUTPUT  vectors    (B, T, D)

    MATH    output[b, t] = weight[token_ids[b, t]]
            It is indexing, not multiplication. Equivalent to multiplying a
            one-hot vector of length V by the (V, D) matrix, without ever
            building the one-hot vector.

    PARAMS  V * D   (small config: 8192 * 384 = 3,145,728)
    """

    def __init__(self, vocab_size: int, embedding_dim: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(vocab_size, embedding_dim))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]                                   # (B, T, D)

    def extra_repr(self) -> str:
        return f"vocab_size={self.vocab_size}, embedding_dim={self.embedding_dim}"


class PositionalEmbedding(nn.Module):
    """A learned vector for each position 0 .. context_length-1.

    WHY IT IS NEEDED
        Attention compares every token with every other token, but the
        comparison itself ignores order: shuffle the input and each token gets
        the same attention scores, just shuffled. Without position information,
        "dog bites man" and "man bites dog" are indistinguishable. Adding a
        position-specific vector to each token breaks that symmetry.

    INPUT   seq_len    int, T <= context_length
    OUTPUT  vectors    (T, D)   broadcast-added to every sequence in the batch

    PARAMS  context_length * D  (small config: 512 * 384 = 196,608)

    LIMITATION
        A learned table has no row for position 512, so the model cannot read
        beyond its trained context length. Rotary embeddings (RoPE) avoid this
        and are the natural upgrade if longer context ever matters.
    """

    def __init__(self, context_length: int, embedding_dim: int) -> None:
        super().__init__()
        self.context_length = context_length
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(context_length, embedding_dim))

    def forward(self, seq_len: int) -> torch.Tensor:
        if seq_len > self.context_length:
            raise ValueError(
                f"sequence length {seq_len} exceeds the model's context length "
                f"{self.context_length}; there is no learned position beyond that"
            )
        return self.weight[:seq_len]                                    # (T, D)

    def extra_repr(self) -> str:
        return f"context_length={self.context_length}, embedding_dim={self.embedding_dim}"


class Embeddings(nn.Module):
    """Token embedding + positional embedding.

    INPUT   token_ids  (B, T)
    OUTPUT  x          (B, T, D)   the input to the first transformer block
    """

    def __init__(
        self, vocab_size: int, context_length: int, embedding_dim: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.token = TokenEmbedding(vocab_size, embedding_dim)
        self.position = PositionalEmbedding(context_length, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        _, seq_len = token_ids.shape
        tokens = self.token(token_ids)                                  # (B, T, D)
        positions = self.position(seq_len)                              # (T, D)
        return self.dropout(tokens + positions)                         # (B, T, D)

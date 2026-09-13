"""BingBongLM: the complete language model.

    token ids (B, T)
        -> Embeddings           (B, T, D)
        -> Transformer          (B, T, D)
        -> language-model head  (B, T, V)   logits
        -> cross-entropy against the next tokens  -> scalar loss

WHAT THE OUTPUT MEANS
    logits[b, t] is a vector of V raw scores: the model's opinion about which
    token comes at position t+1, having seen tokens 0..t. softmax turns those
    scores into a probability distribution over the whole vocabulary.

    Every position predicts its own next token, all in one forward pass. A
    single 512-token training sequence therefore provides 512 prediction
    problems, not one. The causal mask in attention is what keeps those 512
    predictions honest.

THE LOSS
    Cross-entropy: -log(probability the model assigned to the correct token),
    averaged over every position in the batch.

        Assign the right token probability 1.0  -> loss 0
        Assign it probability 1/V (pure guess)  -> loss ln(V)

    For V = 8192, ln(V) = 9.01. An untrained model should start very close to
    that. If step-0 loss is far from ln(V), initialization is broken -- and
    tests/test_model.py checks exactly this.

WEIGHT TYING
    The head maps D -> V and needs a (V, D) matrix. The token embedding table is
    already a (V, D) matrix. With `tie_embeddings`, the head REUSES it:

        logits = h @ E^T

    so the score for token j is the dot product between the model's hidden
    state and token j's own embedding. It saves V * D parameters (3.1M, or 22%
    of the small model) and generally trains at least as well: the vector that
    represents a token is a sensible vector to score it with.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .attention import CausalSelfAttention
from .embeddings import Embeddings
from .feed_forward import FeedForward
from .normalization import LayerNorm
from .transformer import Transformer

# Standard deviation for weight initialization. Small enough that the stack
# starts near the identity, large enough that symmetry between units is broken.
INIT_STD = 0.02


@dataclass(frozen=True)
class ModelConfig:
    """Architecture hyperparameters. Built from YAML via `from_config`."""

    vocab_size: int
    context_length: int
    embedding_dim: int
    num_layers: int
    num_heads: int
    ffn_multiplier: int = 4
    dropout: float = 0.0
    tie_embeddings: bool = True
    fused_kernels: bool = True

    @classmethod
    def from_config(cls, config) -> "ModelConfig":
        """Build from a loaded `src.utils.config.Config`."""
        model = config.model
        positional = model.get("positional", "learned")
        if positional != "learned":
            raise ValueError(f"unsupported positional encoding {positional!r}; only 'learned'")
        return cls(
            vocab_size=config.tokenizer.vocab_size,
            context_length=model.context_length,
            embedding_dim=model.embedding_dim,
            num_layers=model.num_layers,
            num_heads=model.num_heads,
            ffn_multiplier=model.get("ffn_multiplier", 4),
            dropout=model.get("dropout", 0.0),
            tie_embeddings=model.get("tie_embeddings", True),
            fused_kernels=model.get("fused_kernels", True),
        )

    def to_dict(self) -> dict:
        return asdict(self)


def expected_parameter_count(cfg: ModelConfig) -> dict[str, int]:
    """The parameter count, derived by hand from the architecture.

    This is kept deliberately separate from `sum(p.numel())`. The two are
    compared in tests: if they ever disagree, either the documentation or the
    implementation is wrong, and we want to find out which.
    """
    V, T, D, L = cfg.vocab_size, cfg.context_length, cfg.embedding_dim, cfg.num_layers
    hidden = cfg.ffn_multiplier * D

    attention = (D * 3 * D + 3 * D) + (D * D + D)              # qkv_proj + out_proj
    feed_forward = (D * hidden + hidden) + (hidden * D + D)    # expand + contract
    norms = 2 * (2 * D)                                        # two LayerNorms, gamma + beta
    block = attention + feed_forward + norms

    counts = {
        "token_embedding": V * D,
        "positional_embedding": T * D,
        "transformer_blocks": L * block,
        "final_norm": 2 * D,
        "lm_head": 0 if cfg.tie_embeddings else V * D,
    }
    counts["total"] = sum(counts.values())
    return counts


class BingBongLM(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.embeddings = Embeddings(
            cfg.vocab_size, cfg.context_length, cfg.embedding_dim, dropout=cfg.dropout
        )
        self.transformer = Transformer(
            num_layers=cfg.num_layers,
            embedding_dim=cfg.embedding_dim,
            num_heads=cfg.num_heads,
            context_length=cfg.context_length,
            ffn_multiplier=cfg.ffn_multiplier,
            dropout=cfg.dropout,
            fused=cfg.fused_kernels,
        )
        # Untied head only. When tied, forward() reuses the embedding matrix.
        self.lm_head = (
            None if cfg.tie_embeddings
            else nn.Linear(cfg.embedding_dim, cfg.vocab_size, bias=False)
        )

        self.reset_parameters()

    # -- initialization ---------------------------------------------------

    def reset_parameters(self) -> None:
        """Random initialization. Every weight in BingBongAI starts here.

        - Linear weights and both embedding tables: N(0, 0.02)
        - Biases: 0
        - LayerNorm: gamma = 1, beta = 0 (starts as a pure normalizer)
        - The two projections that WRITE INTO the residual stream (attention
          out_proj, feed-forward contract) get their std scaled down by
          1 / sqrt(2 * num_layers).

        Why that last rule: each block adds two contributions to the residual
        stream. With 2L random additions, the stream's variance grows with depth.
        Shrinking those final projections keeps the stream's scale roughly
        independent of how many layers are stacked.
        """
        residual_std = INIT_STD / math.sqrt(2 * self.cfg.num_layers)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=INIT_STD)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        nn.init.normal_(self.embeddings.token.weight, mean=0.0, std=INIT_STD)
        nn.init.normal_(self.embeddings.position.weight, mean=0.0, std=INIT_STD)

        for module in self.modules():
            if isinstance(module, CausalSelfAttention):
                nn.init.normal_(module.out_proj.weight, mean=0.0, std=residual_std)
            elif isinstance(module, FeedForward):
                nn.init.normal_(module.contract.weight, mean=0.0, std=residual_std)

    # -- forward ----------------------------------------------------------

    def forward(
        self, token_ids: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            token_ids: (B, T) int64, values in [0, vocab_size)
            targets:   (B, T) int64, the token that FOLLOWS each input position.
                       Positions set to -100 are ignored by the loss (padding).

        Returns:
            logits: (B, T, V)
            loss:   scalar tensor, or None if no targets were given
        """
        if token_ids.dim() != 2:
            raise ValueError(f"token_ids must be (B, T), got shape {tuple(token_ids.shape)}")

        x = self.embeddings(token_ids)                                  # (B, T, D)
        h = self.transformer(x)                                         # (B, T, D)

        if self.lm_head is None:
            logits = F.linear(h, self.embeddings.token.weight)          # (B, T, V)
        else:
            logits = self.lm_head(h)                                    # (B, T, V)

        loss = None
        if targets is not None:
            if targets.shape != token_ids.shape:
                raise ValueError(
                    f"targets shape {tuple(targets.shape)} must match "
                    f"token_ids shape {tuple(token_ids.shape)}"
                )
            batch, seq_len, vocab = logits.shape
            loss = F.cross_entropy(
                logits.reshape(batch * seq_len, vocab),                 # (B*T, V)
                targets.reshape(batch * seq_len),                       # (B*T,)
                ignore_index=-100,
            )
        return logits, loss

    # -- introspection ----------------------------------------------------

    def num_parameters(self) -> int:
        """Distinct trainable parameters. A tied matrix is counted once."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_breakdown(self) -> dict[str, int]:
        """Measured parameter counts per component, in the same layout as
        `expected_parameter_count`, so the two can be compared directly."""
        counts = {
            "token_embedding": self.embeddings.token.weight.numel(),
            "positional_embedding": self.embeddings.position.weight.numel(),
            "transformer_blocks": sum(p.numel() for p in self.transformer.blocks.parameters()),
            "final_norm": sum(p.numel() for p in self.transformer.final_norm.parameters()),
            "lm_head": 0 if self.lm_head is None else self.lm_head.weight.numel(),
        }
        counts["total"] = self.num_parameters()
        return counts

    @classmethod
    def from_config(cls, config) -> "BingBongLM":
        return cls(ModelConfig.from_config(config))

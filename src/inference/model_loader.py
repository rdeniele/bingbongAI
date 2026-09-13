"""Load a trained BingBongAI model for inference.

A checkpoint describes itself: it stores the architecture (`model_config`), the
full training config (which records where the tokenizer lives), and the
tokenizer's fingerprint. So generating needs only the checkpoint path -- the
YAML config is not required, and cannot drift out of sync with the weights.

The optimizer state in the checkpoint (two-thirds of its 168 MB) is loaded from
disk but discarded; inference only needs the weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from ..model.language_model import BingBongLM, ModelConfig
from ..tokenizer.tokenizer import BPETokenizer
from ..training.checkpoint import CheckpointError, load_checkpoint


@dataclass
class LoadedModel:
    model: BingBongLM
    tokenizer: BPETokenizer
    checkpoint_path: Path
    step: int
    best_val_loss: float | None
    config_name: str


def load_for_inference(
    checkpoint_path: str | Path,
    device: torch.device,
    tokenizer_path: str | Path | None = None,
    fused_kernels: bool = True,
) -> LoadedModel:
    """Build the model a checkpoint describes, load its weights, and pair it
    with the tokenizer it was trained with.

    Args:
        checkpoint_path: a .pt written by the trainer.
        device: where to put the model.
        tokenizer_path: override the tokenizer location recorded in the checkpoint.
        fused_kernels: implementation choice only; weights are identical either way.

    Raises:
        CheckpointError: unreadable checkpoint, or a tokenizer whose fingerprint
            differs from the one the weights were trained with.
    """
    checkpoint_path = Path(checkpoint_path)
    payload = load_checkpoint(checkpoint_path, map_location="cpu")

    model_config_dict = dict(payload.get("model_config") or {})
    if not model_config_dict:
        raise CheckpointError(f"{checkpoint_path} has no model_config; cannot rebuild the model")
    model_config_dict["fused_kernels"] = fused_kernels
    # Dropout is a training-time regulariser; at inference eval() disables it anyway.
    cfg = ModelConfig(**model_config_dict)

    training_config = payload.get("config") or {}
    if tokenizer_path is None:
        tokenizer_path = training_config.get("tokenizer", {}).get("vocab_path")
        if tokenizer_path is None:
            raise CheckpointError(
                f"{checkpoint_path} does not record a tokenizer path; pass tokenizer_path"
            )
    tokenizer = BPETokenizer.load(tokenizer_path)

    expected = payload.get("tokenizer_fingerprint")
    if expected != tokenizer.fingerprint():
        raise CheckpointError(
            f"tokenizer at {tokenizer_path} has fingerprint {tokenizer.fingerprint()}, but "
            f"{checkpoint_path} was trained with {expected}. The model would read every "
            f"token as the wrong one."
        )
    if tokenizer.vocab_size > cfg.vocab_size:
        raise CheckpointError(
            f"tokenizer has {tokenizer.vocab_size} tokens but the model only {cfg.vocab_size}"
        )

    model = BingBongLM(cfg)
    model.load_state_dict(payload["model"])
    model.to(device).eval()

    best = payload.get("best_val_loss")
    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        checkpoint_path=checkpoint_path,
        step=int(payload.get("step", 0)),
        best_val_loss=float(best) if best is not None and best != float("inf") else None,
        config_name=str(training_config.get("name", "unknown")),
    )

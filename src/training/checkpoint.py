"""Saving and loading training state.

A checkpoint holds everything needed to continue training as if it had never
stopped -- not just the weights:

    model           the weights
    optimizer       AdamW's per-parameter moment estimates. Resuming without
                    them restarts every running average from zero, which shows
                    up as a visible loss spike.
    scaler          fp16 loss-scaler state (empty for fp32)
    step            completed optimizer steps; the LR schedule continues from here
    best_val_loss   so `best.pt` is only replaced by a genuinely better model
    loader          the batch sampler's RNG state, so the resumed run draws the
                    same batches an uninterrupted run would have
    rng             torch (CPU and CUDA) and Python RNG states
    config          the full YAML config as a dict -- a checkpoint describes itself
    model_config    the exact architecture, checked on load
    tokenizer_fingerprint
                    loading weights against a different vocabulary would make
                    every embedding row mean the wrong token, so this is checked

SAFETY
    Checkpoints are loaded with `weights_only=True`. A .pt file is a pickle, and
    unrestricted unpickling can execute arbitrary code; weights_only limits
    loading to tensors and plain containers. Only built-in types are stored here
    for exactly that reason.

ATOMICITY
    Writes go to a temporary file first, then are renamed over the target. A
    crash or power cut mid-save leaves the previous checkpoint intact instead of
    a truncated, unloadable file.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import torch

CHECKPOINT_FORMAT_VERSION = 1


class CheckpointError(RuntimeError):
    """A checkpoint is unreadable or incompatible with the current setup."""


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    # RNG states are CPU ByteTensors by definition, even the CUDA ones.
    torch.set_rng_state(state["torch"].cpu())
    random.setstate(_as_python_random_state(state["python"]))
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def _as_python_random_state(state: Any) -> tuple:
    """random.setstate needs tuples; serialisation may hand back lists."""
    version, internal, gauss = state
    return (version, tuple(internal), gauss)


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> Path:
    """Atomically write a checkpoint dict to `path`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"format_version": CHECKPOINT_FORMAT_VERSION,
               "torch_version": str(torch.__version__), **payload}

    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return path


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise CheckpointError(f"checkpoint not found: {path}")

    try:
        payload = torch.load(path, map_location=map_location, weights_only=True)
    except Exception as error:  # corrupted, truncated, or not a checkpoint at all
        raise CheckpointError(f"could not read checkpoint {path}: {error}") from error

    version = payload.get("format_version")
    if version != CHECKPOINT_FORMAT_VERSION:
        raise CheckpointError(
            f"checkpoint {path} has format version {version}; "
            f"this code reads version {CHECKPOINT_FORMAT_VERSION}"
        )
    return payload


def check_compatibility(
    payload: dict[str, Any], model_config: dict[str, Any], tokenizer_fingerprint: str
) -> None:
    """Refuse to load a checkpoint into a mismatched model or tokenizer.

    `fused_kernels` is allowed to differ: it selects an implementation, not an
    architecture, and both implementations share identical weights.
    """
    saved_model = dict(payload["model_config"])
    current_model = dict(model_config)
    saved_model.pop("fused_kernels", None)
    current_model.pop("fused_kernels", None)

    if saved_model != current_model:
        differences = {
            key: (saved_model.get(key), current_model.get(key))
            for key in sorted(set(saved_model) | set(current_model))
            if saved_model.get(key) != current_model.get(key)
        }
        raise CheckpointError(
            "checkpoint architecture does not match the current config "
            f"(checkpoint, current): {differences}"
        )

    saved_fingerprint = payload.get("tokenizer_fingerprint")
    if saved_fingerprint != tokenizer_fingerprint:
        raise CheckpointError(
            f"checkpoint was trained with tokenizer {saved_fingerprint}, but the "
            f"current tokenizer is {tokenizer_fingerprint}. The embedding rows would "
            f"correspond to the wrong tokens."
        )

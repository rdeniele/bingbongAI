"""Shared test fixtures: a tiny but complete training setup that runs on CPU in seconds."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model.language_model import BingBongLM, ModelConfig  # noqa: E402
from src.training.dataloader import BatchLoader  # noqa: E402
from src.training.dataset import TokenDataset, write_token_file  # noqa: E402
from src.training.trainer import Trainer, TrainingSettings  # noqa: E402
from src.utils.logging import RunLogger  # noqa: E402

CPU = torch.device("cpu")
TINY_MODEL = ModelConfig(vocab_size=64, context_length=16, embedding_dim=32,
                         num_layers=2, num_heads=4, dropout=0.0)


def repeating_token_file(path: Path, pattern: list[int], repeats: int) -> Path:
    """A token stream that is one short pattern over and over -- trivially learnable."""
    write_token_file(path, pattern * repeats)
    return path


def make_trainer(
    tmp_path: Path,
    *,
    max_steps: int = 50,
    learning_rate: float = 3e-3,
    checkpoint_dir: Path | None = None,
    seed: int = 0,
    model_cfg: ModelConfig = TINY_MODEL,
    fingerprint: str = "test-tokenizer",
    device: torch.device = CPU,
) -> Trainer:
    pattern = [3, 14, 15, 9, 26, 5, 35, 8, 9, 7, 9, 32, 38, 4, 6, 2, 43, 38, 32, 7]
    train_path = repeating_token_file(tmp_path / "train.bin", pattern, repeats=40)
    val_path = repeating_token_file(tmp_path / "val.bin", pattern, repeats=5)

    settings = TrainingSettings(
        batch_size=8, learning_rate=learning_rate, min_learning_rate=learning_rate / 10,
        max_steps=max_steps, warmup_steps=5, eval_interval=10, eval_batches=2,
        checkpoint_interval=10, keep_last=2, log_interval=10, seed=seed,
        weight_decay=0.0,
    )
    torch.manual_seed(seed)
    model = BingBongLM(model_cfg).to(device)
    T = model_cfg.context_length
    return Trainer(
        model, settings,
        BatchLoader(TokenDataset(train_path, T), settings.batch_size, device, seed=seed),
        BatchLoader(TokenDataset(val_path, T), settings.batch_size, device, seed=seed + 1),
        device,
        config_dict={"name": "test"},
        tokenizer_fingerprint=fingerprint,
        checkpoint_dir=checkpoint_dir,
        logger=RunLogger(None, quiet=True),
    )

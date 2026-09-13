"""Tests for checkpointing: save -> load -> continue.

The strongest check here is `test_resumed_run_matches_uninterrupted_run`: stop
training halfway, restore from disk into a brand-new trainer, finish -- and the
loss at every later step must equal a run that never stopped. That can only pass
if weights, optimizer moments, the step counter, the LR schedule position and
the batch sampler's RNG are ALL restored correctly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model.language_model import ModelConfig  # noqa: E402
from src.training.checkpoint import (  # noqa: E402
    CheckpointError,
    check_compatibility,
    load_checkpoint,
    save_checkpoint,
)
from tests.helpers import TINY_MODEL, make_trainer  # noqa: E402


def test_resumed_run_matches_uninterrupted_run(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    continuous = make_trainer(tmp_path / "a", max_steps=30)
    continuous.fit()

    (tmp_path / "b").mkdir()
    first_half = make_trainer(tmp_path / "b", max_steps=30, checkpoint_dir=tmp_path / "ckpt")
    first_half.fit(max_steps=15)
    first_half.save("halfway.pt")

    # A completely fresh trainer, deliberately built with a DIFFERENT seed, so
    # nothing can match unless the checkpoint restores it.
    resumed = make_trainer(tmp_path / "b", max_steps=30, checkpoint_dir=tmp_path / "ckpt", seed=999)
    resumed.resume(tmp_path / "ckpt" / "halfway.pt")
    assert resumed.step == 15
    resumed.fit()

    expected = dict(continuous.loss_history)
    for step, loss in resumed.loss_history:
        assert loss == pytest.approx(expected[step], abs=1e-6), f"loss differs at step {step}"
    for (name, a), (_, b) in zip(continuous.model.named_parameters(), resumed.model.named_parameters()):
        assert torch.allclose(a, b, atol=1e-6), name


def test_checkpoint_round_trip_restores_weights_and_counters(tmp_path: Path) -> None:
    (tmp_path / "d").mkdir()
    trainer = make_trainer(tmp_path / "d", max_steps=12, checkpoint_dir=tmp_path / "ckpt")
    trainer.fit()
    path = trainer.save("manual.pt")

    fresh = make_trainer(tmp_path / "d", max_steps=12, checkpoint_dir=tmp_path / "ckpt", seed=5)
    fresh.resume(path)
    assert fresh.step == 12
    assert fresh.best_val_loss == trainer.best_val_loss
    for key, value in trainer.model.state_dict().items():
        assert torch.equal(value, fresh.model.state_dict()[key]), key


def test_fit_writes_latest_best_and_prunes_step_files(tmp_path: Path) -> None:
    (tmp_path / "d").mkdir()
    ckpt = tmp_path / "ckpt"
    make_trainer(tmp_path / "d", max_steps=50, checkpoint_dir=ckpt).fit()
    names = sorted(p.name for p in ckpt.iterdir())
    assert "latest.pt" in names and "best.pt" in names
    # keep_last=2: only the two most recent periodic checkpoints survive.
    assert [n for n in names if n.startswith("step_")] == ["step_000040.pt", "step_000050.pt"]
    assert not any(n.endswith(".tmp") for n in names)


def test_checkpoint_contains_everything_needed_to_resume(tmp_path: Path) -> None:
    (tmp_path / "d").mkdir()
    trainer = make_trainer(tmp_path / "d", max_steps=3, checkpoint_dir=tmp_path / "ckpt")
    trainer.fit()
    payload = load_checkpoint(tmp_path / "ckpt" / "latest.pt")
    for key in ("model", "optimizer", "scaler", "step", "best_val_loss", "loader",
                "rng", "config", "model_config", "tokenizer_fingerprint", "format_version"):
        assert key in payload, key


def test_architecture_mismatch_is_refused(tmp_path: Path) -> None:
    (tmp_path / "d").mkdir()
    trainer = make_trainer(tmp_path / "d", max_steps=2, checkpoint_dir=tmp_path / "ckpt")
    trainer.fit()
    other_cfg = ModelConfig(**{**TINY_MODEL.to_dict(), "num_layers": 3})
    other = make_trainer(tmp_path / "d", checkpoint_dir=tmp_path / "ckpt", model_cfg=other_cfg)
    with pytest.raises(CheckpointError, match="architecture does not match"):
        other.resume(tmp_path / "ckpt" / "latest.pt")


def test_tokenizer_mismatch_is_refused(tmp_path: Path) -> None:
    (tmp_path / "d").mkdir()
    trainer = make_trainer(tmp_path / "d", max_steps=2, checkpoint_dir=tmp_path / "ckpt")
    trainer.fit()
    other = make_trainer(tmp_path / "d", checkpoint_dir=tmp_path / "ckpt", fingerprint="different")
    with pytest.raises(CheckpointError, match="tokenizer"):
        other.resume(tmp_path / "ckpt" / "latest.pt")


def test_kernel_choice_may_differ_on_resume() -> None:
    """fused vs manual is an implementation detail with identical weights."""
    saved = {"model_config": {**TINY_MODEL.to_dict(), "fused_kernels": True},
             "tokenizer_fingerprint": "t"}
    check_compatibility(saved, {**TINY_MODEL.to_dict(), "fused_kernels": False}, "t")


def test_corrupt_checkpoint_is_a_clear_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"this is not a checkpoint")
    with pytest.raises(CheckpointError, match="could not read"):
        load_checkpoint(bad)


def test_missing_checkpoint_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(CheckpointError, match="not found"):
        load_checkpoint(tmp_path / "absent.pt")


def test_wrong_format_version_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "old.pt"
    save_checkpoint(path, {})
    payload = torch.load(path, weights_only=True)
    payload["format_version"] = 0
    torch.save(payload, path)
    with pytest.raises(CheckpointError, match="format version"):
        load_checkpoint(path)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_resume_on_cuda(tmp_path: Path) -> None:
    """Regression: resuming on GPU once crashed with 'RNG state must be a torch.ByteTensor',
    because the checkpoint was loaded straight onto CUDA, RNG state included.
    """
    cuda = torch.device("cuda")
    (tmp_path / "d").mkdir()
    trainer = make_trainer(tmp_path / "d", max_steps=4, checkpoint_dir=tmp_path / "ckpt", device=cuda)
    trainer.fit()

    resumed = make_trainer(tmp_path / "d", max_steps=8, checkpoint_dir=tmp_path / "ckpt",
                           device=cuda, seed=3)
    resumed.resume(tmp_path / "ckpt" / "latest.pt")
    assert resumed.step == 4
    assert all(p.device.type == "cuda" for p in resumed.model.parameters())
    resumed.fit()
    assert resumed.step == 8

"""Tests for the training loop.

`test_model_can_overfit_a_tiny_dataset` is the most important test in the
project. If the model cannot memorise a 20-token repeating pattern, then
something between the loss and the weights is broken, and every result from a
real training run would be meaningless.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.trainer import (  # noqa: E402
    TrainingSettings,
    build_optimizer,
    learning_rate_at,
)
from tests.helpers import CPU, TINY_MODEL, make_trainer  # noqa: E402

from src.model.language_model import BingBongLM  # noqa: E402


# -- learning -----------------------------------------------------------------


def test_model_can_overfit_a_tiny_dataset(tmp_path: Path) -> None:
    """Random weights -> training -> loss collapses on a pattern it can memorise.

    The data is a fixed 20-token pattern repeated, so after seeing a few tokens
    the next token is fully determined. A working model must drive loss far
    below the ln(64) = 4.16 it starts at.
    """
    trainer = make_trainer(tmp_path, max_steps=150)
    initial_val = trainer.evaluate()
    summary = trainer.fit()

    assert abs(initial_val - math.log(TINY_MODEL.vocab_size)) < 0.5
    assert summary["final_val_loss"] < 0.1, summary
    assert summary["final_val_loss"] < initial_val / 20


def test_loss_decreases_during_training(tmp_path: Path) -> None:
    trainer = make_trainer(tmp_path, max_steps=60)
    trainer.fit()
    losses = [loss for _, loss in trainer.loss_history]
    first, last = sum(losses[:5]) / 5, sum(losses[-5:]) / 5
    assert last < first * 0.5


def test_training_changes_the_weights(tmp_path: Path) -> None:
    trainer = make_trainer(tmp_path, max_steps=3)
    before = {k: v.clone() for k, v in trainer.model.state_dict().items()}
    trainer.fit()
    after = trainer.model.state_dict()
    changed = [k for k in before if not torch.equal(before[k], after[k])]
    assert len(changed) == len(before), "every parameter tensor should receive an update"


def test_the_untrained_model_cannot_already_do_the_task(tmp_path: Path) -> None:
    """Guards against a vacuous overfit test (e.g. targets leaking into inputs)."""
    trainer = make_trainer(tmp_path, max_steps=1)
    assert trainer.evaluate() > 3.0


def test_step_counter_and_histories(tmp_path: Path) -> None:
    trainer = make_trainer(tmp_path, max_steps=25)
    trainer.fit()
    assert trainer.step == 25
    assert [s for s, _ in trainer.loss_history] == list(range(1, 26))
    assert [s for s, _ in trainer.eval_history] == [0, 10, 20, 25]


def test_divergence_is_detected(tmp_path: Path) -> None:
    trainer = make_trainer(tmp_path, max_steps=5)
    with torch.no_grad():
        trainer.model.embeddings.token.weight.fill_(float("nan"))
    with pytest.raises(FloatingPointError, match="diverged"):
        trainer.fit()


def test_gradient_accumulation_matches_a_larger_batch(tmp_path: Path) -> None:
    """2 micro-batches of 4 must produce the same update as 1 batch of 8.

    Both draw 8 windows from identically seeded generators; accumulation only
    changes how they are grouped. Correct loss scaling makes the gradients equal.
    """
    big = make_trainer(tmp_path, max_steps=1)
    small = make_trainer(tmp_path, max_steps=1)
    small.settings = TrainingSettings(**{**small.settings.__dict__, "batch_size": 4,
                                         "gradient_accumulation_steps": 2})
    small.train_loader.batch_size = 4

    # Same starting weights and the same 8 windows in the same order.
    small.model.load_state_dict(big.model.state_dict())
    starts = torch.randint(0, len(big.train_loader.dataset), (8,), generator=torch.Generator().manual_seed(9))
    big_windows = iter([big.train_loader.dataset.windows(starts)])
    small_windows = iter([big.train_loader.dataset.windows(starts[:4]),
                          big.train_loader.dataset.windows(starts[4:])])
    big.train_loader.next_batch = lambda: next(big_windows)
    small.train_loader.next_batch = lambda: next(small_windows)

    big.train_step()
    small.train_step()
    for (name, a), (_, b) in zip(big.model.named_parameters(), small.model.named_parameters()):
        assert torch.allclose(a, b, atol=1e-6), name


# -- schedule and optimizer ---------------------------------------------------------


def _settings(**overrides) -> TrainingSettings:
    base = dict(batch_size=1, learning_rate=1e-3, min_learning_rate=1e-4,
                max_steps=100, warmup_steps=10)
    return TrainingSettings(**{**base, **overrides})


def test_warmup_rises_linearly_to_peak() -> None:
    s = _settings()
    assert learning_rate_at(0, s) == pytest.approx(1e-4)
    assert learning_rate_at(4, s) == pytest.approx(5e-4)
    assert learning_rate_at(9, s) == pytest.approx(1e-3)


def test_cosine_decays_to_minimum() -> None:
    s = _settings()
    assert learning_rate_at(10, s) == pytest.approx(1e-3)
    midpoint = learning_rate_at(55, s)
    assert midpoint == pytest.approx((1e-3 + 1e-4) / 2, rel=1e-6)
    assert learning_rate_at(100, s) == pytest.approx(1e-4)
    assert learning_rate_at(5000, s) == pytest.approx(1e-4)


def test_schedule_is_monotone_after_warmup() -> None:
    s = _settings()
    values = [learning_rate_at(step, s) for step in range(10, 101)]
    assert all(a >= b for a, b in zip(values, values[1:]))


def test_weight_decay_applies_to_matrices_only() -> None:
    model = BingBongLM(TINY_MODEL)
    optimizer = build_optimizer(model, _settings(weight_decay=0.1), CPU)
    decay_group, no_decay_group = optimizer.param_groups
    assert decay_group["weight_decay"] == 0.1 and no_decay_group["weight_decay"] == 0.0
    assert all(p.dim() >= 2 for p in decay_group["params"])
    assert all(p.dim() == 1 for p in no_decay_group["params"])
    total = sum(p.numel() for g in optimizer.param_groups for p in g["params"])
    assert total == model.num_parameters()


def test_invalid_precision_is_rejected() -> None:
    with pytest.raises(ValueError, match="precision"):
        _settings(precision="int4")


def test_fp16_on_cpu_is_rejected(tmp_path: Path) -> None:
    trainer = make_trainer(tmp_path)
    with pytest.raises(ValueError, match="fp16"):
        type(trainer)(trainer.model, TrainingSettings(**{**trainer.settings.__dict__, "precision": "fp16"}),
                      trainer.train_loader, trainer.val_loader, CPU,
                      config_dict={}, tokenizer_fingerprint="x", checkpoint_dir=None)

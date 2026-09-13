"""Tests for causal self-attention.

The most important test in this file is `test_no_future_token_leakage`. A leak
through the causal mask is the classic silent bug in a language model: training
loss drops beautifully (the model just copies the next token) and generation is
garbage. Nothing crashes. Only a test like this catches it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model.attention import CausalSelfAttention  # noqa: E402

B, T, D, H = 2, 12, 48, 4


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(0)


def make_attention(fused: bool) -> CausalSelfAttention:
    attention = CausalSelfAttention(D, H, context_length=32, dropout=0.0, fused=fused)
    return attention.eval()


# -- shapes ----------------------------------------------------------------


@pytest.mark.parametrize("fused", [False, True])
def test_output_shape_matches_input(fused: bool) -> None:
    out = make_attention(fused)(torch.randn(B, T, D))
    assert out.shape == (B, T, D)


def test_attention_weights_shape() -> None:
    _, weights = make_attention(False)(torch.randn(B, T, D), return_attention=True)
    assert weights.shape == (B, H, T, T)


def test_head_dim() -> None:
    assert make_attention(False).head_dim == D // H


def test_indivisible_heads_are_rejected() -> None:
    with pytest.raises(ValueError, match="divisible"):
        CausalSelfAttention(50, 4, context_length=16)


def test_sequence_longer_than_context_is_rejected() -> None:
    attention = CausalSelfAttention(D, H, context_length=8)
    with pytest.raises(ValueError, match="exceeds context length"):
        attention(torch.randn(1, 9, D))


def test_parameter_count() -> None:
    """4D^2 + 4D, as documented."""
    attention = make_attention(False)
    assert sum(p.numel() for p in attention.parameters()) == 4 * D * D + 4 * D


# -- the causal mask -------------------------------------------------------


def test_weights_above_the_diagonal_are_exactly_zero() -> None:
    _, weights = make_attention(False)(torch.randn(B, T, D), return_attention=True)
    upper = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    assert torch.all(weights[..., upper] == 0.0)


def test_each_row_of_weights_sums_to_one() -> None:
    _, weights = make_attention(False)(torch.randn(B, T, D), return_attention=True)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(B, H, T), atol=1e-6)


def test_first_position_attends_only_to_itself() -> None:
    _, weights = make_attention(False)(torch.randn(B, T, D), return_attention=True)
    assert torch.allclose(weights[..., 0, 0], torch.ones(B, H))


@pytest.mark.parametrize("fused", [False, True])
def test_no_future_token_leakage(fused: bool) -> None:
    """Changing tokens at positions >= j must not change outputs at positions < j.

    Checked for every split point, on both the manual and the fused kernel.
    """
    attention = make_attention(fused)
    x = torch.randn(1, T, D)
    baseline = attention(x)

    for split in range(1, T):
        perturbed = x.clone()
        perturbed[:, split:] = torch.randn(1, T - split, D) * 10.0
        out = attention(perturbed)
        assert torch.equal(out[:, :split], baseline[:, :split]) or torch.allclose(
            out[:, :split], baseline[:, :split], atol=1e-6
        ), f"output before position {split} changed when only the future changed"
        # And the future positions really did change -- the test is not vacuous.
        assert not torch.allclose(out[:, split:], baseline[:, split:])


@pytest.mark.parametrize("fused", [False, True])
def test_gradient_does_not_flow_from_past_outputs_to_future_inputs(fused: bool) -> None:
    """The same property, stated through autograd: d(out[t]) / d(x[s]) == 0 for s > t."""
    attention = make_attention(fused)
    x = torch.randn(1, T, D, requires_grad=True)
    out = attention(x)

    t = 4
    out[0, t].sum().backward()
    grad_per_position = x.grad[0].abs().sum(dim=-1)                  # (T,)
    assert torch.all(grad_per_position[t + 1:] == 0.0)
    assert torch.all(grad_per_position[: t + 1] > 0.0)


# -- manual vs fused -------------------------------------------------------


def test_fused_kernel_matches_manual_math() -> None:
    """The fused path is only allowed to be faster, never different."""
    manual = make_attention(False)
    fused = make_attention(True)
    fused.load_state_dict(manual.state_dict())

    x = torch.randn(B, T, D)
    assert torch.allclose(manual(x), fused(x), atol=1e-5)


def test_fused_kernel_gradients_match_manual() -> None:
    manual = make_attention(False)
    fused = make_attention(True)
    fused.load_state_dict(manual.state_dict())

    x1 = torch.randn(B, T, D, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    manual(x1).pow(2).sum().backward()
    fused(x2).pow(2).sum().backward()

    assert torch.allclose(x1.grad, x2.grad, atol=1e-4)
    for (name, p1), (_, p2) in zip(manual.named_parameters(), fused.named_parameters()):
        assert torch.allclose(p1.grad, p2.grad, atol=1e-4), f"gradient mismatch in {name}"


def test_mask_buffer_is_not_saved_in_state_dict() -> None:
    """Derived from config, so it must not bloat or desync checkpoints."""
    assert "causal_mask" not in make_attention(False).state_dict()

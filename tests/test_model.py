"""Tests for the full language model and its components."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model.embeddings import Embeddings  # noqa: E402
from src.model.feed_forward import FeedForward  # noqa: E402
from src.model.language_model import (  # noqa: E402
    BingBongLM,
    ModelConfig,
    expected_parameter_count,
)
from src.model.normalization import LayerNorm  # noqa: E402
from src.model.transformer_block import TransformerBlock  # noqa: E402
from src.utils.config import load_config  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# Small enough to run fast on CPU; same code paths as the real config.
TEST_CFG = ModelConfig(
    vocab_size=300, context_length=32, embedding_dim=48,
    num_layers=2, num_heads=4, dropout=0.0,
)


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(0)


def random_batch(cfg: ModelConfig, batch: int = 2, seq_len: int = 16) -> torch.Tensor:
    return torch.randint(0, cfg.vocab_size, (batch, seq_len))


# -- layer norm -----------------------------------------------------------


def test_layer_norm_output_is_normalized() -> None:
    norm = LayerNorm(64, fused=False)
    y = norm(torch.randn(4, 10, 64) * 7 + 3)
    assert torch.allclose(y.mean(dim=-1), torch.zeros(4, 10), atol=1e-5)
    assert torch.allclose(y.std(dim=-1, unbiased=False), torch.ones(4, 10), atol=1e-3)


def test_layer_norm_manual_matches_fused_including_gradients() -> None:
    norm = LayerNorm(64)
    with torch.no_grad():
        norm.weight.normal_()
        norm.bias.normal_()

    x1 = torch.randn(3, 5, 64, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)

    manual = norm.manual_forward(x1)
    fused = norm(x2)
    assert torch.allclose(manual, fused, atol=1e-5)

    manual.pow(2).sum().backward()
    manual_grads = (x1.grad.clone(), norm.weight.grad.clone(), norm.bias.grad.clone())
    norm.zero_grad()
    fused.pow(2).sum().backward()

    assert torch.allclose(manual_grads[0], x2.grad, atol=1e-4)
    assert torch.allclose(manual_grads[1], norm.weight.grad, atol=1e-4)
    assert torch.allclose(manual_grads[2], norm.bias.grad, atol=1e-4)


# -- components -----------------------------------------------------------


def test_embeddings_shape() -> None:
    emb = Embeddings(vocab_size=300, context_length=32, embedding_dim=48)
    assert emb(torch.randint(0, 300, (2, 16))).shape == (2, 16, 48)


def test_embeddings_reject_sequences_past_context_length() -> None:
    emb = Embeddings(vocab_size=300, context_length=8, embedding_dim=48)
    with pytest.raises(ValueError, match="exceeds the model's context length"):
        emb(torch.randint(0, 300, (1, 9)))


def test_position_changes_the_embedding() -> None:
    """The same token at two positions must get two different vectors."""
    emb = Embeddings(vocab_size=300, context_length=32, embedding_dim=48)
    torch.nn.init.normal_(emb.token.weight)
    torch.nn.init.normal_(emb.position.weight)
    out = emb(torch.full((1, 4), 7))
    assert not torch.allclose(out[0, 0], out[0, 1])


def test_feed_forward_shape_and_parameter_count() -> None:
    ff = FeedForward(48, multiplier=4)
    assert ff(torch.randn(2, 16, 48)).shape == (2, 16, 48)
    assert sum(p.numel() for p in ff.parameters()) == 8 * 48 * 48 + 5 * 48


def test_feed_forward_is_position_wise() -> None:
    """Changing position 3 must not affect the output at any other position."""
    ff = FeedForward(48).eval()
    x = torch.randn(1, 8, 48)
    y = x.clone()
    y[0, 3] = torch.randn(48)
    diff = (ff(x) - ff(y)).abs().sum(dim=-1)[0]
    assert diff[3] > 0
    assert torch.all(diff[torch.arange(8) != 3] == 0)


def test_transformer_block_preserves_shape() -> None:
    block = TransformerBlock(48, 4, context_length=32)
    assert block(torch.randn(2, 16, 48)).shape == (2, 16, 48)


# -- full model: shapes and loss ------------------------------------------


def test_forward_output_dimensions() -> None:
    model = BingBongLM(TEST_CFG)
    logits, loss = model(random_batch(TEST_CFG))
    assert logits.shape == (2, 16, TEST_CFG.vocab_size)
    assert loss is None


def test_loss_is_a_finite_scalar() -> None:
    model = BingBongLM(TEST_CFG)
    ids = random_batch(TEST_CFG)
    _, loss = model(ids, targets=ids)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_initial_loss_is_close_to_uniform_guessing() -> None:
    """Random weights should predict roughly uniformly: loss ~ ln(V).

    If this fails, initialization is broken and training would start from a
    strange place. For V=300, ln(V) = 5.70.
    """
    model = BingBongLM(TEST_CFG)
    ids = torch.randint(0, TEST_CFG.vocab_size, (8, 32))
    targets = torch.randint(0, TEST_CFG.vocab_size, (8, 32))
    _, loss = model(ids, targets=targets)
    assert abs(loss.item() - math.log(TEST_CFG.vocab_size)) < 0.3


def test_ignore_index_excludes_positions_from_loss() -> None:
    model = BingBongLM(TEST_CFG)
    ids = random_batch(TEST_CFG)
    targets = ids.clone()
    _, full = model(ids, targets=targets)
    targets[:, 8:] = -100
    _, partial = model(ids, targets=targets)
    assert not torch.isclose(full, partial)


def test_mismatched_targets_shape_is_rejected() -> None:
    model = BingBongLM(TEST_CFG)
    with pytest.raises(ValueError, match="must match"):
        model(random_batch(TEST_CFG, seq_len=16), targets=random_batch(TEST_CFG, seq_len=15))


def test_model_rejects_too_long_input() -> None:
    model = BingBongLM(TEST_CFG)
    with pytest.raises(ValueError):
        model(random_batch(TEST_CFG, seq_len=TEST_CFG.context_length + 1))


def test_no_future_token_leakage_through_the_whole_model() -> None:
    """End to end: logits at position t depend only on tokens 0..t."""
    model = BingBongLM(TEST_CFG).eval()
    ids = random_batch(TEST_CFG, batch=1, seq_len=20)
    baseline, _ = model(ids)

    split = 11
    changed = ids.clone()
    changed[0, split:] = (changed[0, split:] + 1) % TEST_CFG.vocab_size
    out, _ = model(changed)

    assert torch.allclose(out[0, :split], baseline[0, :split], atol=1e-5)
    assert not torch.allclose(out[0, split:], baseline[0, split:])


def test_backward_reaches_every_parameter() -> None:
    model = BingBongLM(TEST_CFG)
    ids = random_batch(TEST_CFG)
    _, loss = model(ids, targets=ids)
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(param.grad).all(), f"{name} has a non-finite gradient"
        assert param.grad.abs().sum() > 0, f"{name} has an all-zero gradient"


def test_manual_and_fused_models_are_equivalent() -> None:
    manual = BingBongLM(ModelConfig(**{**TEST_CFG.to_dict(), "fused_kernels": False})).eval()
    fused = BingBongLM(ModelConfig(**{**TEST_CFG.to_dict(), "fused_kernels": True})).eval()
    fused.load_state_dict(manual.state_dict())
    ids = random_batch(TEST_CFG)
    assert torch.allclose(manual(ids)[0], fused(ids)[0], atol=1e-4)


def test_same_seed_gives_identical_models() -> None:
    torch.manual_seed(123)
    a = BingBongLM(TEST_CFG)
    torch.manual_seed(123)
    b = BingBongLM(TEST_CFG)
    for (name, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert torch.equal(pa, pb), name


# -- weight tying ---------------------------------------------------------


def test_tied_head_shares_the_embedding_matrix() -> None:
    model = BingBongLM(TEST_CFG)
    assert model.lm_head is None
    ids = random_batch(TEST_CFG)
    _, loss = model(ids, targets=ids)
    loss.backward()
    # The embedding receives gradient from BOTH the input lookup and the output
    # projection -- the second contribution only exists if tying is real.
    assert model.embeddings.token.weight.grad is not None


def test_untied_head_adds_exactly_v_times_d_parameters() -> None:
    tied = BingBongLM(TEST_CFG)
    untied = BingBongLM(ModelConfig(**{**TEST_CFG.to_dict(), "tie_embeddings": False}))
    assert untied.num_parameters() - tied.num_parameters() == TEST_CFG.vocab_size * TEST_CFG.embedding_dim


# -- parameter counts: implementation vs documented arithmetic -------------


def test_measured_breakdown_matches_formula_for_test_config() -> None:
    model = BingBongLM(TEST_CFG)
    assert model.parameter_breakdown() == expected_parameter_count(TEST_CFG)


@pytest.mark.parametrize(
    "config_file, documented_total",
    [
        ("configs/small.yaml", 13_989_888),   # README / ARCHITECTURE.md
        ("configs/tiny.yaml", 4_273_664),     # HARDWARE.md / ARCHITECTURE.md
    ],
)
def test_real_configs_match_the_documented_parameter_counts(
    config_file: str, documented_total: int
) -> None:
    """The numbers written in the docs before any code existed must be true."""
    model = BingBongLM.from_config(load_config(ROOT / config_file))
    assert model.num_parameters() == documented_total
    assert model.parameter_breakdown() == expected_parameter_count(model.cfg)


def test_greedy_generation_never_emits_ids_the_tokenizer_cannot_decode() -> None:
    """Regression: model vocab (8192) can exceed tokenizer vocab (484 for synthetic).

    Found in Phase 7: an untrained model argmax'd id 2777, which crashed decode.
    """
    from src.inference.generate import generate_greedy
    from src.tokenizer.train_tokenizer import train_tokenizer

    tok = train_tokenizer(["the cat sat on the mat. " * 30], vocab_size=280,
                          special_tokens={"pad": "<|pad|>", "unk": "<|unk|>",
                                          "bos": "<|bos|>", "eos": "<|eos|>"}, verbose=False)
    wide = ModelConfig(vocab_size=4096, context_length=32, embedding_dim=48, num_layers=1, num_heads=4)
    model = BingBongLM(wide)
    with torch.no_grad():                      # force the unused rows to win every argmax
        model.embeddings.token.weight[tok.vocab_size:] *= 50.0
    text = generate_greedy(model, tok, "the cat", max_new_tokens=20)
    assert isinstance(text, str)

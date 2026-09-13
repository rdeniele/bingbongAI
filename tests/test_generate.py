"""Tests for text generation: sampling, determinism, stopping, streaming, loading."""

from __future__ import annotations

import math
import sys
from collections import Counter
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.generate import (  # noqa: E402
    GenerationConfig,
    IncrementalDecoder,
    generate,
    generate_greedy,
    sample_next_token,
    stream_tokens,
)
from src.inference.model_loader import load_for_inference  # noqa: E402
from src.model.language_model import BingBongLM, ModelConfig  # noqa: E402
from src.tokenizer.train_tokenizer import train_tokenizer  # noqa: E402
from src.training.checkpoint import CheckpointError, save_checkpoint  # noqa: E402

SPECIALS = {"pad": "<|pad|>", "unk": "<|unk|>", "bos": "<|bos|>", "eos": "<|eos|>"}
CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def tokenizer():
    return train_tokenizer(["the cat sat on the mat. the dog sat on the log. " * 30],
                           vocab_size=300, special_tokens=SPECIALS, verbose=False)


@pytest.fixture
def model(tokenizer) -> BingBongLM:
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=tokenizer.vocab_size, context_length=24,
                      embedding_dim=32, num_layers=1, num_heads=4)
    return BingBongLM(cfg).eval()


def fixed_logits() -> torch.Tensor:
    """Logits with a known order: token 0 best, then 1, 2, ... descending."""
    return torch.linspace(3.0, -3.0, steps=10)


# -- sample_next_token ------------------------------------------------------


def test_temperature_zero_is_argmax() -> None:
    logits = torch.tensor([0.1, 2.5, -1.0, 2.4])
    token, probs = sample_next_token(logits, temperature=0.0, top_k=None, generator=None)
    assert token == 1
    assert probs.tolist() == [0.0, 1.0, 0.0, 0.0]


def test_top_k_one_is_argmax_at_any_temperature() -> None:
    gen = torch.Generator().manual_seed(0)
    for _ in range(50):
        token, _ = sample_next_token(fixed_logits(), temperature=5.0, top_k=1, generator=gen)
        assert token == 0


def test_top_k_never_samples_outside_the_top_k() -> None:
    gen = torch.Generator().manual_seed(0)
    seen = {sample_next_token(fixed_logits(), 10.0, top_k=3, generator=gen)[0] for _ in range(500)}
    assert seen <= {0, 1, 2}
    assert seen == {0, 1, 2}          # high temperature: all three do get picked


def test_top_k_probabilities_outside_k_are_exactly_zero() -> None:
    _, probs = sample_next_token(fixed_logits(), 1.0, top_k=4, generator=torch.Generator())
    assert torch.all(probs[4:] == 0.0)
    assert probs.sum().item() == pytest.approx(1.0)


def test_temperature_one_matches_softmax() -> None:
    logits = fixed_logits()
    _, probs = sample_next_token(logits, 1.0, top_k=None, generator=torch.Generator())
    assert torch.allclose(probs, torch.softmax(logits, dim=-1))


def entropy(p: torch.Tensor) -> float:
    p = p[p > 0]
    return float(-(p * p.log()).sum())


def test_higher_temperature_flattens_the_distribution() -> None:
    entropies = [entropy(sample_next_token(fixed_logits(), t, None, torch.Generator())[1])
                 for t in (0.25, 0.5, 1.0, 2.0, 8.0)]
    assert all(a < b for a, b in zip(entropies, entropies[1:]))
    assert entropies[-1] < math.log(10) + 1e-6


def test_empirical_frequencies_follow_the_distribution() -> None:
    """Sampling is actually drawing from probs, not something else."""
    logits = torch.log(torch.tensor([0.6, 0.3, 0.1]))
    gen = torch.Generator().manual_seed(123)
    counts = Counter(sample_next_token(logits, 1.0, None, gen)[0] for _ in range(4000))
    assert counts[0] / 4000 == pytest.approx(0.6, abs=0.03)
    assert counts[1] / 4000 == pytest.approx(0.3, abs=0.03)
    assert counts[2] / 4000 == pytest.approx(0.1, abs=0.03)


def test_vocab_limit_excludes_ids_the_tokenizer_cannot_decode() -> None:
    logits = torch.tensor([0.0, 0.0, 0.0, 100.0, 100.0])      # best two are out of range
    gen = torch.Generator().manual_seed(0)
    for temperature in (0.0, 1.0, 3.0):
        for _ in range(50):
            token, _ = sample_next_token(logits, temperature, None, gen, vocab_limit=3)
            assert token < 3


@pytest.mark.parametrize("kwargs, message", [
    ({"temperature": -0.1}, "temperature"),
    ({"top_k": 0}, "top_k"),
    ({"max_new_tokens": -1}, "max_new_tokens"),
])
def test_invalid_generation_settings_are_rejected(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        GenerationConfig(**kwargs)


# -- determinism -------------------------------------------------------------


def test_same_seed_gives_identical_text(model, tokenizer) -> None:
    config = GenerationConfig(max_new_tokens=30, temperature=1.5, top_k=None, seed=42)
    a = generate(model, tokenizer, "the cat", config)
    b = generate(model, tokenizer, "the cat", config)
    assert a.token_ids == b.token_ids and a.seed == b.seed == 42


def test_different_seeds_give_different_text(model, tokenizer) -> None:
    """An untrained model at high temperature is close to uniform, so 5 seeds
    producing 30 tokens each will not all coincide unless the seed is ignored."""
    outputs = {tuple(generate(model, tokenizer, "the cat",
                              GenerationConfig(max_new_tokens=30, temperature=2.0,
                                               top_k=None, seed=s)).token_ids)
               for s in range(5)}
    assert len(outputs) == 5


def test_unseeded_sampling_reports_a_seed_that_reproduces_it(model, tokenizer) -> None:
    first = generate(model, tokenizer, "the dog",
                     GenerationConfig(max_new_tokens=25, temperature=1.2, top_k=None))
    assert first.seed is not None
    again = generate(model, tokenizer, "the dog",
                     GenerationConfig(max_new_tokens=25, temperature=1.2, top_k=None, seed=first.seed))
    assert again.token_ids == first.token_ids


def test_greedy_ignores_the_seed_and_is_deterministic(model, tokenizer) -> None:
    results = [generate(model, tokenizer, "the cat",
                        GenerationConfig(max_new_tokens=20, temperature=0.0, seed=s)).token_ids
               for s in (1, 2, 3)]
    assert results[0] == results[1] == results[2]
    assert generate(model, tokenizer, "the cat", GenerationConfig(temperature=0.0)).seed is None


def test_generation_does_not_touch_the_global_rng(model, tokenizer) -> None:
    torch.manual_seed(7)
    expected = torch.rand(3)
    torch.manual_seed(7)
    generate(model, tokenizer, "the cat", GenerationConfig(max_new_tokens=10, temperature=1.0, seed=1))
    assert torch.equal(torch.rand(3), expected)


# -- stopping and limits -----------------------------------------------------


def test_max_new_tokens_is_respected(model, tokenizer) -> None:
    result = generate(model, tokenizer, "the", GenerationConfig(max_new_tokens=7, temperature=0.0))
    assert len(result.token_ids) <= 7
    if result.stop_reason == "max_new_tokens":
        assert len(result.token_ids) == 7


def test_zero_new_tokens_returns_nothing(model, tokenizer) -> None:
    result = generate(model, tokenizer, "the", GenerationConfig(max_new_tokens=0))
    assert result.text == "" and result.token_ids == []


def test_generation_stops_at_eos(model, tokenizer) -> None:
    with torch.no_grad():                        # make <|eos|> overwhelmingly likely
        bias = torch.zeros(model.cfg.vocab_size)
        bias[tokenizer.eos_id] = 1e4
    original_forward = model.forward
    model.forward = lambda ids, targets=None: (original_forward(ids)[0] + bias, None)
    result = generate(model, tokenizer, "the cat", GenerationConfig(max_new_tokens=20, temperature=0.0))
    assert result.stop_reason == "eos"
    assert result.token_ids == []


def test_stop_string_ends_generation(model, tokenizer) -> None:
    free = generate(model, tokenizer, "the cat", GenerationConfig(max_new_tokens=30, temperature=0.0))
    assert len(free.text) > 2
    stop = free.text[1:3]                        # something that will definitely appear
    stopped = generate(model, tokenizer, "the cat",
                       GenerationConfig(max_new_tokens=30, temperature=0.0, stop_strings=(stop,)))
    assert stopped.stop_reason == "stop_string"
    assert stop in stopped.text
    assert len(stopped.token_ids) <= len(free.token_ids)


def test_prompt_longer_than_context_is_truncated_not_an_error(model, tokenizer) -> None:
    long_prompt = "the cat sat on the mat. " * 40
    result = generate(model, tokenizer, long_prompt, GenerationConfig(max_new_tokens=5, temperature=0.0))
    assert result.prompt_truncated
    assert result.prompt_tokens > model.cfg.context_length


def test_empty_prompt_still_generates(model, tokenizer) -> None:
    result = generate(model, tokenizer, "", GenerationConfig(max_new_tokens=5, temperature=0.0))
    assert result.prompt_tokens == 1


def test_generation_leaves_the_model_in_its_original_mode(model, tokenizer) -> None:
    model.train()
    generate(model, tokenizer, "the", GenerationConfig(max_new_tokens=3, temperature=0.0))
    assert model.training


def test_generate_greedy_wrapper_matches_generate(model, tokenizer) -> None:
    text = generate_greedy(model, tokenizer, "the dog", max_new_tokens=15)
    assert text == generate(model, tokenizer, "the dog",
                            GenerationConfig(max_new_tokens=15, temperature=0.0)).text


# -- one token at a time -----------------------------------------------------------


def test_each_step_depends_only_on_previous_tokens(model, tokenizer) -> None:
    """Generating 10 tokens then 5 more from the result equals generating 15 at once.

    This is autoregression stated as a test: the model's only state is the token
    sequence itself.
    """
    greedy = GenerationConfig(temperature=0.0)
    whole = generate(model, tokenizer, "the cat",
                     GenerationConfig(max_new_tokens=15, temperature=0.0))
    first = generate(model, tokenizer, "the cat",
                     GenerationConfig(max_new_tokens=10, temperature=0.0))
    prefix_ids = tokenizer.encode("the cat") + first.token_ids
    # Continue from the exact ids, not re-encoded text, so tokenization cannot differ.
    ids = list(prefix_ids)
    with torch.no_grad():
        for _ in range(5):
            logits, _ = model(torch.tensor([ids[-model.cfg.context_length:]]))
            token, _ = sample_next_token(logits[0, -1], 0.0, None, None, tokenizer.vocab_size)
            ids.append(token)
    assert ids[len(tokenizer.encode("the cat")):] == whole.token_ids
    assert greedy.greedy


def test_trace_records_every_step(model, tokenizer) -> None:
    result = generate(model, tokenizer, "the",
                      GenerationConfig(max_new_tokens=6, temperature=0.7, top_k=5, seed=3),
                      trace=True)
    assert len(result.steps) == len(result.token_ids)
    for step, token_id in zip(result.steps, result.token_ids):
        assert step.token_id == token_id
        assert 0.0 <= step.model_probability <= 1.0
        assert 0.0 < step.sampling_probability <= 1.0     # it was sampled, so it had mass
        assert len(step.alternatives) == 5


# -- streaming ---------------------------------------------------------------


def test_streamed_pieces_join_to_the_final_text(model, tokenizer) -> None:
    config = GenerationConfig(max_new_tokens=25, temperature=1.0, top_k=None, seed=11)
    pieces = list(stream_tokens(model, tokenizer, "the cat", config))
    assert "".join(pieces) == generate(model, tokenizer, "the cat", config).text


def test_incremental_decoder_holds_back_partial_characters(tokenizer) -> None:
    """🤖 is 4 UTF-8 bytes. Fed one byte-token at a time, nothing prints until it is whole."""
    emoji_ids = list("🤖".encode("utf-8"))            # raw byte ids 0-255 exist in every tokenizer
    decoder = IncrementalDecoder(tokenizer)
    deltas = [decoder.push(i) for i in emoji_ids]
    assert deltas[:3] == ["", "", ""]
    assert deltas[3] == "🤖"
    assert decoder.flush() == ""


def test_incremental_decoder_flushes_invalid_trailing_bytes(tokenizer) -> None:
    decoder = IncrementalDecoder(tokenizer)
    assert decoder.push(ord("a")) == "a"
    assert decoder.push(0xF0) == ""                   # start of a 4-byte char that never finishes
    assert decoder.flush() == "�"


# -- loading a checkpoint ------------------------------------------------------


def write_checkpoint(tmp_path: Path, model: BingBongLM, tokenizer, fingerprint: str | None = None) -> Path:
    tok_dir = tmp_path / "tok"
    tokenizer.save(tok_dir)
    return save_checkpoint(tmp_path / "model.pt", {
        "model": model.state_dict(),
        "step": 123,
        "best_val_loss": 0.5,
        "config": {"name": "unit", "tokenizer": {"vocab_path": str(tok_dir)}},
        "model_config": model.cfg.to_dict(),
        "tokenizer_fingerprint": fingerprint or tokenizer.fingerprint(),
    })


def test_load_for_inference_reproduces_the_model(model, tokenizer, tmp_path: Path) -> None:
    path = write_checkpoint(tmp_path, model, tokenizer)
    loaded = load_for_inference(path, CPU)
    assert loaded.step == 123 and loaded.config_name == "unit" and loaded.best_val_loss == 0.5
    ids = torch.tensor([tokenizer.encode("the cat sat")])
    with torch.no_grad():
        assert torch.allclose(loaded.model(ids)[0], model(ids)[0], atol=1e-5)
    assert not loaded.model.training


def test_load_for_inference_rejects_a_mismatched_tokenizer(model, tokenizer, tmp_path: Path) -> None:
    path = write_checkpoint(tmp_path, model, tokenizer, fingerprint="0000000000000000")
    with pytest.raises(CheckpointError, match="fingerprint"):
        load_for_inference(path, CPU)

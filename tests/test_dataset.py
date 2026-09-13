"""Tests for dataset preparation and batching."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tokenizer.train_tokenizer import train_tokenizer  # noqa: E402
from src.training.dataloader import BatchLoader  # noqa: E402
from src.training.dataset import (  # noqa: E402
    TokenDataset,
    read_text_files,
    tokenize_documents,
    write_token_file,
)
from src.training.synthetic import build_corpus, completion_cases  # noqa: E402

SPECIALS = {"pad": "<|pad|>", "unk": "<|unk|>", "bos": "<|bos|>", "eos": "<|eos|>"}
CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def tokenizer():
    return train_tokenizer([build_corpus(20, seed=1)], vocab_size=400,
                           special_tokens=SPECIALS, verbose=False)


def make_dataset(tmp_path: Path, n_tokens: int, context_length: int) -> TokenDataset:
    path = tmp_path / "tokens.bin"
    write_token_file(path, list(range(n_tokens)))     # token id == its position
    return TokenDataset(path, context_length)


# -- windows ---------------------------------------------------------------


def test_target_is_input_shifted_by_one(tmp_path: Path) -> None:
    """The definition of next-token prediction: y[t] is the token after x[t]."""
    ds = make_dataset(tmp_path, n_tokens=1000, context_length=16)
    x, y = ds.windows(torch.tensor([0, 37, 983]))
    assert x.shape == (3, 16) and y.shape == (3, 16)
    assert torch.equal(y[:, :-1], x[:, 1:])
    assert torch.equal(y, x + 1)            # ids equal positions in this fixture
    assert x[1, 0] == 37


def test_windows_are_int64(tmp_path: Path) -> None:
    """Stored as uint16, but embedding lookup needs int64."""
    ds = make_dataset(tmp_path, n_tokens=100, context_length=8)
    x, y = ds.windows(torch.tensor([0]))
    assert x.dtype == torch.int64 and y.dtype == torch.int64


def test_last_valid_window_reaches_the_final_token(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path, n_tokens=100, context_length=8)
    assert len(ds) == 92
    _, y = ds.windows(torch.tensor([len(ds) - 1]))
    assert y[0, -1] == 99


def test_too_little_data_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="fewer than"):
        make_dataset(tmp_path, n_tokens=8, context_length=8)


def test_missing_token_file_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="prepare_data"):
        TokenDataset(tmp_path / "nope.bin", 8)


def test_token_file_round_trip(tmp_path: Path) -> None:
    ids = [0, 1, 255, 256, 8191, 65535]
    write_token_file(tmp_path / "t.bin", ids)
    assert np.fromfile(tmp_path / "t.bin", dtype=np.uint16).tolist() == ids


# -- tokenizing and splitting ----------------------------------------------


def test_every_document_is_split_and_ends_with_eos(tokenizer, tmp_path: Path) -> None:
    docs = [(tmp_path / "a.txt", "The color of snow is white.\n" * 30),
            (tmp_path / "b.txt", "After one comes two.\n" * 30)]
    corpus = tokenize_documents(docs, tokenizer, val_fraction=0.1)

    assert len(corpus.per_document) == 2
    for doc in corpus.per_document:
        assert doc["train_tokens"] > 0 and doc["val_tokens"] > 0
    assert corpus.train_ids.count(tokenizer.eos_id) == 0      # eos sits in each doc's tail
    assert corpus.val_ids.count(tokenizer.eos_id) == 2
    assert len(corpus.train_ids) + len(corpus.val_ids) == sum(d["tokens"] for d in corpus.per_document)


def test_split_is_lossless(tokenizer, tmp_path: Path) -> None:
    """Train + val, decoded back, must be exactly the original documents."""
    text = "The capital of Peru is Lima.\n" * 40
    corpus = tokenize_documents([(tmp_path / "a.txt", text)], tokenizer, val_fraction=0.25)
    rebuilt = tokenizer.decode(corpus.train_ids + corpus.val_ids, skip_special=True)
    assert rebuilt == text


def test_corpus_literal_special_tokens_are_not_injected(tokenizer, tmp_path: Path) -> None:
    corpus = tokenize_documents([(tmp_path / "a.txt", "hi <|eos|> there " * 20)],
                                tokenizer, val_fraction=0.1)
    all_ids = corpus.train_ids + corpus.val_ids
    assert all_ids.count(tokenizer.eos_id) == 1              # only the real end-of-document


def test_invalid_val_fraction_is_rejected(tokenizer, tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        tokenize_documents([(tmp_path / "a.txt", "text")], tokenizer, val_fraction=1.5)


def test_read_text_files_finds_txt_and_md_recursively(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "sub" / "b.md").write_text("beta", encoding="utf-8")
    (tmp_path / "c.bin").write_bytes(b"\x00\x01")
    (tmp_path / "empty.txt").write_text("   ", encoding="utf-8")
    names = [p.name for p, _ in read_text_files(tmp_path)]
    assert names == ["a.txt", "b.md"]


# -- batch loader ------------------------------------------------------------


def test_batch_shapes(tmp_path: Path) -> None:
    loader = BatchLoader(make_dataset(tmp_path, 500, 32), batch_size=4, device=CPU, seed=0)
    x, y = loader.next_batch()
    assert x.shape == (4, 32) and y.shape == (4, 32)


def test_same_seed_gives_same_batches(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path, 500, 32)
    a = BatchLoader(ds, 4, CPU, seed=7)
    b = BatchLoader(ds, 4, CPU, seed=7)
    for _ in range(5):
        assert torch.equal(a.next_batch()[0], b.next_batch()[0])


def test_loader_state_restores_the_batch_sequence(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path, 500, 32)
    loader = BatchLoader(ds, 4, CPU, seed=7)
    loader.next_batch()
    state = loader.state_dict()
    expected = [loader.next_batch()[0] for _ in range(3)]

    restored = BatchLoader(ds, 4, CPU, seed=999)          # different seed on purpose
    restored.load_state_dict(state)
    for want in expected:
        assert torch.equal(restored.next_batch()[0], want)


def test_fixed_batches_are_identical_every_call_and_do_not_disturb_training(tmp_path: Path) -> None:
    ds = make_dataset(tmp_path, 500, 32)
    loader = BatchLoader(ds, 4, CPU, seed=7)
    reference = BatchLoader(ds, 4, CPU, seed=7)

    first = [x for x, _ in loader.fixed_batches(3, seed=42)]
    second = [x for x, _ in loader.fixed_batches(3, seed=42)]
    assert all(torch.equal(a, b) for a, b in zip(first, second))
    # Evaluating must not shift the training batch sequence.
    assert torch.equal(loader.next_batch()[0], reference.next_batch()[0])


# -- synthetic corpus ---------------------------------------------------------


def test_synthetic_answers_are_determined_by_their_prefix() -> None:
    """The whole experiment relies on each prefix having exactly one answer."""
    seen: dict[str, str] = {}
    for case in completion_cases():
        assert seen.setdefault(case.prefix, case.answer) == case.answer
    assert len(seen) == len(completion_cases()) == 33


def test_synthetic_corpus_is_reproducible() -> None:
    assert build_corpus(5, seed=3) == build_corpus(5, seed=3)
    assert build_corpus(5, seed=3) != build_corpus(5, seed=4)

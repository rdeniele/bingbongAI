"""Tests for the tokenizer.

The critical property is LOSSLESSNESS: encode then decode must return exactly
what went in. If that ever breaks, every downstream number -- loss, generated
text, retrieval -- is quietly measuring the wrong thing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tokenizer.tokenizer import (  # noqa: E402
    BYTE_VOCAB_SIZE,
    BPETokenizer,
    _PRETOKEN_PATTERN,
)
from src.tokenizer.train_tokenizer import (  # noqa: E402
    build_word_frequencies,
    learn_merges,
    merge_word,
    train_tokenizer,
)

SPECIALS = {"pad": "<|pad|>", "unk": "<|unk|>", "bos": "<|bos|>", "eos": "<|eos|>"}

CORPUS = [
    "The color of the sky is blue. The color of grass is green.\n",
    "the quick brown fox jumps over the lazy dog. the dog sleeps.\n",
    "def train(model, data):\n    for batch in data:\n        loss = model(batch)\n",
    "Programming is the art of telling a computer what to do, precisely.\n",
]

ROUND_TRIP_SAMPLES = [
    "",
    "a",
    " ",
    "\n",
    "The color of the sky is blue.",
    "hello world",
    "   leading and trailing   ",
    "line one\nline two\n\nline four",
    "tabs\there",
    "punctuation!!! ...? (parens) [brackets] {braces} <angle>",
    "numbers 0 42 3.14159 1,000,000 007",
    "snake_case camelCase kebab-case CONSTANT_NAME __dunder__",
    "don't can't it's we've they're I'd I'm",
    "unicode: café naïve résumé Köln",
    "emoji: 🤖🔥✨",
    "cjk: 日本語のテキスト 中文字符",
    "filipino: kumusta ka? ayos lang ako.",
    "mixed 🤖 café _x_ 42! \n\t done",
    "def f(x): return x ** 2  # comment",
    "a" * 500,
    "word " * 200,
]


@pytest.fixture(scope="module")
def tokenizer() -> BPETokenizer:
    """A small tokenizer trained on the test corpus."""
    return train_tokenizer(CORPUS, vocab_size=400, special_tokens=SPECIALS, verbose=False)


# -- pre-tokenization ----------------------------------------------------


@pytest.mark.parametrize("text", ROUND_TRIP_SAMPLES)
def test_pretokenizer_covers_every_character(text: str) -> None:
    """The regex must partition the text, dropping nothing.

    This is the foundation of losslessness. If the pattern ever fails to match
    some character, that character silently disappears during encoding and no
    other test would obviously catch it.
    """
    assert "".join(_PRETOKEN_PATTERN.findall(text)) == text


# -- round trip ----------------------------------------------------------


@pytest.mark.parametrize("text", ROUND_TRIP_SAMPLES)
def test_encode_decode_round_trip(tokenizer: BPETokenizer, text: str) -> None:
    """text -> encode -> tokens -> decode -> the original text, exactly."""
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_round_trip_on_bytes_never_seen_in_training(tokenizer: BPETokenizer) -> None:
    """Byte-level means there is no such thing as out-of-vocabulary input."""
    text = "ЖЖЖ ﷽ 𝔘𝔫𝔦𝔠𝔬𝔡𝔢 \x00\x01\x02"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_all_token_ids_are_in_range(tokenizer: BPETokenizer) -> None:
    for text in ROUND_TRIP_SAMPLES:
        for token_id in tokenizer.encode(text):
            assert 0 <= token_id < tokenizer.vocab_size


def test_encoding_is_deterministic(tokenizer: BPETokenizer) -> None:
    text = "The color of the sky is blue."
    assert tokenizer.encode(text) == tokenizer.encode(text)


# -- vocabulary ----------------------------------------------------------


def test_vocab_size_is_bytes_plus_merges_plus_specials(tokenizer: BPETokenizer) -> None:
    assert tokenizer.vocab_size == BYTE_VOCAB_SIZE + tokenizer.num_merges + len(SPECIALS)


def test_requested_vocab_size_is_respected() -> None:
    tok = train_tokenizer(CORPUS, vocab_size=400, special_tokens=SPECIALS, verbose=False)
    # May stop early if the corpus runs out of repeated pairs, but never exceed.
    assert tok.vocab_size <= 400


def test_vocab_size_below_the_floor_is_rejected() -> None:
    with pytest.raises(ValueError, match="too small"):
        train_tokenizer(CORPUS, vocab_size=100, special_tokens=SPECIALS, verbose=False)


def test_bpe_actually_compresses(tokenizer: BPETokenizer) -> None:
    """A trained tokenizer must beat raw bytes on text resembling its corpus."""
    text = "The color of the sky is blue. the quick brown fox jumps over the lazy dog."
    assert len(tokenizer.encode(text)) < len(text.encode("utf-8"))


def test_merges_are_learned_in_frequency_order() -> None:
    """The first merge must be the most frequent adjacent pair in the corpus.

    Single pre-token chunks are used here on purpose. In running text the
    pre-tokenizer attaches a leading space to each word, so in "ab ab ac ac"
    the most frequent pair is actually (space, 'a') -- see the test below.
    """
    merges = learn_merges(["ababab acac"], num_merges=1)
    assert len(merges) == 1
    assert merges[0] == (ord("a"), ord("b"))


def test_space_is_attached_to_the_following_word() -> None:
    """" the" should become one token, not " " + "the".

    This is why the pre-tokenizer's alternatives start with " ?" -- most words
    in running text are space-prefixed, so folding the space in roughly halves
    the token count.
    """
    tok = train_tokenizer(
        ["the cat and the dog and the bird "] * 20,
        vocab_size=300,
        special_tokens=SPECIALS,
        verbose=False,
    )
    pieces = [tok.decode([i]) for i in tok.encode("the cat and the dog")]
    assert " the" in pieces
    assert " " not in pieces


def test_min_frequency_stops_useless_merges() -> None:
    """A pair seen once should not be given a token id."""
    merges = learn_merges(["abcdefgh"], num_merges=50, min_frequency=2)
    assert merges == []


# -- special tokens ------------------------------------------------------


def test_special_token_ids_are_distinct_and_at_the_top(tokenizer: BPETokenizer) -> None:
    ids = {name: tokenizer.token_id(name) for name in SPECIALS}
    assert len(set(ids.values())) == len(SPECIALS)
    for token_id in ids.values():
        assert BYTE_VOCAB_SIZE + tokenizer.num_merges <= token_id < tokenizer.vocab_size


def test_add_bos_and_eos(tokenizer: BPETokenizer) -> None:
    ids = tokenizer.encode("hello", add_bos=True, add_eos=True)
    assert ids[0] == tokenizer.bos_id
    assert ids[-1] == tokenizer.eos_id
    assert tokenizer.decode(ids[1:-1]) == "hello"


def test_skip_special_drops_control_tokens(tokenizer: BPETokenizer) -> None:
    ids = tokenizer.encode("hello", add_bos=True, add_eos=True)
    assert tokenizer.decode(ids, skip_special=True) == "hello"


def test_special_literal_is_recognised_when_allowed(tokenizer: BPETokenizer) -> None:
    ids = tokenizer.encode("a<|eos|>b", allow_special=True)
    assert tokenizer.eos_id in ids


def test_special_literal_is_inert_when_disallowed(tokenizer: BPETokenizer) -> None:
    """Untrusted text must not be able to inject control tokens."""
    ids = tokenizer.encode("a<|eos|>b", allow_special=False)
    assert tokenizer.eos_id not in ids
    assert tokenizer.decode(ids) == "a<|eos|>b"


def test_unknown_token_exists_but_is_never_emitted(tokenizer: BPETokenizer) -> None:
    """The unk id is reserved; byte-level encoding structurally cannot produce it."""
    assert 0 <= tokenizer.unk_id < tokenizer.vocab_size
    weird = "🜁🜂🜃🜄 \x00\xff ΩΩΩ 𐐷𐐷"
    assert tokenizer.unk_id not in tokenizer.encode(weird, allow_special=False)


def test_unknown_special_name_raises(tokenizer: BPETokenizer) -> None:
    with pytest.raises(KeyError):
        tokenizer.token_id("nope")


# -- save / load ---------------------------------------------------------


def test_save_then_load_preserves_behaviour(tokenizer: BPETokenizer, tmp_path: Path) -> None:
    tokenizer.save(tmp_path)
    reloaded = BPETokenizer.load(tmp_path)

    assert reloaded.vocab_size == tokenizer.vocab_size
    assert reloaded.merges == tokenizer.merges
    assert reloaded.special_tokens == tokenizer.special_tokens
    assert reloaded.fingerprint() == tokenizer.fingerprint()

    for text in ROUND_TRIP_SAMPLES:
        assert reloaded.encode(text) == tokenizer.encode(text)
        assert reloaded.decode(reloaded.encode(text)) == text


def test_load_from_missing_directory_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Train one first"):
        BPETokenizer.load(tmp_path / "does-not-exist")


def test_fingerprint_changes_when_vocabulary_changes(tokenizer: BPETokenizer) -> None:
    """Different vocabulary must mean a different fingerprint, or the checkpoint
    guard in TRAINING.md would not catch a mismatched tokenizer."""
    other = train_tokenizer(CORPUS, vocab_size=300, special_tokens=SPECIALS, verbose=False)
    assert other.fingerprint() != tokenizer.fingerprint()


# -- error handling ------------------------------------------------------


def test_decoding_an_out_of_range_id_raises(tokenizer: BPETokenizer) -> None:
    with pytest.raises(ValueError, match="outside this tokenizer's vocabulary"):
        tokenizer.decode([tokenizer.vocab_size + 10])


def test_training_on_an_empty_corpus_raises() -> None:
    with pytest.raises(ValueError, match="corpus is empty"):
        learn_merges([""], num_merges=10)


# -- internals -----------------------------------------------------------


def test_merge_word_replaces_every_occurrence() -> None:
    assert merge_word((1, 2, 3, 1, 2), (1, 2), 99) == (99, 3, 99)


def test_merge_word_handles_overlap_left_to_right() -> None:
    assert merge_word((1, 1, 1), (1, 1), 99) == (99, 1)


def test_word_frequencies_count_repeats() -> None:
    freqs = build_word_frequencies(["the the the"])
    assert sum(freqs.values()) == 3

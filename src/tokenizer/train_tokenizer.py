"""Learning BPE merges from a corpus.

THE ALGORITHM, IN ONE PARAGRAPH
-------------------------------
Start with text as raw bytes -- 256 possible tokens, so "the" is three tokens.
Count every adjacent pair of tokens in the corpus. Take the most frequent pair,
declare it a new token, and replace every occurrence. Now recount and repeat.
After a few thousand rounds the frequent sequences of English -- " the", "ing",
" the ", whole common words -- have each collapsed into a single token, while
rare sequences are still made of small pieces. That is the whole idea: the
vocabulary spends its budget where the text actually is.

IMPLEMENTATION NOTE
-------------------
The naive version rescans the entire corpus for every merge, which is far too
slow. Instead we exploit the fact that text is enormously repetitive: collapse
the corpus into a table of UNIQUE pre-token chunks with their frequencies, then
do all the merging on that table with counts as weights. A corpus of a million
words typically has only tens of thousands of distinct chunks, so this is orders
of magnitude less work for an identical result.
"""

from __future__ import annotations

import time
from collections import Counter
from typing import Callable, Iterable

from .tokenizer import BYTE_VOCAB_SIZE, BPETokenizer, _PRETOKEN_PATTERN

# A "word" here is one pre-token chunk represented as its byte/token ids.
Word = tuple[int, ...]


def build_word_frequencies(texts: Iterable[str]) -> dict[Word, int]:
    """Collapse a corpus into {unique pre-token chunk as ids: count}."""
    counts: Counter[bytes] = Counter()
    for text in texts:
        for chunk in _PRETOKEN_PATTERN.findall(text):
            counts[chunk.encode("utf-8")] += 1
    return {tuple(raw): count for raw, count in counts.items()}


def count_pairs(word_freqs: dict[Word, int]) -> Counter[tuple[int, int]]:
    """Count adjacent token pairs across the corpus, weighted by word frequency."""
    pairs: Counter[tuple[int, int]] = Counter()
    for word, freq in word_freqs.items():
        for pair in zip(word, word[1:]):
            pairs[pair] += freq
    return pairs


def merge_word(word: Word, pair: tuple[int, int], new_id: int) -> Word:
    """Replace every occurrence of `pair` in `word` with `new_id`."""
    if len(word) < 2:
        return word

    left, right = pair
    out: list[int] = []
    index = 0
    limit = len(word) - 1
    while index < len(word):
        if index < limit and word[index] == left and word[index + 1] == right:
            out.append(new_id)
            index += 2
        else:
            out.append(word[index])
            index += 1
    return tuple(out)


def learn_merges(
    texts: Iterable[str],
    num_merges: int,
    min_frequency: int = 2,
    progress: Callable[[int, int, tuple[int, int], int], None] | None = None,
) -> list[tuple[int, int]]:
    """Learn `num_merges` BPE merges from a corpus.

    Args:
        texts: the corpus, as an iterable of strings.
        num_merges: how many merges to learn. Final vocabulary size will be
            256 + num_merges + (number of special tokens).
        min_frequency: stop early if the best remaining pair occurs fewer than
            this many times. Merging a pair seen once creates a token that will
            essentially never be used again -- wasted vocabulary, and wasted
            embedding parameters, since every token costs `embedding_dim`
            weights whether it is used or not.
        progress: optional callback(merge_index, total, pair, frequency).

    Returns:
        Merges in the order learned. Order IS the data -- encoding replays them
        in exactly this sequence.
    """
    if num_merges < 0:
        raise ValueError(f"num_merges must be non-negative, got {num_merges}")

    word_freqs = build_word_frequencies(texts)
    if not word_freqs:
        raise ValueError("corpus is empty: no text to learn merges from")

    merges: list[tuple[int, int]] = []

    for step in range(num_merges):
        pairs = count_pairs(word_freqs)
        if not pairs:
            break  # every word has collapsed to a single token

        best_pair, best_count = pairs.most_common(1)[0]
        if best_count < min_frequency:
            break

        new_id = BYTE_VOCAB_SIZE + len(merges)
        merges.append(best_pair)

        # Rebuild the frequency table with this pair merged. Only words that
        # actually contain the pair change, but checking costs about as much as
        # rebuilding for corpora of the size this project trains on.
        word_freqs = {
            merge_word(word, best_pair, new_id): freq for word, freq in word_freqs.items()
        }

        if progress is not None:
            progress(step + 1, num_merges, best_pair, best_count)

    return merges


def train_tokenizer(
    texts: Iterable[str],
    vocab_size: int,
    special_tokens: dict[str, str],
    min_frequency: int = 2,
    verbose: bool = True,
) -> BPETokenizer:
    """Train a BPETokenizer to a target vocabulary size.

    The budget: 256 base byte tokens and one id per special token are fixed
    costs, and everything left over is spent on merges.
    """
    texts = list(texts)
    num_merges = vocab_size - BYTE_VOCAB_SIZE - len(special_tokens)
    if num_merges < 0:
        raise ValueError(
            f"vocab_size {vocab_size} is too small: 256 byte tokens plus "
            f"{len(special_tokens)} special tokens already need "
            f"{BYTE_VOCAB_SIZE + len(special_tokens)} ids"
        )

    started = time.perf_counter()

    def report(step: int, total: int, pair: tuple[int, int], freq: int) -> None:
        if step % 500 == 0 or step == total:
            elapsed = time.perf_counter() - started
            print(f"  merge {step:>6}/{total} | top pair seen {freq:>8}x | {elapsed:6.1f}s")

    if verbose:
        print(f"Learning up to {num_merges} merges (target vocab {vocab_size})...")

    merges = learn_merges(
        texts,
        num_merges=num_merges,
        min_frequency=min_frequency,
        progress=report if verbose else None,
    )

    if verbose and len(merges) < num_merges:
        print(
            f"  stopped early at {len(merges)} merges: no remaining pair occurs "
            f"at least {min_frequency} times. The corpus is too small to support "
            f"a vocabulary of {vocab_size}."
        )

    return BPETokenizer(merges=merges, special_tokens=special_tokens)

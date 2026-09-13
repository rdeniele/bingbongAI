"""The synthetic learning-proof dataset: patterns a working model MUST memorise.

Used by scripts/make_synthetic.py (to write the corpus) and by
scripts/prove_learning.py (to test the trained model on exactly those patterns).

Every sentence ends in a word that is fully determined by the words before it.
Given "The color of snow is", only " white" is ever correct. That makes success
measurable without judgement calls: the model either completes the pattern or
it does not.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

COLORS = [
    ("the sky", "blue"),
    ("grass", "green"),
    ("snow", "white"),
    ("coal", "black"),
    ("blood", "red"),
    ("the sun", "yellow"),
    ("an orange", "orange"),
    ("chocolate", "brown"),
    ("lavender", "purple"),
    ("a cloud", "grey"),
]

COUNTING = [
    ("one", "two"),
    ("two", "three"),
    ("three", "four"),
    ("four", "five"),
    ("five", "six"),
    ("six", "seven"),
    ("seven", "eight"),
    ("eight", "nine"),
    ("nine", "ten"),
]

ANIMALS = [
    ("a dog", "barks"),
    ("a cat", "meows"),
    ("a cow", "moos"),
    ("a bird", "sings"),
    ("a lion", "roars"),
    ("a duck", "quacks"),
    ("a horse", "neighs"),
    ("a frog", "croaks"),
]

CAPITALS = [
    ("the Philippines", "Manila"),
    ("Japan", "Tokyo"),
    ("France", "Paris"),
    ("Kenya", "Nairobi"),
    ("Peru", "Lima"),
    ("Norway", "Oslo"),
]


@dataclass(frozen=True)
class CompletionCase:
    """One pattern split into the prompt and the word that must follow it."""

    category: str
    prefix: str     # "The color of snow is"
    answer: str     # "white"

    @property
    def sentence(self) -> str:
        return f"{self.prefix} {self.answer}."


def completion_cases() -> list[CompletionCase]:
    """Every pattern in the dataset, in a fixed order."""
    cases: list[CompletionCase] = []
    cases += [CompletionCase("color", f"The color of {thing} is", color) for thing, color in COLORS]
    cases += [CompletionCase("counting", f"After {a} comes", b) for a, b in COUNTING]
    cases += [CompletionCase("animal", subject.capitalize(), sound) for subject, sound in ANIMALS]
    cases += [CompletionCase("capital", f"The capital of {country} is", city)
              for country, city in CAPITALS]
    return cases


def build_sentences() -> list[str]:
    return [case.sentence for case in completion_cases()]


def build_corpus(repeats: int, seed: int) -> str:
    """`repeats` passes over all sentences, each pass independently shuffled.

    Shuffling matters for interpreting the loss: because sentence ORDER is random,
    the first tokens of each sentence are genuinely unpredictable. The model can
    learn everything within a sentence, but it cannot know which sentence comes
    next. So the training loss has a floor above zero -- see
    `estimate_order_entropy_floor`.
    """
    rng = random.Random(seed)
    sentences = build_sentences()
    lines: list[str] = []
    for _ in range(repeats):
        shuffled = sentences[:]
        rng.shuffle(shuffled)
        lines.extend(shuffled)
    return "\n".join(lines) + "\n"


def estimate_order_entropy_floor(tokenizer, num_sentences: int | None = None) -> float:
    """Approximate lowest achievable average loss (nats/token) on this corpus.

    Suppose a model memorises every sentence perfectly but cannot predict WHICH
    sentence comes next. Each sentence then carries ln(N) nats of unavoidable
    surprise (a uniform choice among N sentences), spread over that sentence's
    tokens. Averaged per token:

        floor ~= ln(N) / (average tokens per sentence, including the newline)

    This is an approximation, and slightly pessimistic: sentences are shuffled
    WITHOUT replacement within each pass, so a model that tracks recent context
    can do a little better near the end of a pass. It is a reference line for
    reading the loss curve, not a hard bound.
    """
    import math

    sentences = build_sentences()
    n = num_sentences or len(sentences)
    total_tokens = sum(len(tokenizer.encode(s + "\n", allow_special=False)) for s in sentences)
    return math.log(n) / (total_tokens / len(sentences))

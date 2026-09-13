"""Autoregressive text generation: how BingBongAI writes, one token at a time.

THE LOOP

    ids = encode(prompt)
    repeat up to max_new_tokens times:
        window = ids[-context_length:]            the most recent T tokens
        logits = model(window)                    (1, T, V)
        scores = logits[0, -1]                    (V,)   ONLY the last position
        next   = choose(scores)                   one token id
        if next is <|eos|>: stop
        ids.append(next)                          it becomes input for the next round

The model predicts exactly ONE token per forward pass. A paragraph is that one
prediction repeated, each time feeding the model its own previous choice. Nothing
plans ahead: every token is chosen knowing only the tokens before it, and once
chosen it is never revised.

Only the last position's logits matter. The other positions are the model's
predictions for tokens that already exist -- that is what training uses, but it
is irrelevant for writing.

When the text grows past `context_length` (512), only the most recent 512 tokens
are fed in. Anything earlier is gone: the model cannot see it at all.

HOW `choose` WORKS  (sample_next_token)

  1. Restrict to real tokens. The output layer can be wider than the tokenizer
     (synthetic tokenizer: 484 tokens, model: 8192 rows). Ids with no text are
     never eligible.

  2. Temperature. Divide the logits by T before softmax:
         p_i = exp(z_i / T) / sum_j exp(z_j / T)
       T = 1     the model's own distribution, unchanged
       T < 1     sharper: likely tokens become likelier, the model gets conservative
       T > 1     flatter: unlikely tokens gain probability, output gets more varied
       T -> 0    all probability on the single top token: greedy decoding
     T = 0 is treated as greedy exactly (dividing by zero is not attempted).

  3. Top-k. Keep only the k highest-scoring tokens; set every other score to
     -infinity so its probability becomes exactly 0. This cuts off the long tail
     of individually unlikely tokens, which together can carry real probability
     and are where most nonsense comes from.

  4. Sample one id from the resulting distribution.

DETERMINISM

  Greedy (temperature 0) is deterministic by construction: argmax has no
  randomness.

  Sampling is made reproducible with a seed. Every call owns a private
  torch.Generator, so the same seed + prompt + settings + checkpoint gives the
  same text, and generation never disturbs (or is disturbed by) any other use of
  the global random state. When no seed is given one is drawn from the operating
  system -- and REPORTED in the result, so any output worth keeping can be
  reproduced later.

  Sampling happens on the CPU with that generator. The logits come from the GPU
  and can differ from CPU logits in the last few bits of precision, so the same
  seed is reproducible on the same device, but is not guaranteed identical across
  CPU and GPU.

PERFORMANCE NOTE
  Each step re-runs the model over the whole window, so the cost of step n grows
  with n. The standard fix is a key/value cache (keep each layer's k and v from
  previous steps and only compute the new token). It is not implemented yet --
  it belongs to the optimisation phase, and adds complexity best added once
  generation is known to be correct. scripts/generate.py reports measured
  tokens/second so the effect will be visible when it lands.
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field
from typing import Iterator

import torch
from torch import nn

REPLACEMENT_CHAR = "�"


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 100
    temperature: float = 1.0      # 0 = greedy
    top_k: int | None = 50        # None = no cutoff
    seed: int | None = None       # None = draw one from the OS (and report it)
    stop_strings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.max_new_tokens < 0:
            raise ValueError(f"max_new_tokens must be >= 0, got {self.max_new_tokens}")
        if self.temperature < 0:
            raise ValueError(f"temperature must be >= 0, got {self.temperature}")
        if self.top_k is not None and self.top_k < 1:
            raise ValueError(f"top_k must be >= 1 or None, got {self.top_k}")

    @property
    def greedy(self) -> bool:
        return self.temperature == 0


@dataclass
class TokenStep:
    """What happened at one generation step. Filled in only when tracing."""

    token_id: int
    text: str
    model_probability: float          # softmax of the raw logits, before temperature/top-k
    sampling_probability: float       # probability in the distribution actually sampled from
    alternatives: list[tuple[str, float]] = field(default_factory=list)  # top candidates


@dataclass
class GenerationResult:
    prompt: str
    text: str                         # the generated continuation only
    token_ids: list[int]
    prompt_tokens: int
    prompt_truncated: bool            # prompt was longer than the context window
    stop_reason: str                  # "max_new_tokens" | "eos" | "stop_string"
    seed: int | None                  # None for greedy
    seconds: float
    steps: list[TokenStep] = field(default_factory=list)

    @property
    def tokens_per_second(self) -> float:
        return len(self.token_ids) / self.seconds if self.seconds > 0 else 0.0


def sample_next_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int | None,
    generator: torch.Generator | None,
    vocab_limit: int | None = None,
) -> tuple[int, torch.Tensor]:
    """Choose one token id from a vector of logits.

    Args:
        logits: (V,) raw scores for the next token.
        temperature: 0 for greedy, otherwise divides the logits.
        top_k: keep only the k best tokens, or None for all.
        generator: CPU torch.Generator for sampling (unused when greedy).
        vocab_limit: only ids below this are eligible.

    Returns:
        (token_id, probs) where probs (V_eligible,) is the distribution the id
        was drawn from -- one-hot for greedy.
    """
    scores = logits.detach().float().cpu()
    if vocab_limit is not None:
        scores = scores[:vocab_limit]                                   # (V_eligible,)

    if temperature == 0:
        token_id = int(torch.argmax(scores))
        probs = torch.zeros_like(scores)
        probs[token_id] = 1.0
        return token_id, probs

    scores = scores / temperature
    if top_k is not None and top_k < scores.numel():
        kth_best = torch.topk(scores, top_k).values[-1]
        scores = scores.masked_fill(scores < kth_best, float("-inf"))

    probs = torch.softmax(scores, dim=-1)                               # (V_eligible,)
    token_id = int(torch.multinomial(probs, num_samples=1, generator=generator))
    return token_id, probs


class IncrementalDecoder:
    """Turn a growing list of token ids into text deltas, safely.

    A single character can span several tokens -- an emoji is 4 UTF-8 bytes, and
    a byte-level tokenizer may emit them one byte-token at a time. Decoding after
    the first byte produces U+FFFD, and printing that immediately would show
    garbage that later "turns into" the right character. So text ending in U+FFFD
    is held back until more tokens complete it.
    """

    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.ids: list[int] = []
        self.emitted = ""

    def push(self, token_id: int) -> str:
        self.ids.append(token_id)
        text = self.tokenizer.decode(self.ids, skip_special=True)
        if text.endswith(REPLACEMENT_CHAR):
            return ""                                   # probably an incomplete character
        delta = text[len(self.emitted):]
        self.emitted = text
        return delta

    def flush(self) -> str:
        """Whatever is left, including a genuinely invalid trailing fragment."""
        text = self.tokenizer.decode(self.ids, skip_special=True)
        delta = text[len(self.emitted):]
        self.emitted = text
        return delta


def _prepare(model: nn.Module, tokenizer, prompt: str) -> tuple[list[int], bool]:
    ids = tokenizer.encode(prompt, allow_special=False)
    if not ids:
        # An empty prompt still needs one position to predict from.
        ids = [tokenizer.bos_id]
    truncated = len(ids) > model.cfg.context_length
    return ids, truncated


@torch.no_grad()
def stream_tokens(
    model: nn.Module,
    tokenizer,
    prompt: str,
    config: GenerationConfig,
    result: GenerationResult | None = None,
    trace: bool = False,
    trace_alternatives: int = 5,
) -> Iterator[str]:
    """Generate, yielding text as it becomes printable.

    If `result` is given it is filled in as generation proceeds, so a caller can
    stream to the screen and still read the stop reason, seed and timing after.
    """
    device = next(model.parameters()).device
    context_length = model.cfg.context_length
    was_training = model.training
    model.eval()

    ids, truncated = _prepare(model, tokenizer, prompt)
    seed = None if config.greedy else (config.seed if config.seed is not None
                                       else secrets.randbits(31))
    generator = None if seed is None else torch.Generator().manual_seed(seed)
    decoder = IncrementalDecoder(tokenizer)

    if result is not None:
        result.prompt_tokens = len(ids)
        result.prompt_truncated = truncated
        result.seed = seed
        result.stop_reason = "max_new_tokens"

    started = time.perf_counter()
    try:
        for _ in range(config.max_new_tokens):
            window = torch.tensor([ids[-context_length:]], dtype=torch.long, device=device)  # (1, T)
            logits, _ = model(window)                                                        # (1, T, V)
            last = logits[0, -1]                                                             # (V,)

            token_id, probs = sample_next_token(
                last, config.temperature, config.top_k, generator,
                vocab_limit=tokenizer.vocab_size,
            )

            if trace and result is not None:
                raw = torch.softmax(last.float().cpu()[: tokenizer.vocab_size], dim=-1)
                top = torch.topk(raw, min(trace_alternatives, raw.numel()))
                result.steps.append(TokenStep(
                    token_id=token_id,
                    text=tokenizer.decode([token_id]),
                    model_probability=float(raw[token_id]),
                    sampling_probability=float(probs[token_id]),
                    alternatives=[(tokenizer.decode([int(i)]), float(p))
                                  for p, i in zip(top.values, top.indices)],
                ))

            if token_id == tokenizer.eos_id:
                if result is not None:
                    result.stop_reason = "eos"
                break

            ids.append(token_id)
            if result is not None:
                result.token_ids.append(token_id)

            delta = decoder.push(token_id)
            if delta:
                yield delta

            if config.stop_strings and any(s in decoder.emitted for s in config.stop_strings):
                if result is not None:
                    result.stop_reason = "stop_string"
                break

        tail = decoder.flush()
        if tail:
            yield tail
    finally:
        if result is not None:
            result.seconds = time.perf_counter() - started
            result.text = decoder.emitted
        if was_training:
            model.train()


def generate(
    model: nn.Module,
    tokenizer,
    prompt: str,
    config: GenerationConfig,
    trace: bool = False,
) -> GenerationResult:
    """Generate a continuation of `prompt` and return it with full metadata."""
    result = GenerationResult(
        prompt=prompt, text="", token_ids=[], prompt_tokens=0, prompt_truncated=False,
        stop_reason="max_new_tokens", seed=None, seconds=0.0,
    )
    for _ in stream_tokens(model, tokenizer, prompt, config, result=result, trace=trace):
        pass
    return result


def generate_greedy(
    model: nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    stop_strings: tuple[str, ...] = (),
    device: torch.device | None = None,   # kept for compatibility; the model's device is used
) -> str:
    """Always take the most likely next token. Deterministic; used for evaluation."""
    config = GenerationConfig(max_new_tokens=max_new_tokens, temperature=0.0,
                              top_k=None, stop_strings=stop_strings)
    return generate(model, tokenizer, prompt, config).text


@torch.no_grad()
def next_token_probability(model: nn.Module, tokenizer, prompt: str, continuation: str) -> float:
    """Probability the model assigns to the FIRST token of `continuation` right after `prompt`.

    Computed over the model's FULL output (all V rows), which is the same
    distribution the training loss is measured on.
    """
    device = next(model.parameters()).device
    prompt_ids = tokenizer.encode(prompt, allow_special=False)
    target_id = tokenizer.encode(prompt + continuation, allow_special=False)[len(prompt_ids)]
    was_training = model.training
    model.eval()
    window = torch.tensor([prompt_ids[-model.cfg.context_length:]], dtype=torch.long, device=device)
    logits, _ = model(window)
    probability = torch.softmax(logits[0, -1].float(), dim=-1)[target_id].item()
    if was_training:
        model.train()
    return probability

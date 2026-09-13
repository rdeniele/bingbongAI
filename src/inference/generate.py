"""Autoregressive text generation.

HOW THE MODEL WRITES, ONE TOKEN AT A TIME

    ids = encode(prompt)
    repeat:
        logits = model(ids[-context_length:])    (1, T, V)
        scores = logits[0, -1]                   (V,)  prediction for the NEXT token only
        next   = choose(scores)                  one token id
        ids.append(next)                         the choice becomes input for the next round

The model only ever predicts ONE token. Longer text is that single prediction
repeated, each time feeding the model its own previous output. Nothing plans
ahead: every word is chosen knowing only what came before it.

Only the logits at the LAST position are used. The others are the model's
predictions for tokens that already exist -- useful for training, irrelevant
for writing.

`context_length` caps what the model can see. Once the text is longer, only the
most recent 512 tokens are fed in and earlier ones are forgotten entirely.

STATUS
    Phase 7 needs only greedy decoding (always pick the highest-scoring token),
    which is deterministic and therefore right for measuring what the model has
    learned. Phase 8 adds temperature, top-k sampling and a seeded
    deterministic mode here.
"""

from __future__ import annotations

import torch
from torch import nn


@torch.no_grad()
def generate_greedy(
    model: nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    stop_strings: tuple[str, ...] = (),
    device: torch.device | None = None,
) -> str:
    """Continue `prompt` by always choosing the most likely next token.

    Returns only the newly generated text, not the prompt. Stops early once the
    generated text contains any of `stop_strings`, or at <|eos|>.
    """
    device = device or next(model.parameters()).device
    context_length = model.cfg.context_length
    was_training = model.training
    model.eval()

    ids = tokenizer.encode(prompt, allow_special=False)
    if not ids:
        ids = [tokenizer.bos_id]
    new_ids: list[int] = []

    for _ in range(max_new_tokens):
        window = torch.tensor([ids[-context_length:]], dtype=torch.long, device=device)  # (1, T)
        logits, _ = model(window)                                                        # (1, T, V)
        # Only ids the tokenizer can decode are eligible. The model's output layer
        # may be wider than the tokenizer (the synthetic tokenizer has 484 tokens,
        # the model 8192 output rows); an untrained model will happily score an
        # unused row highest, and that id has no text.
        scores = logits[0, -1, : tokenizer.vocab_size]                                   # (V_tok,)
        next_id = int(torch.argmax(scores))                                              # scalar
        if next_id == tokenizer.eos_id:
            break
        ids.append(next_id)
        new_ids.append(next_id)
        if stop_strings and any(s in tokenizer.decode(new_ids) for s in stop_strings):
            break

    if was_training:
        model.train()
    return tokenizer.decode(new_ids, skip_special=True)


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

"""Generate text with a trained BingBongAI checkpoint. Fully offline.

    python scripts/generate.py --prompt "The color of snow is"
    python scripts/generate.py                                   # interactive: type prompts
    python scripts/generate.py --prompt "A lion" --temperature 0.8 --top-k 40 --seed 7
    python scripts/generate.py --prompt "A lion" --greedy
    python scripts/generate.py --prompt "A lion" --explain       # show every token decision

Decoding modes
    --greedy            always the single most likely token. Deterministic.
    --temperature T     sample; T<1 conservative, T>1 adventurous. Default 0.8.
    --top-k K           sample only among the K most likely tokens. Default 40.
    --seed N            make sampling reproducible. Without it a seed is drawn and
                        PRINTED, so any output can be reproduced afterwards.

With --explain, each step prints the chosen token, the probability the model gave
it, the probability it had after temperature/top-k, and the model's top five
candidates -- the whole generation process, one token at a time.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from src.inference.generate import (  # noqa: E402
    GenerationConfig,
    GenerationResult,
    stream_tokens,
)
from src.inference.model_loader import load_for_inference  # noqa: E402
from src.training.checkpoint import CheckpointError  # noqa: E402

DEFAULT_CHECKPOINT = "checkpoints/synthetic/best.pt"


def safe_print(text: str = "", end: str = "\n") -> None:
    """Print without crashing on characters the Windows console cannot show."""
    try:
        print(text, end=end, flush=True)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "ascii"
        print(text.encode(encoding, errors="backslashreplace").decode(encoding), end=end, flush=True)


def show(text: str) -> str:
    """Make whitespace visible inside the --explain table."""
    return repr(text)[1:-1]


def run_once(loaded, prompt: str, config: GenerationConfig, explain: bool) -> None:
    result = GenerationResult(prompt=prompt, text="", token_ids=[], prompt_tokens=0,
                              prompt_truncated=False, stop_reason="max_new_tokens",
                              seed=None, seconds=0.0)
    safe_print("\nPrompt:")
    safe_print(prompt)
    safe_print("\nBingBongAI:")

    if explain:
        for _ in stream_tokens(loaded.model, loaded.tokenizer, prompt, config,
                               result=result, trace=True):
            pass
        safe_print(result.text)
        safe_print(f"\n  {'step':>4}  {'chosen':<14} {'p(model)':>9} {'p(sampled)':>11}   top candidates")
        for i, step in enumerate(result.steps, 1):
            alts = "  ".join(f"{show(t)!s}:{p:.2f}" for t, p in step.alternatives)
            safe_print(f"  {i:>4}  {show(step.text):<14} {step.model_probability:>9.3f} "
                       f"{step.sampling_probability:>11.3f}   {alts}")
    else:
        for piece in stream_tokens(loaded.model, loaded.tokenizer, prompt, config, result=result):
            safe_print(piece, end="")
        safe_print()

    details = [f"{len(result.token_ids)} tokens in {result.seconds:.2f}s",
               f"{result.tokens_per_second:.0f} tok/s",
               f"stop: {result.stop_reason}"]
    details.append("greedy" if result.seed is None else f"seed {result.seed}")
    if result.prompt_truncated:
        details.append(f"prompt truncated to last {loaded.model.cfg.context_length} tokens")
    safe_print(f"\n[{', '.join(details)}]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text with BingBongAI.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--prompt", default=None, help="omit for interactive mode")
    parser.add_argument("--max-new-tokens", type=int, default=60)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40, help="0 disables top-k")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--greedy", action="store_true", help="deterministic argmax decoding")
    parser.add_argument("--stop", action="append", default=[], help="stop after this text (repeatable)")
    parser.add_argument("--explain", action="store_true", help="show every token decision")
    parser.add_argument("--device", default="auto", help="auto | cuda | cpu")
    parser.add_argument("--tokenizer", default=None, help="override the tokenizer path")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    try:
        loaded = load_for_inference(args.checkpoint, device, tokenizer_path=args.tokenizer)
    except (CheckpointError, FileNotFoundError) as error:
        raise SystemExit(f"cannot load model: {error}") from None

    try:
        config = GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=0.0 if args.greedy else args.temperature,
            top_k=None if (args.greedy or args.top_k == 0) else args.top_k,
            seed=args.seed,
            stop_strings=tuple(args.stop),
        )
    except ValueError as error:
        raise SystemExit(f"invalid setting: {error}") from None

    model = loaded.model
    safe_print("=" * 56)
    safe_print("  BingBongAI - Generate  (offline)")
    safe_print("=" * 56)
    val = f", val loss {loaded.best_val_loss:.4f}" if loaded.best_val_loss is not None else ""
    safe_print(f"  Checkpoint: {loaded.checkpoint_path} ({loaded.config_name}, step {loaded.step}{val})")
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    safe_print(f"  Model:      {model.num_parameters():,} parameters | "
               f"tokenizer {loaded.tokenizer.vocab_size} tokens | {device_name}")
    if config.greedy:
        safe_print("  Decoding:   greedy (deterministic)")
    else:
        top_k = config.top_k if config.top_k is not None else "off"
        seed = config.seed if config.seed is not None else "random (printed after each output)"
        safe_print(f"  Decoding:   temperature {config.temperature}, top-k {top_k}, seed {seed}")

    if args.prompt is not None:
        run_once(loaded, args.prompt, config, args.explain)
        return

    safe_print("\n  Interactive mode. Enter a prompt; empty line or 'exit' quits.")
    while True:
        try:
            prompt = input("\n> ")
        except (EOFError, KeyboardInterrupt):
            safe_print()
            break
        if prompt.strip().lower() in {"", "exit", "quit"}:
            break
        run_once(loaded, prompt, config, args.explain)


if __name__ == "__main__":
    main()

"""Phase 8 experiment: what do temperature and top-k actually do to BingBongAI's output?

    python scripts/sampling_sweep.py

Runs on the trained synthetic checkpoint and MEASURES, for each decoding setting:

  accuracy     33 patterns x 3 seeds: does sampling still produce the right word?
  validity     of the complete lines generated freely, how many are real corpus
               sentences (vs. mixed-up hybrids like "The color of Japan is Tokyo.")
  diversity    how many distinct outputs 20 different seeds produce
  starts       how often generated sentences begin with "The" / "After" / "A",
               compared with the corpus's true frequencies

plus generation speed on GPU and CPU. Everything is seeded; results go to
experiments/phase8_sampling.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from src.inference.generate import GenerationConfig, generate  # noqa: E402
from src.inference.model_loader import load_for_inference  # noqa: E402
from src.training.synthetic import build_sentences, completion_cases  # noqa: E402

SETTINGS = [
    ("greedy", 0.0, None),
    ("T=0.5", 0.5, None),
    ("T=0.8, top-k 40", 0.8, 40),
    ("T=1.0", 1.0, None),
    ("T=1.5", 1.5, None),
    ("T=1.5, top-k 5", 1.5, 5),
    ("T=2.5", 2.5, None),
]
FREE_PROMPT = "The color of snow is"
FREE_SEEDS = range(20)
FREE_TOKENS = 48
ACCURACY_SEEDS = (0, 1, 2)


def measure_setting(loaded, temperature: float, top_k: int | None) -> dict:
    model, tok = loaded.model, loaded.tokenizer
    cases = completion_cases()
    corpus = set(build_sentences())
    greedy = temperature == 0

    # accuracy: the determined word, sampled, in context
    correct = trials = 0
    for seed in ([None] if greedy else ACCURACY_SEEDS):
        for i, case in enumerate(cases):
            prompt = cases[i - 1].sentence + "\n" + case.prefix
            out = generate(model, tok, prompt, GenerationConfig(
                max_new_tokens=8, temperature=temperature, top_k=top_k,
                seed=seed, stop_strings=(".", "\n"))).text
            correct += out.split(".")[0].split("\n")[0].strip() == case.answer
            trials += 1

    # free generation: validity, diversity, sentence starts
    outputs, lines = [], []
    for seed in ([None] if greedy else FREE_SEEDS):
        text = generate(model, tok, FREE_PROMPT, GenerationConfig(
            max_new_tokens=FREE_TOKENS, temperature=temperature, top_k=top_k, seed=seed)).text
        outputs.append(text)
        full = (FREE_PROMPT + text).split("\n")
        lines.extend(line for line in full[:-1] if line)       # drop the unfinished last line

    starts = Counter(line.split(" ")[0] for line in lines)
    total = max(1, len(lines))
    return {
        "accuracy": {"correct": correct, "trials": trials, "rate": correct / trials},
        "free_generation": {
            "samples": len(outputs),
            "distinct_outputs": len(set(outputs)),
            "complete_lines": len(lines),
            "valid_corpus_sentences": sum(line in corpus for line in lines),
            "validity_rate": sum(line in corpus for line in lines) / total,
            "invalid_examples": sorted({line for line in lines if line not in corpus})[:6],
            "start_word_share": {w: starts[w] / total for w in ("The", "After", "A")},
            "example": outputs[0],
        },
    }


def measure_speed(checkpoint: str, device: torch.device, tokens: int = 60, repeats: int = 3) -> dict:
    loaded = load_for_inference(checkpoint, device)
    config = GenerationConfig(max_new_tokens=tokens, temperature=0.0)
    generate(loaded.model, loaded.tokenizer, "The color of", GenerationConfig(max_new_tokens=5, temperature=0.0))
    rates = []
    for _ in range(repeats):
        r = generate(loaded.model, loaded.tokenizer, "The color of", config)
        rates.append(r.tokens_per_second)
    return {"device": str(device), "tokens": tokens, "repeats": repeats,
            "tokens_per_second": rates, "mean_tokens_per_second": sum(rates) / len(rates)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure the effect of decoding settings.")
    parser.add_argument("--checkpoint", default="checkpoints/synthetic/best.pt")
    parser.add_argument("--out", default="experiments/phase8_sampling.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = load_for_inference(args.checkpoint, device)
    print(f"Checkpoint {args.checkpoint}: step {loaded.step}, val {loaded.best_val_loss}")

    sentences = build_sentences()
    corpus_starts = Counter(s.split(" ")[0] for s in sentences)
    corpus_share = {w: corpus_starts[w] / len(sentences) for w in ("The", "After", "A")}

    started = time.perf_counter()
    results = {}
    for name, temperature, top_k in SETTINGS:
        r = measure_setting(loaded, temperature, top_k)
        results[name] = {"temperature": temperature, "top_k": top_k, **r}
        f = r["free_generation"]
        s = f["start_word_share"]
        print(f"  {name:<17} accuracy {r['accuracy']['correct']:>3}/{r['accuracy']['trials']:<3} "
              f"valid lines {f['valid_corpus_sentences']:>3}/{f['complete_lines']:<3} "
              f"distinct {f['distinct_outputs']:>2}/{f['samples']:<2} "
              f"starts The {s['The']:.2f} After {s['After']:.2f} A {s['A']:.2f}")
    print(f"  corpus start shares:  The {corpus_share['The']:.3f} After {corpus_share['After']:.3f} "
          f"A {corpus_share['A']:.3f}")
    sweep_seconds = time.perf_counter() - started

    speed = [measure_speed(args.checkpoint, torch.device("cpu"))]
    if torch.cuda.is_available():
        speed.insert(0, measure_speed(args.checkpoint, torch.device("cuda")))
    for s in speed:
        print(f"  speed on {s['device']}: {s['mean_tokens_per_second']:.1f} tok/s "
              f"(runs: {', '.join(f'{x:.1f}' for x in s['tokens_per_second'])})")

    record = {
        "experiment": "phase8_sampling",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "checkpoint": args.checkpoint,
        "checkpoint_step": loaded.step,
        "torch_version": str(torch.__version__),
        "free_prompt": FREE_PROMPT,
        "free_tokens": FREE_TOKENS,
        "corpus_start_word_share": corpus_share,
        "settings": results,
        "speed": speed,
        "sweep_seconds": sweep_seconds,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"Record: {out}")


if __name__ == "__main__":
    main()

"""Phase 7: prove that BingBongAI learns, starting from random weights.

    python scripts/prove_learning.py --config configs/synthetic.yaml

Builds the model from a fixed seed, measures it BEFORE training, trains it,
measures it AFTER, and writes every number to experiments/.

WHAT IS MEASURED, BEFORE AND AFTER
  1. Validation loss.
  2. Completion accuracy on all 33 patterns. Each prompt stops right before a
     word that is fully determined ("The color of snow is" -> "white"). The model
     writes greedily; the answer counts only if it matches exactly.
       - "bare":        the prompt alone, at the very start of the context
       - "in context":  the prompt after one other sentence and a newline, the
                        way sentences actually appear in the training stream
  3. The probability assigned to the correct answer's first token.
  4. Free-running greedy text from a few prompts.

HOW TO READ THE LOSS
  Sentence ORDER in the corpus is random, so the start of each sentence cannot be
  predicted even by a perfect model. The loss therefore has a floor above zero.
  The script prints an estimate of that floor so the curve can be judged against
  it rather than against zero.

This run starts fresh every time and replaces its own previous outputs
(checkpoints/<name>/ and runs/<name>/).
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from scripts.train import build_trainer, resolve_device  # noqa: E402
from src.inference.generate import generate_greedy, next_token_probability  # noqa: E402
from src.training.synthetic import completion_cases, estimate_order_entropy_floor  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logging import RunLogger  # noqa: E402

SAMPLE_PROMPTS = ["The color of snow is", "After three comes", "The capital of Japan is", "A lion"]


def evaluate_completions(model, tokenizer) -> dict:
    cases = completion_cases()
    results = []
    for index, case in enumerate(cases):
        # Context: the previous pattern in the list, then a newline -- a real sentence
        # from the corpus, but never the one being tested.
        context = cases[index - 1].sentence + "\n"
        row = {"category": case.category, "prefix": case.prefix, "answer": case.answer}
        for condition, prompt in (("bare", case.prefix), ("in_context", context + case.prefix)):
            generated = generate_greedy(model, tokenizer, prompt, max_new_tokens=8,
                                        stop_strings=(".", "\n"))
            predicted = generated.split(".")[0].split("\n")[0].strip()
            row[condition] = {
                "generated": generated,
                "predicted": predicted,
                "correct": predicted == case.answer,
                "answer_probability": next_token_probability(model, tokenizer, prompt, " " + case.answer),
            }
        results.append(row)

    def summarise(condition: str) -> dict:
        rows = [r[condition] for r in results]
        return {
            "correct": sum(r["correct"] for r in rows),
            "total": len(rows),
            "mean_answer_probability": sum(r["answer_probability"] for r in rows) / len(rows),
        }

    return {"bare": summarise("bare"), "in_context": summarise("in_context"), "cases": results}


def free_samples(model, tokenizer) -> dict[str, str]:
    return {p: generate_greedy(model, tokenizer, p, max_new_tokens=30) for p in SAMPLE_PROMPTS}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prove BingBongAI learns from random initialization.")
    parser.add_argument("--config", default="configs/synthetic.yaml")
    parser.add_argument("--out", default="experiments/phase7_learning_proof.json")
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(config.training.get("device", "auto"))

    run_dir = Path(config.paths.get("run_dir", "runs")) / config.name
    checkpoint_dir = Path(config.paths.checkpoint_dir) / config.name
    for stale in (run_dir, checkpoint_dir):
        if stale.exists():
            shutil.rmtree(stale)

    logger = RunLogger(run_dir)
    trainer, tokenizer, dataset_tokens = build_trainer(config, device, logger)
    model = trainer.model
    floor = estimate_order_entropy_floor(tokenizer)
    uniform = math.log(model.cfg.vocab_size)

    logger.info("=" * 72)
    logger.info("  Phase 7 - Does BingBongAI learn?")
    logger.info("=" * 72)

    # ---------------- BEFORE --------------------------------------------------
    torch.manual_seed(0)
    before_val = trainer.evaluate()
    before = evaluate_completions(model, tokenizer)
    before_samples = free_samples(model, tokenizer)
    logger.info(f"  BEFORE training (random weights, seed {trainer.settings.seed})")
    logger.info(f"    val loss:                 {before_val:.4f}   (ln V = {uniform:.4f})")
    logger.info(f"    completions, bare:        {before['bare']['correct']}/{before['bare']['total']}")
    logger.info(f"    completions, in context:  {before['in_context']['correct']}/{before['in_context']['total']}")
    logger.info(f"    mean p(correct answer):   {before['in_context']['mean_answer_probability']:.6f}")
    logger.info(f"    sample: 'The color of snow is' -> {before_samples['The color of snow is']!r}")
    logger.info("")

    # ---------------- TRAIN ---------------------------------------------------
    trainer.describe(dataset_tokens)
    logger.info(f"  Estimated loss floor from random sentence order: ~{floor:.3f} (see synthetic.py)")
    logger.info("")
    started = time.perf_counter()
    summary = trainer.fit()
    train_seconds = time.perf_counter() - started
    peak_vram = (torch.cuda.max_memory_reserved(device) / 1024**2) if device.type == "cuda" else None

    # ---------------- AFTER ---------------------------------------------------
    after_val = trainer.evaluate()
    after = evaluate_completions(model, tokenizer)
    after_samples = free_samples(model, tokenizer)

    # Write the record FIRST, so a display problem below can never lose the results.
    record = {
        "experiment": "phase7_learning_proof",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": args.config,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "torch_version": str(torch.__version__),
        "parameters": model.num_parameters(),
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "model_vocab_size": model.cfg.vocab_size,
        "dataset_tokens": dataset_tokens,
        "settings": trainer.settings.__dict__,
        "uniform_loss_ln_v": uniform,
        "estimated_order_entropy_floor": floor,
        "train_seconds": train_seconds,
        "peak_vram_reserved_mib": peak_vram,
        "summary": summary,
        "val_loss_history": trainer.eval_history,
        "train_loss_every_step": trainer.loss_history,
        "before": {"val_loss": before_val, "completions": before, "samples": before_samples},
        "after": {"val_loss": after_val, "completions": after, "samples": after_samples},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2), encoding="utf-8")

    logger.info("")
    logger.info("=" * 72)
    logger.info("  RESULT")
    logger.info("=" * 72)
    logger.info(f"  {'':<30}{'before':>14}{'after':>14}")
    logger.info(f"  {'validation loss':<30}{before_val:>14.4f}{after_val:>14.4f}")
    for cond, label in (("bare", "completions (bare)"), ("in_context", "completions (in context)")):
        b, a = before[cond], after[cond]
        logger.info(f"  {label:<30}{b['correct']:>11}/{b['total']}{a['correct']:>11}/{a['total']}")
    logger.info(f"  {'mean p(answer), in context':<30}"
                f"{before['in_context']['mean_answer_probability']:>14.6f}"
                f"{after['in_context']['mean_answer_probability']:>14.6f}")
    logger.info(f"  Loss floor estimate ~{floor:.3f}.  Training took {train_seconds:.1f}s "
                f"for {summary['steps_this_session']} steps.")

    logger.info("\n  Per pattern (in context):")
    for b_row, a_row in zip(before["cases"], after["cases"]):
        mark = "ok " if a_row["in_context"]["correct"] else "MISS"
        logger.info(f"    [{mark}] {a_row['prefix']:<34} -> {a_row['in_context']['predicted']!r:<12}"
                    f" p={a_row['in_context']['answer_probability']:.3f}"
                    f"   (before: {b_row['in_context']['predicted']!r})")

    wrong_bare = [r for r in after["cases"] if not r["bare"]["correct"]]
    if wrong_bare:
        logger.info("\n  Bare-prompt misses:")
        for r in wrong_bare:
            logger.info(f"    {r['prefix']!r} -> {r['bare']['generated']!r} (expected {r['answer']!r})")

    logger.info("\n  Free generation, greedy, 30 tokens:")
    for prompt in SAMPLE_PROMPTS:
        logger.info(f"    {prompt!r}")
        logger.info(f"      before: {before_samples[prompt]!r}")
        logger.info(f"      after:  {after_samples[prompt]!r}")

    logger.info(f"\n  Full record: {out}")
    logger.close()


if __name__ == "__main__":
    main()

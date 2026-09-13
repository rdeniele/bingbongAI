"""Generate the tiny synthetic dataset used to prove the model can learn.

    python scripts/make_synthetic.py

This is NOT a corpus for building an intelligent model. It is a controlled
experiment (TRAINING.md, "The Phase-7 experiment"): a handful of rigid patterns
that a working model MUST be able to memorise. If training cannot learn these,
there is a bug in the model or the optimizer, and no amount of real data would
fix it.

The patterns themselves live in src/training/synthetic.py, so the evaluation in
scripts/prove_learning.py tests exactly what was written here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.synthetic import build_corpus, build_sentences  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the synthetic learning-proof dataset.")
    parser.add_argument("--out", default="data/raw/synthetic/patterns.txt", help="output file path")
    parser.add_argument("--repeats", type=int, default=200,
                        help="how many shuffled passes over the sentence set to write")
    parser.add_argument("--seed", type=int, default=1337, help="shuffle seed, for reproducibility")
    args = parser.parse_args()

    text = build_corpus(args.repeats, args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")

    print(f"Wrote {out_path}")
    print(f"  unique sentences: {len(build_sentences())}")
    print(f"  repeats:          {args.repeats}")
    print(f"  total lines:      {text.count(chr(10)):,}")
    print(f"  characters:       {len(text):,}")
    print(f"  seed:             {args.seed}")


if __name__ == "__main__":
    main()

"""Train BingBongAI's tokenizer on a corpus of text files.

    python scripts/train_tokenizer.py --config configs/small.yaml

Reads every .txt/.md file under the config's `data.corpus_dir`, learns BPE
merges from them, saves the tokenizer, and prints a round-trip check plus real
compression statistics measured on that corpus.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running this script directly, without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tokenizer.tokenizer import BPETokenizer  # noqa: E402
from src.tokenizer.train_tokenizer import train_tokenizer  # noqa: E402
from src.training.dataset import read_text_files  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the BingBongAI tokenizer.")
    parser.add_argument("--config", required=True, help="path to a YAML config")
    parser.add_argument(
        "--corpus",
        default=None,
        help="override the corpus directory from the config",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    corpus_dir = Path(args.corpus or config.data.get("corpus_dir", "data/raw"))
    vocab_size = config.tokenizer.vocab_size
    special_tokens = config.tokenizer.special_tokens.to_dict()
    out_dir = Path(config.tokenizer.vocab_path)
    min_frequency = config.tokenizer.get("min_merge_frequency", 2)

    print("=" * 60)
    print("  BingBongAI - Tokenizer Training")
    print("=" * 60)
    print(f"  Config:       {args.config}")
    print(f"  Corpus:       {corpus_dir}")
    print(f"  Target vocab: {vocab_size}")
    print(f"  Specials:     {', '.join(sorted(special_tokens))}")
    print()

    try:
        documents = read_text_files(corpus_dir)
    except FileNotFoundError as error:
        raise SystemExit(str(error)) from None
    texts = [text for _, text in documents]
    total_chars = sum(len(t) for t in texts)
    total_bytes = sum(len(t.encode("utf-8")) for t in texts)

    print(f"  Documents:    {len(documents)}")
    for path, text in documents:
        print(f"    {path.as_posix():<50} {len(text):>9,} chars")
    print(f"  Total:        {total_chars:,} characters / {total_bytes:,} bytes")
    print()

    tokenizer = train_tokenizer(
        texts,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=min_frequency,
    )

    path = tokenizer.save(out_dir)
    print()
    print(f"  Saved to:     {path}")
    print(f"  Vocab size:   {tokenizer.vocab_size} "
          f"(256 bytes + {tokenizer.num_merges} merges + {len(special_tokens)} specials)")
    print(f"  Fingerprint:  {tokenizer.fingerprint()}")

    # -- measured compression, on the actual corpus -----------------------
    total_tokens = sum(len(tokenizer.encode(text)) for text in texts)
    print()
    print("  Measured on this corpus:")
    print(f"    tokens:            {total_tokens:,}")
    print(f"    bytes per token:   {total_bytes / total_tokens:.2f}")
    print(f"    chars per token:   {total_chars / total_tokens:.2f}")
    print(f"    vs raw bytes:      {total_bytes / total_tokens:.2f}x shorter")

    # -- round trip check --------------------------------------------------
    reloaded = BPETokenizer.load(out_dir)
    failures = 0
    for doc_path, text in documents:
        if reloaded.decode(reloaded.encode(text)) != text:
            print(f"    ROUND TRIP FAILED on {doc_path}")
            failures += 1
    if failures:
        raise SystemExit(f"\n  {failures} document(s) failed the round trip. Not usable.")
    print(f"    round trip:        exact on all {len(documents)} documents")
    print()


if __name__ == "__main__":
    main()

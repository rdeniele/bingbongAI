"""Tokenize a corpus and write the train/validation token files.

    python scripts/prepare_data.py --config configs/synthetic.yaml

Requires a trained tokenizer at the config's `tokenizer.vocab_path`
(see scripts/train_tokenizer.py). Writes:

    data.train_path   flat uint16 token ids
    data.val_path     flat uint16 token ids
    data.meta_path    counts, source files, and the tokenizer fingerprint

The fingerprint in the metadata is checked again at training time, so token
files can never be silently paired with a different tokenizer than the one that
produced them.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tokenizer.tokenizer import BPETokenizer  # noqa: E402
from src.training.dataset import (  # noqa: E402
    read_text_files,
    tokenize_documents,
    write_meta,
    write_token_file,
)
from src.utils.config import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare BingBongAI training data.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    data = config.data
    context_length = config.model.context_length

    tokenizer = BPETokenizer.load(config.tokenizer.vocab_path)
    if tokenizer.vocab_size > config.tokenizer.vocab_size:
        raise SystemExit(
            f"tokenizer has {tokenizer.vocab_size} tokens but the model's vocab_size is "
            f"{config.tokenizer.vocab_size}; the model could not embed every token"
        )

    print("=" * 64)
    print("  BingBongAI - Data Preparation")
    print("=" * 64)
    print(f"  Config:     {args.config}")
    print(f"  Corpus:     {data.corpus_dir}")
    print(f"  Tokenizer:  {config.tokenizer.vocab_path}  "
          f"(vocab {tokenizer.vocab_size}, fingerprint {tokenizer.fingerprint()})")

    started = time.perf_counter()
    documents = read_text_files(data.corpus_dir)
    corpus = tokenize_documents(documents, tokenizer, val_fraction=data.val_split)
    elapsed = time.perf_counter() - started

    for doc in corpus.per_document:
        print(f"    {doc['path']:<44} {doc['tokens']:>10,} tokens")

    n_train, n_val = len(corpus.train_ids), len(corpus.val_ids)
    window = context_length + 1
    for name, count in (("train", n_train), ("validation", n_val)):
        if count < window:
            raise SystemExit(
                f"\n  The {name} split has {count:,} tokens, fewer than one window of "
                f"{window}. Add more text, lower val_split, or shorten context_length."
            )

    write_token_file(data.train_path, corpus.train_ids)
    write_token_file(data.val_path, corpus.val_ids)

    total_chars = sum(len(text) for _, text in documents)
    meta = {
        "config": args.config,
        "tokenizer_path": config.tokenizer.vocab_path,
        "tokenizer_fingerprint": tokenizer.fingerprint(),
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "dtype": "uint16",
        "val_split": data.val_split,
        "train_tokens": n_train,
        "val_tokens": n_val,
        "total_characters": total_chars,
        "documents": corpus.per_document,
    }
    write_meta(data.meta_path, meta)

    print()
    print(f"  Train tokens:       {n_train:>10,}  -> {data.train_path}")
    print(f"  Validation tokens:  {n_val:>10,}  -> {data.val_path}")
    print(f"  Chars per token:    {total_chars / (n_train + n_val):>10.2f}")
    print(f"  Windows of {context_length} tokens: {n_train - context_length:,} train start "
          f"positions, {n_val - context_length:,} validation")
    print(f"  Took {elapsed:.2f}s. Metadata -> {data.meta_path}")
    print()


if __name__ == "__main__":
    main()

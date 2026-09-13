"""Turning a folder of text files into training data.

THE PIPELINE
    text files
      -> tokenizer.encode()            each document becomes a list of ids
      -> + <|eos|> after each document  so the model sees where documents end
      -> split each document: first 90% to train, last 10% to validation
      -> concatenate into one long stream per split
      -> write as a flat binary file of uint16 token ids

WHY ONE LONG STREAM
    The model trains on fixed windows of `context_length` tokens. Rather than
    padding every document to that length (wasting compute on padding), all
    documents are joined into one stream and windows are cut from anywhere in
    it. A window may cross a document boundary; the <|eos|> token between them
    is how the model learns that what follows is unrelated.

WHY SPLIT INSIDE EACH DOCUMENT
    Splitting the concatenated stream at 90% would put entire late documents in
    validation and none of them in training. Splitting each document keeps every
    source represented in both splits.

    Be clear about what validation means for the SYNTHETIC corpus: the same 33
    sentences appear in both splits, so validation loss there measures
    memorisation, not generalisation. On a real corpus it measures
    generalisation to unseen text.

WHY uint16
    Token ids are below 65,536 (vocabulary is 8,192), so two bytes each is
    enough -- half the disk and RAM of int32. Ids are widened to int64 only when
    a batch is built, because that is what the embedding lookup requires.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

TOKEN_DTYPE = np.uint16
MAX_STORABLE_VOCAB = np.iinfo(TOKEN_DTYPE).max + 1   # 65,536
TEXT_SUFFIXES = {".txt", ".md"}


def read_text_files(directory: str | Path) -> list[tuple[Path, str]]:
    """Every non-empty .txt/.md file under `directory`, recursively, sorted by path."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"corpus directory not found: {directory}")

    documents: list[tuple[Path, str]] = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8", errors="replace")
            if text.strip():
                documents.append((path, text))

    if not documents:
        raise FileNotFoundError(f"no non-empty .txt or .md files found under {directory}")
    return documents


@dataclass
class TokenizedCorpus:
    train_ids: list[int] = field(default_factory=list)
    val_ids: list[int] = field(default_factory=list)
    per_document: list[dict] = field(default_factory=list)


def tokenize_documents(
    documents: list[tuple[Path, str]], tokenizer, val_fraction: float
) -> TokenizedCorpus:
    """Encode documents and split each one into train and validation parts."""
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be between 0 and 1, got {val_fraction}")
    if tokenizer.vocab_size > MAX_STORABLE_VOCAB:
        raise ValueError(
            f"vocab size {tokenizer.vocab_size} does not fit in {TOKEN_DTYPE.__name__}"
        )

    corpus = TokenizedCorpus()
    for path, text in documents:
        # allow_special=False: a corpus file that happens to contain the text
        # "<|eos|>" must not be able to inject a real control token.
        ids = tokenizer.encode(text, allow_special=False) + [tokenizer.eos_id]
        split = len(ids) - int(len(ids) * val_fraction)
        corpus.train_ids.extend(ids[:split])
        corpus.val_ids.extend(ids[split:])
        corpus.per_document.append(
            {"path": path.as_posix(), "tokens": len(ids),
             "train_tokens": split, "val_tokens": len(ids) - split}
        )
    return corpus


def write_token_file(path: str | Path, ids: list[int]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(ids, dtype=TOKEN_DTYPE).tofile(path)


def write_meta(path: str | Path, meta: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def read_meta(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"dataset metadata not found at {path}. Prepare the data first:\n"
            f"  python scripts/prepare_data.py --config <config>"
        )
    return json.loads(path.read_text(encoding="utf-8"))


class TokenDataset:
    """A flat stream of token ids, sliced into (input, target) windows.

    For a window starting at position s:

        input  x = tokens[s     : s + T    ]      (T,)
        target y = tokens[s + 1 : s + T + 1]      (T,)

    y is x shifted left by one. At every position t, the target is simply the
    token that actually came next. That shift IS next-token prediction; there is
    no other labelling.

    The file is read fully into memory. At uint16 that is 2 bytes per token, so
    even a 200M-token corpus is 400 MiB -- fine with this machine's 23 GiB RAM.
    A memory-mapped file is the upgrade if a corpus ever outgrows RAM.
    """

    def __init__(self, path: str | Path, context_length: int) -> None:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(
                f"token file not found: {path}. Run scripts/prepare_data.py first."
            )
        self.path = path
        self.context_length = context_length
        self.tokens = np.fromfile(path, dtype=TOKEN_DTYPE)

        if len(self.tokens) < context_length + 1:
            raise ValueError(
                f"{path} holds {len(self.tokens):,} tokens, fewer than the "
                f"{context_length + 1} needed for one window of context_length "
                f"{context_length}. Use more text or a shorter context."
            )

    @property
    def num_tokens(self) -> int:
        return int(len(self.tokens))

    def __len__(self) -> int:
        """Number of distinct window start positions."""
        return self.num_tokens - self.context_length

    def windows(self, starts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather a batch of windows.

        Args:
            starts: (B,) int64 start positions, each in [0, len(self))
        Returns:
            x: (B, T) int64,  y: (B, T) int64
        """
        offsets = np.arange(self.context_length + 1)                   # (T+1,)
        index = starts.numpy()[:, None] + offsets[None, :]              # (B, T+1)
        chunk = torch.from_numpy(self.tokens[index].astype(np.int64))   # (B, T+1)
        return chunk[:, :-1], chunk[:, 1:]

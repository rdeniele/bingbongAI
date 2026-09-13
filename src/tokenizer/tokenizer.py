"""BingBongAI's tokenizer: byte-level Byte Pair Encoding, implemented from scratch.

WHY THIS DESIGN
---------------
A language model cannot read text. It reads integers. The tokenizer is the
bridge, and the choice of bridge has consequences that follow the model forever
(the vocabulary is baked into the embedding table, so it cannot be changed
without retraining).

Three obvious options, and why byte-level BPE wins for this project:

  Character-level
      Trivial to build. But sequences get very long -- "transformer" costs 11
      positions -- and attention cost grows with the SQUARE of sequence length.
      A 512-token context would hold about two sentences.

  Word-level
      Short sequences, but the vocabulary is unbounded. Every unseen word,
      typo, or bit of code becomes <|unk|> and the information is destroyed.

  Byte-level BPE  <-- our choice
      Start from the 256 possible byte values, then repeatedly merge the most
      frequent adjacent pair into a new token. Common words collapse into one
      token; rare words degrade gracefully into pieces rather than vanishing.

The decisive property is that byte-level BPE is LOSSLESS AND CLOSED. Every
possible byte is already in the base vocabulary, so any input -- emoji, Tagalog,
Python source, a corrupted file -- encodes and decodes exactly. There is no
input it cannot represent.

ON THE UNKNOWN TOKEN
--------------------
`<|unk|>` exists in the vocabulary because the project spec calls for an unknown
mechanism, and having the id reserved keeps the vocabulary layout stable. But it
is worth being precise: with a byte-level base vocabulary, `encode()` can never
emit it. Every byte from 0 to 255 has an id, and all text is bytes. That is not a
gap in the implementation -- it is the guarantee the design buys us.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

# Pre-tokenization pattern.
#
# Before learning merges we chop text into chunks, and merges are only ever
# learned or applied WITHIN a chunk. This stops BPE from inventing tokens that
# straddle a word boundary -- without it, a frequent bigram like "of the" could
# become one token, which wastes vocabulary and generalises badly.
#
# The leading " ?" on several branches attaches a space to the front of a word,
# so " the" is one token rather than " " + "the". Most words in running text are
# space-prefixed, so this roughly halves the token count.
#
# The alternatives are ordered and together cover every character:
#   contractions | letters | digits | punctuation | trailing space | whitespace
# `tests/test_tokenizer.py` asserts that coverage is total -- if this pattern
# ever dropped a character, encode/decode would silently stop being lossless.
_PRETOKEN_PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?[^\W\d]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+""",
    re.UNICODE,
)

# Token ids 0..255 are the raw byte values. Merges start here.
BYTE_VOCAB_SIZE = 256

SAVE_FORMAT_VERSION = 1


class BPETokenizer:
    """Encode text to token ids and back.

    Attributes:
        merges: ordered list of (left_id, right_id) pairs. Position in the list
            is the merge's rank; lower rank means it was learned earlier, which
            means it was more frequent, which means it applies first.
        special_tokens: name -> literal string, e.g. {"bos": "<|bos|>"}.
    """

    def __init__(
        self,
        merges: list[tuple[int, int]],
        special_tokens: dict[str, str] | None = None,
    ) -> None:
        self.merges = [tuple(pair) for pair in merges]
        self.special_tokens = dict(special_tokens or {})

        # rank lookup: pair -> the order it was learned in. Used by encode().
        self._merge_ranks: dict[tuple[int, int], int] = {
            pair: rank for rank, pair in enumerate(self.merges)
        }

        # id -> bytes, for every non-special token.
        # Base bytes first, then each merge is the concatenation of its parts.
        self._id_to_bytes: dict[int, bytes] = {i: bytes([i]) for i in range(BYTE_VOCAB_SIZE)}
        for rank, (left, right) in enumerate(self.merges):
            self._id_to_bytes[BYTE_VOCAB_SIZE + rank] = (
                self._id_to_bytes[left] + self._id_to_bytes[right]
            )

        # Special tokens occupy the highest ids, AFTER the merges. Placing them
        # last means adding a special token never renumbers an existing token.
        first_special_id = BYTE_VOCAB_SIZE + len(self.merges)
        self._special_to_id: dict[str, int] = {}
        self._id_to_special: dict[int, str] = {}
        for offset, name in enumerate(sorted(self.special_tokens)):
            literal = self.special_tokens[name]
            token_id = first_special_id + offset
            self._special_to_id[literal] = token_id
            self._id_to_special[token_id] = literal

        self._vocab_size = first_special_id + len(self.special_tokens)

        # Matches any special token literal, so encode() can split them out
        # before byte-level processing. Longest-first avoids a prefix of one
        # special token shadowing another.
        if self._special_to_id:
            escaped = sorted((re.escape(s) for s in self._special_to_id), key=len, reverse=True)
            self._special_pattern: re.Pattern[str] | None = re.compile(f"({'|'.join(escaped)})")
        else:
            self._special_pattern = None

    # -- properties ------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """Total number of distinct token ids: 256 bytes + merges + specials."""
        return self._vocab_size

    @property
    def num_merges(self) -> int:
        return len(self.merges)

    def token_id(self, special_name: str) -> int:
        """Id of a special token by name, e.g. `token_id("bos")`."""
        if special_name not in self.special_tokens:
            raise KeyError(
                f"no special token named {special_name!r}; "
                f"have {sorted(self.special_tokens)}"
            )
        return self._special_to_id[self.special_tokens[special_name]]

    @property
    def pad_id(self) -> int:
        return self.token_id("pad")

    @property
    def bos_id(self) -> int:
        return self.token_id("bos")

    @property
    def eos_id(self) -> int:
        return self.token_id("eos")

    @property
    def unk_id(self) -> int:
        """Reserved, but unreachable from encode() -- see the module docstring."""
        return self.token_id("unk")

    # -- encoding --------------------------------------------------------

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
        allow_special: bool = True,
    ) -> list[int]:
        """Turn text into token ids.

        Args:
            text: input string.
            add_bos / add_eos: wrap the output in begin/end-of-sequence tokens.
            allow_special: if True, literal special-token strings appearing in
                `text` are recognised and become their single reserved id. Set
                False for untrusted input, so that a user typing "<|eos|>" gets
                encoded as those literal characters rather than a control token.

        Returns:
            A list of ids, each in [0, vocab_size).
        """
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)

        if allow_special and self._special_pattern is not None:
            for piece in self._special_pattern.split(text):
                if not piece:
                    continue
                if piece in self._special_to_id:
                    ids.append(self._special_to_id[piece])
                else:
                    ids.extend(self._encode_ordinary(piece))
        else:
            ids.extend(self._encode_ordinary(text))

        if add_eos:
            ids.append(self.eos_id)
        return ids

    def _encode_ordinary(self, text: str) -> list[int]:
        """Encode text containing no special tokens."""
        ids: list[int] = []
        for chunk in _PRETOKEN_PATTERN.findall(text):
            ids.extend(self._encode_chunk(chunk.encode("utf-8")))
        return ids

    def _encode_chunk(self, raw: bytes) -> list[int]:
        """Apply the learned merges to one pre-token's bytes.

        Merges must be applied in the order they were LEARNED, not greedily by
        length. Repeatedly find the lowest-ranked mergeable pair present and
        merge every occurrence of it, until no learned pair remains.
        """
        ids = list(raw)
        if len(ids) < 2:
            return ids

        while True:
            best_pair: tuple[int, int] | None = None
            best_rank = len(self.merges)
            for pair in zip(ids, ids[1:]):
                rank = self._merge_ranks.get(pair)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_pair = pair

            if best_pair is None:
                return ids

            new_id = BYTE_VOCAB_SIZE + best_rank
            merged: list[int] = []
            index = 0
            while index < len(ids):
                if (
                    index < len(ids) - 1
                    and ids[index] == best_pair[0]
                    and ids[index + 1] == best_pair[1]
                ):
                    merged.append(new_id)
                    index += 2
                else:
                    merged.append(ids[index])
                    index += 1
            ids = merged

    # -- decoding --------------------------------------------------------

    def decode(self, ids: Iterable[int], skip_special: bool = False) -> str:
        """Turn token ids back into text.

        Every token maps to a byte string; concatenate them and decode as UTF-8.

        `errors="replace"` matters: a *generating* model can emit an id sequence
        that is a partial multi-byte character (half an emoji). Raising there
        would crash generation, so an invalid fragment becomes U+FFFD instead.
        For any id sequence produced by `encode()`, the round trip is exact.
        """
        pieces: list[bytes] = []
        for token_id in ids:
            token_id = int(token_id)
            if token_id in self._id_to_special:
                if not skip_special:
                    pieces.append(self._id_to_special[token_id].encode("utf-8"))
                continue
            try:
                pieces.append(self._id_to_bytes[token_id])
            except KeyError:
                raise ValueError(
                    f"token id {token_id} is outside this tokenizer's vocabulary "
                    f"of size {self.vocab_size}"
                ) from None
        return b"".join(pieces).decode("utf-8", errors="replace")

    # -- persistence -----------------------------------------------------

    def save(self, directory: str | Path) -> Path:
        """Write the tokenizer to `directory/tokenizer.json`."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "tokenizer.json"

        payload = {
            "format_version": SAVE_FORMAT_VERSION,
            "type": "byte_level_bpe",
            "vocab_size": self.vocab_size,
            "byte_vocab_size": BYTE_VOCAB_SIZE,
            "special_tokens": self.special_tokens,
            "merges": [[left, right] for left, right in self.merges],
        }
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        return path

    @classmethod
    def load(cls, directory: str | Path) -> "BPETokenizer":
        """Load a tokenizer previously written by `save()`."""
        directory = Path(directory)
        path = directory / "tokenizer.json" if directory.is_dir() else directory
        if not path.is_file():
            raise FileNotFoundError(
                f"no tokenizer at {path}. Train one first:\n"
                f"  python scripts/train_tokenizer.py --config configs/small.yaml"
            )

        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        version = payload.get("format_version")
        if version != SAVE_FORMAT_VERSION:
            raise ValueError(
                f"tokenizer at {path} has format version {version}, "
                f"but this code reads version {SAVE_FORMAT_VERSION}"
            )

        return cls(
            merges=[(left, right) for left, right in payload["merges"]],
            special_tokens=payload["special_tokens"],
        )

    def fingerprint(self) -> str:
        """A short stable hash of the vocabulary.

        Stored in every checkpoint. Loading a checkpoint against a tokenizer
        with a different fingerprint means the embedding rows no longer mean
        what they meant during training -- the model would emit fluent-looking
        nonsense. Cheap check, catches a genuinely baffling class of bug.
        """
        digest = hashlib.sha256()
        digest.update(f"v{SAVE_FORMAT_VERSION}".encode("utf-8"))
        for left, right in self.merges:
            digest.update(f"{left},{right};".encode("utf-8"))
        for name in sorted(self.special_tokens):
            digest.update(f"{name}={self.special_tokens[name]};".encode("utf-8"))
        return digest.hexdigest()[:16]

    def __repr__(self) -> str:
        return (
            f"BPETokenizer(vocab_size={self.vocab_size}, merges={self.num_merges}, "
            f"specials={sorted(self.special_tokens)}, fingerprint={self.fingerprint()})"
        )

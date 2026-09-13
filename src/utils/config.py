"""Configuration loading for BingBongAI.

Every model and training hyperparameter lives in a YAML file under `configs/`.
Nothing in `src/` hardcodes a hyperparameter -- code reads it from a Config.

The Config object is a thin, attribute-accessible wrapper over nested dicts, so
`cfg.model.embedding_dim` works while the underlying data stays plain and
serialisable (important: the whole config gets written into every checkpoint).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when a config file is missing a required value or is inconsistent."""


class Config:
    """Attribute access over a nested dict loaded from YAML.

    >>> cfg = Config({"model": {"embedding_dim": 384}})
    >>> cfg.model.embedding_dim
    384
    """

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    def __getattr__(self, name: str) -> Any:
        try:
            value = self._data[name]
        except KeyError:
            raise AttributeError(
                f"config has no key {name!r}; available: {sorted(self._data)}"
            ) from None
        return Config(value) if isinstance(value, dict) else value

    def __getitem__(self, name: str) -> Any:
        return getattr(self, name)

    def get(self, name: str, default: Any = None) -> Any:
        """Fetch a key, returning `default` when it is absent."""
        if name not in self._data:
            return default
        return getattr(self, name)

    def __contains__(self, name: str) -> bool:
        return name in self._data

    def to_dict(self) -> dict[str, Any]:
        """A deep copy of the raw data, safe to store in a checkpoint."""
        return copy.deepcopy(self._data)

    def __repr__(self) -> str:
        return f"Config({self._data!r})"


def load_config(path: str | Path) -> Config:
    """Load and validate a YAML config file."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise ConfigError(f"config file {path} must contain a YAML mapping at the top level")

    config = Config(data)
    validate_config(config)
    return config


def validate_config(config: Config) -> None:
    """Check the invariants that would otherwise fail confusingly much later.

    Catching these here turns a cryptic shape mismatch deep inside attention
    into a clear message before anything is allocated.
    """
    required_sections = ("tokenizer", "model", "training", "data")
    for section in required_sections:
        if section not in config:
            raise ConfigError(f"config is missing required section: {section!r}")

    model = config.model
    embedding_dim = model.embedding_dim
    num_heads = model.num_heads

    if embedding_dim % num_heads != 0:
        raise ConfigError(
            f"embedding_dim ({embedding_dim}) must be divisible by num_heads "
            f"({num_heads}); head dimension would be {embedding_dim / num_heads}"
        )

    vocab_size = config.tokenizer.vocab_size
    num_specials = len(config.tokenizer.special_tokens.to_dict())
    if vocab_size <= 256 + num_specials:
        raise ConfigError(
            f"vocab_size ({vocab_size}) must exceed 256 base byte tokens plus "
            f"{num_specials} special tokens; there would be no room for any BPE merges"
        )

    for name in ("context_length", "num_layers", "num_heads", "embedding_dim"):
        if model[name] <= 0:
            raise ConfigError(f"model.{name} must be positive, got {model[name]}")

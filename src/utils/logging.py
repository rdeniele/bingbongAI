"""Run logging: human-readable console output plus a machine-readable record.

Every training run writes `runs/<name>/metrics.jsonl`, one JSON object per
line. That file is the source of truth for any loss number quoted in the
documentation -- results are copied from it, never typed from memory.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any


class RunLogger:
    def __init__(self, run_dir: str | Path | None, quiet: bool = False) -> None:
        self.quiet = quiet
        self.run_dir = Path(run_dir) if run_dir is not None else None
        self._metrics_file = None
        self._text_file = None
        if self.run_dir is not None:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            # Append mode: a resumed run continues the same record.
            self._metrics_file = (self.run_dir / "metrics.jsonl").open("a", encoding="utf-8")
            self._text_file = (self.run_dir / "log.txt").open("a", encoding="utf-8")

    def info(self, message: str = "") -> None:
        if not self.quiet:
            try:
                print(message, flush=True)
            except UnicodeEncodeError:
                # The Windows console is often cp1252. Model output can contain
                # characters it cannot show (an untrained model emits broken
                # byte sequences that decode to U+FFFD). Escape them on screen;
                # the log file below still gets the exact text in UTF-8.
                encoding = sys.stdout.encoding or "ascii"
                print(message.encode(encoding, errors="backslashreplace").decode(encoding), flush=True)
        if self._text_file is not None:
            self._text_file.write(message + "\n")
            self._text_file.flush()

    def metrics(self, **values: Any) -> None:
        if self._metrics_file is None:
            return
        record = {"time": round(time.time(), 3), **values}
        self._metrics_file.write(json.dumps(record) + "\n")
        self._metrics_file.flush()

    def close(self) -> None:
        for handle in (self._metrics_file, self._text_file):
            if handle is not None:
                handle.close()
        self._metrics_file = self._text_file = None


def format_count(n: int) -> str:
    """13989888 -> '13.99M', 40861 -> '40.9K'."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)

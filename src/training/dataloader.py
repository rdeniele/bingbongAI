"""Random batches of training windows.

SAMPLING
    Each batch picks `batch_size` random start positions in the token stream.
    There are no fixed epochs: with a stream of N tokens there are about N
    distinct windows, and uniform random sampling covers them evenly over time.

REPRODUCIBILITY
    The loader owns a private torch.Generator. Its state is saved in every
    checkpoint, so a resumed run draws exactly the batches an uninterrupted run
    would have drawn. tests/test_checkpoint.py checks this by comparing the loss
    at every step of a stopped-and-resumed run against a continuous one.

    Evaluation uses `fixed_batches`, which draws from a freshly seeded generator
    every time. Every evaluation therefore scores the SAME validation windows,
    so val-loss changes between evaluations reflect the model, not the sample.
"""

from __future__ import annotations

from typing import Iterator

import torch

from .dataset import TokenDataset


class BatchLoader:
    def __init__(
        self, dataset: TokenDataset, batch_size: int, device: torch.device, seed: int
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size}")
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        self.generator = torch.Generator().manual_seed(seed)

    def _to_device(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.device.type == "cuda":
            # Pinned (page-locked) host memory lets the copy to GPU run
            # asynchronously instead of blocking the training loop.
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """x, y each (B, T) int64 on the target device."""
        starts = torch.randint(0, len(self.dataset), (self.batch_size,), generator=self.generator)
        return self._to_device(*self.dataset.windows(starts))

    def fixed_batches(self, num_batches: int, seed: int) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """The same `num_batches` batches every call, for comparable evaluations."""
        generator = torch.Generator().manual_seed(seed)
        for _ in range(num_batches):
            starts = torch.randint(0, len(self.dataset), (self.batch_size,), generator=generator)
            yield self._to_device(*self.dataset.windows(starts))

    def state_dict(self) -> dict:
        return {"generator": self.generator.get_state()}

    def load_state_dict(self, state: dict) -> None:
        self.generator.set_state(state["generator"].cpu())

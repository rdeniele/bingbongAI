"""The training loop.

ONE OPTIMIZER STEP

    for each micro-batch (gradient_accumulation_steps of them):
        x, y   = next batch                        (B, T), (B, T)
        logits = model(x)                          (B, T, V)
        loss   = cross_entropy(logits, y)          scalar
        (loss / accumulation_steps).backward()     adds d(loss)/d(weight) into .grad
    clip the global gradient norm to grad_clip
    optimizer.step()                               every weight moves against its gradient
    optimizer.zero_grad()

WHAT BACKPROPAGATION PRODUCES
    loss.backward() applies the chain rule from the loss back through the head,
    every block, and the embeddings, and leaves in each parameter's .grad the
    partial derivative d(loss)/d(parameter): how much the loss would rise if that
    single number were nudged up. Nothing is updated yet at that point.

WHAT THE OPTIMIZER DOES WITH IT
    Plain gradient descent would do  w <- w - lr * grad.  AdamW instead keeps two
    running averages per weight -- of the gradient (momentum) and of its square
    (scale) -- and steps by  lr * m / (sqrt(v) + eps). Every weight effectively
    gets its own step size, so rarely-updated weights (embeddings of uncommon
    tokens) still move a useful amount. "W" is decoupled weight decay: each step
    also shrinks weights slightly toward zero, applied only to matrices, never to
    biases or LayerNorm parameters.

LEARNING-RATE SCHEDULE
    Linear warmup from ~0 to learning_rate over warmup_steps, then cosine decay to
    min_learning_rate at max_steps. Warmup matters because AdamW's variance
    estimate is unreliable for the first steps; large early updates based on it
    can wreck a freshly initialized network.

GRADIENT CLIPPING
    If the norm of all gradients together exceeds grad_clip, they are scaled down
    as a group (direction preserved). One unusual batch cannot then throw the
    weights far off course.
"""

from __future__ import annotations

import contextlib
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ..utils.logging import RunLogger, format_count
from .checkpoint import (
    capture_rng_state,
    check_compatibility,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
)
from .dataloader import BatchLoader

MIB = 1024**2
EVAL_SEED = 12345           # fixed: every evaluation scores the same validation windows
VRAM_WARN_FRACTION = 0.95


@dataclass(frozen=True)
class TrainingSettings:
    batch_size: int
    learning_rate: float
    max_steps: int
    min_learning_rate: float = 0.0
    warmup_steps: int = 0
    gradient_accumulation_steps: int = 1
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95
    eval_interval: int = 250
    eval_batches: int = 20
    checkpoint_interval: int = 500
    keep_last: int = 2
    log_interval: int = 25
    seed: int = 1337
    precision: str = "fp32"

    @classmethod
    def from_config(cls, config) -> "TrainingSettings":
        t = config.training
        fields = cls.__dataclass_fields__
        values = {name: t[name] for name in fields if name in t}
        return cls(**values)

    def __post_init__(self) -> None:
        if self.precision not in {"fp32", "bf16", "fp16"}:
            raise ValueError(f"precision must be fp32, bf16 or fp16, got {self.precision!r}")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be at least 1")
        if self.warmup_steps < 0 or self.max_steps < 1:
            raise ValueError("warmup_steps must be >= 0 and max_steps >= 1")


def learning_rate_at(step: int, s: TrainingSettings) -> float:
    """Learning rate for optimizer step `step` (0-based)."""
    if step < s.warmup_steps:
        return s.learning_rate * (step + 1) / s.warmup_steps
    if step >= s.max_steps:
        return s.min_learning_rate
    decay_span = max(1, s.max_steps - s.warmup_steps)
    progress = (step - s.warmup_steps) / decay_span                  # 0 -> 1
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))              # 1 -> 0
    return s.min_learning_rate + cosine * (s.learning_rate - s.min_learning_rate)


def build_optimizer(model: nn.Module, s: TrainingSettings, device: torch.device) -> torch.optim.AdamW:
    """AdamW with weight decay on matrices only.

    Decay pulls weights toward zero. That is useful regularisation for weight
    matrices and embeddings (2-D). For biases and LayerNorm scale/shift (1-D) it
    only fights the parameters' job -- a LayerNorm gain decayed toward zero
    shrinks the signal -- so those are excluded.
    """
    decay, no_decay = [], []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        (decay if param.dim() >= 2 else no_decay).append(param)

    groups = [
        {"params": decay, "weight_decay": s.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        groups,
        lr=s.learning_rate,
        betas=(s.beta1, s.beta2),
        fused=device.type == "cuda",       # single-kernel update on GPU; same algorithm
    )


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        settings: TrainingSettings,
        train_loader: BatchLoader,
        val_loader: BatchLoader,
        device: torch.device,
        *,
        config_dict: dict[str, Any],
        tokenizer_fingerprint: str,
        checkpoint_dir: str | Path | None,
        logger: RunLogger | None = None,
    ) -> None:
        self.model = model
        self.settings = settings
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config_dict = config_dict
        self.tokenizer_fingerprint = tokenizer_fingerprint
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir is not None else None
        self.logger = logger or RunLogger(None)

        if settings.precision == "fp16" and device.type != "cuda":
            raise ValueError("fp16 training needs CUDA; use fp32 or bf16 on CPU")

        self.optimizer = build_optimizer(model, settings, device)
        self.autocast_dtype = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}[
            settings.precision
        ]
        # fp16 gradients can underflow to zero; the scaler multiplies the loss up
        # before backward and divides gradients back down before the update.
        self.scaler = torch.amp.GradScaler(device.type, enabled=settings.precision == "fp16")

        self.step = 0
        self.best_val_loss = math.inf
        self.loss_history: list[tuple[int, float]] = []      # (step, train loss) for every step
        self.eval_history: list[tuple[int, float]] = []      # (step, val loss)
        self._vram_warned = False
        self._last_grad_norm = 0.0

    # -- pieces -----------------------------------------------------------

    def _autocast(self):
        if self.autocast_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype)

    def train_step(self) -> float:
        """One optimizer step. Returns the mean loss over its micro-batches."""
        s = self.settings
        lr = learning_rate_at(self.step, s)
        for group in self.optimizer.param_groups:
            group["lr"] = lr

        self.model.train()
        total_loss = 0.0
        for _ in range(s.gradient_accumulation_steps):
            x, y = self.train_loader.next_batch()
            with self._autocast():
                _, loss = self.model(x, targets=y)
            # Divide so accumulated gradients equal the gradient of the MEAN loss.
            self.scaler.scale(loss / s.gradient_accumulation_steps).backward()
            total_loss += loss.detach().float().item()

        self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), s.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

        mean_loss = total_loss / s.gradient_accumulation_steps
        if not math.isfinite(mean_loss):
            raise FloatingPointError(
                f"loss became {mean_loss} at step {self.step + 1} "
                f"(grad norm {float(grad_norm):.3g}). Training has diverged; "
                f"lower the learning rate or check the data."
            )
        self.step += 1
        self.loss_history.append((self.step, mean_loss))
        self._last_grad_norm = float(grad_norm)
        return mean_loss

    @torch.no_grad()
    def evaluate(self) -> float:
        """Mean validation loss over the same fixed windows every time."""
        self.model.eval()
        losses = []
        for x, y in self.val_loader.fixed_batches(self.settings.eval_batches, seed=EVAL_SEED):
            with self._autocast():
                _, loss = self.model(x, targets=y)
            losses.append(loss.float().item())
        self.model.train()
        return sum(losses) / len(losses)

    # -- checkpoints ------------------------------------------------------

    def _payload(self) -> dict[str, Any]:
        model_cfg = getattr(self.model, "cfg", None)
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "step": self.step,
            "best_val_loss": self.best_val_loss,
            "loader": self.train_loader.state_dict(),
            "rng": capture_rng_state(),
            "config": self.config_dict,
            "model_config": model_cfg.to_dict() if model_cfg is not None else {},
            "tokenizer_fingerprint": self.tokenizer_fingerprint,
        }

    def save(self, name: str) -> Path | None:
        if self.checkpoint_dir is None:
            return None
        return save_checkpoint(self.checkpoint_dir / name, self._payload())

    def _prune_step_checkpoints(self) -> None:
        if self.checkpoint_dir is None:
            return
        step_files = sorted(self.checkpoint_dir.glob("step_*.pt"))
        excess = len(step_files) - self.settings.keep_last
        for old in step_files[: max(0, excess)]:
            old.unlink()

    def resume(self, path: str | Path) -> None:
        # Always load to CPU. RNG states must stay CPU tensors, and
        # load_state_dict copies weights and optimizer moments onto whatever
        # device the model and optimizer already live on.
        payload = load_checkpoint(path, map_location="cpu")
        model_cfg = getattr(self.model, "cfg", None)
        check_compatibility(
            payload,
            model_cfg.to_dict() if model_cfg is not None else {},
            self.tokenizer_fingerprint,
        )
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scaler.load_state_dict(payload["scaler"])
        self.step = int(payload["step"])
        self.best_val_loss = float(payload["best_val_loss"])
        self.train_loader.load_state_dict(payload["loader"])
        restore_rng_state(payload["rng"])
        self.logger.info(f"Resumed from {path} at step {self.step} "
                         f"(best val loss so far {self.best_val_loss:.4f})")

    # -- memory -----------------------------------------------------------

    def _check_vram(self, budget_bytes: int) -> float | None:
        """Peak reserved MiB; warn once if it approaches what was actually free."""
        if self.device.type != "cuda":
            return None
        peak = torch.cuda.max_memory_reserved(self.device)
        if not self._vram_warned and peak > VRAM_WARN_FRACTION * budget_bytes:
            self._vram_warned = True
            self.logger.info(
                f"\n  WARNING: peak VRAM reserved ({peak / MIB:,.0f} MiB) is at or above the "
                f"{budget_bytes / MIB:,.0f} MiB that was available when training started.\n"
                f"  On Windows the driver may silently spill into system RAM instead of\n"
                f"  raising out-of-memory, making training several times slower\n"
                f"  (HARDWARE.md section 4). Lower batch_size and raise\n"
                f"  gradient_accumulation_steps to keep the same effective batch.\n"
            )
        return peak / MIB

    # -- the loop ---------------------------------------------------------

    def fit(self, max_steps: int | None = None) -> dict[str, Any]:
        s = self.settings
        target = max_steps if max_steps is not None else s.max_steps
        tokens_per_step = s.batch_size * s.gradient_accumulation_steps * self.train_loader.dataset.context_length

        budget = 0
        if self.device.type == "cuda":
            free, _ = torch.cuda.mem_get_info(self.device)
            budget = free + torch.cuda.memory_reserved(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)

        if self.step >= target:
            self.logger.info(f"Already at step {self.step}; target is {target}. Nothing to do.")
            return self._summary()

        if self.step == 0:
            initial = self.evaluate()
            self.eval_history.append((0, initial))
            self.logger.metrics(step=0, val_loss=initial)
            self.logger.info(f"  step {0:>6}/{target} | val {initial:.4f}  (before any training)")

        interval_loss, interval_steps = 0.0, 0
        interval_start = time.perf_counter()
        # Time spent evaluating and saving is excluded from tokens/sec, so the
        # throughput figure measures training alone.
        overhead = 0.0
        run_start = interval_start
        start_step = self.step

        while self.step < target:
            loss = self.train_step()
            interval_loss += loss
            interval_steps += 1

            log_now = self.step % s.log_interval == 0 or self.step == target
            eval_now = self.step % s.eval_interval == 0 or self.step == target
            ckpt_now = self.step % s.checkpoint_interval == 0 or self.step == target

            record: dict[str, Any] = {}
            if log_now:
                if self.device.type == "cuda":
                    torch.cuda.synchronize(self.device)
                elapsed = time.perf_counter() - interval_start - overhead
                record = {
                    "step": self.step,
                    "train_loss": interval_loss / interval_steps,
                    "lr": learning_rate_at(self.step - 1, s),
                    "grad_norm": self._last_grad_norm,
                    "tokens_per_sec": interval_steps * tokens_per_step / elapsed,
                }
                vram = self._check_vram(budget)
                if vram is not None:
                    record["peak_vram_reserved_mib"] = vram
                interval_loss, interval_steps = 0.0, 0
                interval_start = time.perf_counter()
                overhead = 0.0

            if eval_now:
                started = time.perf_counter()
                val = self.evaluate()
                self.eval_history.append((self.step, val))
                record.setdefault("step", self.step)
                record["val_loss"] = val
                if val < self.best_val_loss:
                    self.best_val_loss = val
                    self.save("best.pt")
                    record["new_best"] = True
                overhead += time.perf_counter() - started

            if record:
                self.logger.metrics(**record)
                self.logger.info(self._format_line(record, target))

            if ckpt_now:
                started = time.perf_counter()
                if self.save("latest.pt") is not None and s.keep_last > 0:
                    self.save(f"step_{self.step:06d}.pt")
                    self._prune_step_checkpoints()
                overhead += time.perf_counter() - started

        wall = time.perf_counter() - run_start
        summary = self._summary()
        summary["wall_seconds_this_session"] = wall
        summary["steps_this_session"] = self.step - start_step
        return summary

    def _format_line(self, r: dict[str, Any], target: int) -> str:
        parts = [f"  step {r['step']:>6}/{target}"]
        if "train_loss" in r:
            parts.append(f"train {r['train_loss']:.4f}")
        if "val_loss" in r:
            parts.append(f"val {r['val_loss']:.4f}{' *' if r.get('new_best') else ''}")
        if "lr" in r:
            parts.append(f"lr {r['lr']:.2e}")
        if "tokens_per_sec" in r:
            parts.append(f"{r['tokens_per_sec'] / 1000:.1f}k tok/s")
        if "peak_vram_reserved_mib" in r:
            parts.append(f"vram {r['peak_vram_reserved_mib']:,.0f} MiB")
        return " | ".join(parts)

    def _summary(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "best_val_loss": self.best_val_loss,
            "final_train_loss": self.loss_history[-1][1] if self.loss_history else None,
            "final_val_loss": self.eval_history[-1][1] if self.eval_history else None,
        }

    def describe(self, dataset_tokens: dict[str, int]) -> None:
        """The run header, from measured values only."""
        s = self.settings
        n_params = sum(p.numel() for p in self.model.parameters())
        if self.device.type == "cuda":
            device = f"CUDA ({torch.cuda.get_device_name(self.device)})"
        else:
            device = "CPU"
        effective = s.batch_size * s.gradient_accumulation_steps
        T = self.train_loader.dataset.context_length
        self.logger.info("=" * 72)
        self.logger.info("  BingBongAI Training")
        self.logger.info("=" * 72)
        self.logger.info(f"  Parameters:     {format_count(n_params)} ({n_params:,})")
        self.logger.info(f"  Dataset tokens: train {dataset_tokens['train']:,} / "
                         f"validation {dataset_tokens['val']:,}")
        self.logger.info(f"  Device:         {device}, precision {s.precision}")
        self.logger.info(f"  Batch:          {s.batch_size} x {s.gradient_accumulation_steps} accum "
                         f"x {T} tokens = {effective * T:,} tokens/step")
        self.logger.info(f"  Schedule:       lr {s.learning_rate:.1e} -> {s.min_learning_rate:.1e}, "
                         f"warmup {s.warmup_steps}, max {s.max_steps} steps")
        self.logger.info(f"  Starting step:  {self.step}")
        self.logger.info("")

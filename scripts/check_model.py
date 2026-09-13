"""Build the model from a config and MEASURE it on this machine.

    python scripts/check_model.py --config configs/small.yaml

Reports:
  - parameter count per component, measured and compared with the formula
  - step-0 loss versus ln(vocab_size)
  - for each attention/normalization implementation (manual, fused):
    peak GPU memory, time per full training step, tokens per second

The training steps here run on RANDOM token ids. That is a compute benchmark,
not training: the model learns nothing meaningful from noise, but the cost of a
forward + backward + optimizer step does not depend on what the tokens are. The
numbers therefore predict real training throughput, and say whether the chosen
batch size fits in VRAM at all.
"""

from __future__ import annotations

import argparse
import gc
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from src.model.language_model import (  # noqa: E402
    BingBongLM,
    ModelConfig,
    expected_parameter_count,
)
from src.utils.config import load_config  # noqa: E402

MIB = 1024**2


def benchmark(
    cfg: ModelConfig,
    batch_size: int,
    device: torch.device,
    warmup_steps: int,
    measure_steps: int,
    learning_rate: float,
) -> dict:
    """Run real training steps on random data; return measurements or an OOM marker."""
    torch.manual_seed(0)
    model = BingBongLM(cfg).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    seq_len = cfg.context_length

    def step() -> float:
        ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=device)
        targets = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), device=device)
        _, loss = model(ids, targets=targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        return loss.item()

    try:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        # Warmup: the first steps pay one-off costs (kernel selection, allocator
        # growth, AdamW lazily creating its state). Timing them would lie.
        for _ in range(warmup_steps):
            step()

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        for _ in range(measure_steps):
            step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started

        result = {
            "ok": True,
            "seconds_per_step": elapsed / measure_steps,
            "tokens_per_second": batch_size * seq_len * measure_steps / elapsed,
        }
        if device.type == "cuda":
            result["peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / MIB
            result["peak_reserved_mib"] = torch.cuda.max_memory_reserved(device) / MIB
        return result

    except torch.OutOfMemoryError:
        return {"ok": False, "error": "out of memory"}
    finally:
        del model, optimizer
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure the BingBongAI model on this machine.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--measure-steps", type=int, default=10)
    parser.add_argument("--batch-sizes", type=int, nargs="*", default=None,
                        help="batch sizes to try (default: the config's, then half, then quarter)")
    args = parser.parse_args()

    config = load_config(args.config)
    base_cfg = ModelConfig.from_config(config)
    configured_batch = config.training.batch_size
    batch_sizes = args.batch_sizes or [configured_batch, configured_batch // 2, configured_batch // 4]
    batch_sizes = [b for b in batch_sizes if b >= 1]

    print("=" * 72)
    print("  BingBongAI - Model Check")
    print("=" * 72)
    print(f"  Config: {args.config}  ({config.name})")
    print(f"  V={base_cfg.vocab_size}  T={base_cfg.context_length}  D={base_cfg.embedding_dim}  "
          f"L={base_cfg.num_layers}  H={base_cfg.num_heads}  tied={base_cfg.tie_embeddings}")

    # -- parameters -------------------------------------------------------
    model = BingBongLM(base_cfg)
    measured = model.parameter_breakdown()
    expected = expected_parameter_count(base_cfg)
    print("\n  Parameters                 measured       formula")
    for key in measured:
        flag = "" if measured[key] == expected[key] else "   <-- MISMATCH"
        share = f"{100 * measured[key] / measured['total']:5.1f}%" if key != "total" else "      "
        print(f"    {key:<22} {measured[key]:>12,}  {expected[key]:>12,}  {share}{flag}")
    if measured != expected:
        raise SystemExit("\n  Parameter count does not match the formula. Stopping.")

    # -- step-0 loss ------------------------------------------------------
    torch.manual_seed(0)
    with torch.no_grad():
        ids = torch.randint(0, base_cfg.vocab_size, (4, base_cfg.context_length))
        targets = torch.randint(0, base_cfg.vocab_size, (4, base_cfg.context_length))
        _, loss = model.eval()(ids, targets=targets)
    uniform = math.log(base_cfg.vocab_size)
    print(f"\n  Step-0 loss on random tokens: {loss.item():.4f}   "
          f"(uniform guessing = ln({base_cfg.vocab_size}) = {uniform:.4f}, "
          f"difference {loss.item() - uniform:+.4f})")
    del model

    # -- hardware benchmark -----------------------------------------------
    if not torch.cuda.is_available():
        print("\n  CUDA not available: skipping GPU benchmark.")
        return

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    free, total = torch.cuda.mem_get_info(device)
    print(f"\n  Device: {props.name}, {total / MIB:,.0f} MiB total, "
          f"{free / MIB:,.0f} MiB free right now")
    print(f"  torch {torch.__version__}, precision fp32, "
          f"{args.warmup_steps} warmup + {args.measure_steps} measured steps each")
    print("  Workload: forward + backward + grad clip + AdamW step, random tokens\n")

    header = f"    {'kernels':<8} {'batch':>5} {'tokens/step':>11} {'peak alloc':>12} " \
             f"{'peak reserved':>14} {'s/step':>8} {'tokens/s':>10}"
    print(header)
    print("    " + "-" * (len(header) - 4))

    learning_rate = config.training.learning_rate
    for fused in (False, True):
        cfg = ModelConfig(**{**base_cfg.to_dict(), "fused_kernels": fused})
        label = "fused" if fused else "manual"
        for batch in batch_sizes:
            r = benchmark(cfg, batch, device, args.warmup_steps, args.measure_steps, learning_rate)
            tokens = batch * cfg.context_length
            if r["ok"]:
                print(f"    {label:<8} {batch:>5} {tokens:>11,} {r['peak_allocated_mib']:>9,.0f} MiB "
                      f"{r['peak_reserved_mib']:>10,.0f} MiB {r['seconds_per_step']:>8.3f} "
                      f"{r['tokens_per_second']:>10,.0f}")
            else:
                print(f"    {label:<8} {batch:>5} {tokens:>11,}   {r['error'].upper()}")

    print()
    print("  Notes: peak alloc = memory held by tensors; peak reserved = what the CUDA")
    print("  allocator took from the driver (the number that must fit in VRAM).")
    print("  A laptop GPU throttles under sustained load: long-run throughput will be lower.")


if __name__ == "__main__":
    main()

# HARDWARE.md

Environment report for the machine BingBongAI is being built on.

**Measured on:** 2026-09-13
**How:** `python scripts/inspect_hardware.py` plus Windows CIM queries (`Win32_ComputerSystem`,
`Win32_Processor`, `Win32_VideoController`, `Win32_LogicalDisk`) and `nvidia-smi`.
Raw machine-readable output: [data/processed/hardware_report.json](data/processed/hardware_report.json).

Everything in the "Measured" sections was read off this machine. Everything in the "Estimated"
sections is arithmetic or projection and is labelled as such. Section 4 holds the first real
benchmark, run in Phase 4.

---

## 1. Measured hardware

| Component | Value |
|---|---|
| Machine | ASUS TUF Gaming A15 (FA506ICB) |
| OS | Windows 11 Home Single Language, build 10.0.26200, 64-bit |
| CPU | AMD Ryzen 7 4800H with Radeon Graphics |
| CPU cores | 8 physical / 16 logical, base 2.90 GHz |
| System RAM | 25,146,515,456 bytes = **23.4 GiB** total (10.4 GiB free at inspection time) |
| Discrete GPU | **NVIDIA GeForce RTX 3050 Laptop GPU** |
| GPU VRAM | **4096 MiB (4 GiB)**, 168 MiB already in use by Discord at inspection |
| GPU compute capability | **8.6** (Ampere) |
| NVIDIA driver | 616.92, CUDA UMD 13.4 |
| Integrated GPU | AMD Radeon Graphics (Renoir iGPU), 512 MiB — **not usable for PyTorch on Windows** |
| Disk C: | 255.6 GB total, **13.8 GB free** ⚠ |
| Disk D: | 219.8 GB total, **115.2 GB free** ← project lives here (`D:\personal\projects\bingbongAI`) |

## 2. Measured software environment

| Component | Value |
|---|---|
| Python | **3.14.4** (CPython, 64-bit) at `C:\Users\ronde\AppData\Local\Python\pythoncore-3.14-64\python.exe` |
| pip | 26.0.1 |
| NumPy | 2.4.4 (installed) |
| PyTorch | **2.14.0+cu126** in `D:/personal/projects/bingbongAI/.venv` (installed Phase 2) |
| CUDA via PyTorch | **Available.** Built against CUDA 12.6; device reports 4.0 GiB, compute 8.6 |
| conda | not found |

**PyTorch wheel availability (verified against the package index, not assumed):**
`pip index versions torch` for this Python 3.14 interpreter reports `2.14.0, 2.13.0, 2.12.1,
2.12.0, 2.11.0, 2.10.0, 2.9.1, 2.9.0`, and the CUDA 12.6 index (`download.pytorch.org/whl/cu126`)
reports the matching `+cu126` builds. So **Python 3.14 + CUDA PyTorch is installable here.**
Driver CUDA UMD 13.4 is newer than the cu126 toolkit, which is the supported direction.

### Known environment risks
- **C: has only 13.8 GB free.** A CUDA PyTorch install pulls ~5–6 GB of NVIDIA wheels, and pip's
  cache defaults to `C:\Users\ronde\AppData\Local\pip\cache`. Put the virtualenv on D: **and**
  redirect the pip cache to D:, or C: will get uncomfortably tight. *(Estimated sizes, from
  typical wheel sizes — not measured on this machine.)*
- Python 3.14 is recent enough that some optional ML tooling may not have wheels yet. Core
  torch + numpy is confirmed available; anything else gets checked before it is added to
  `requirements.txt`.
- This is a **laptop GPU at a 75 W cap**. Sustained training will thermally throttle. Any
  tokens/sec number must be measured over a long run, not over the first 50 steps.

---

## 3. What is realistically possible on this machine

The binding constraint is **4 GiB of VRAM**, of which roughly **3.7 GiB is usable** in practice
(driver reserve + whatever Windows/Discord is holding). 23.4 GiB of system RAM is generous and
will not be the bottleneck for the model; it matters for dataset preprocessing.

### Memory arithmetic (Estimated — formula, not measured)

Training a model with AdamW in fp32 costs roughly **16 bytes per parameter**:
4 (weights) + 4 (gradients) + 4 + 4 (Adam's two moment buffers). Activations are on top of that
and scale with `batch_size × context_length × embedding_dim × num_layers`.

| Model | Params (computed) | Optimizer+weights @16 B/param | Verdict on 4 GiB |
|---|---|---|---|
| tiny (4 layers, d=256, V=4096, T=256) | **4.27 M** | ~68 MiB | Very comfortable |
| small (6 layers, d=384, V=8192, T=512) | **13.99 M** | ~224 MiB | Comfortable |
| medium (8 layers, d=512, V=16384, T=512) | **33.87 M** | ~542 MiB | Should fit; needs measuring |
| 100 M+ | — | ~1.6 GiB+ | Possible only with gradient accumulation / fp16 / checkpointing. Not a near-term target. |

Parameter counts are **exact arithmetic** from the architecture planned in `ARCHITECTURE.md`
(weight-tied LM head, learned positional embeddings), not guesses — but they will be
**verified against the real implementation in Phase 3** before being treated as fact.

### Answers to the seven Phase-1 questions

**1. Recommended first model size: ~4.3 M parameters.**
`vocab_size=4096, embedding_dim=256, num_layers=4, num_heads=4, context_length=256`.
Small enough that a full training run is minutes, not hours — which is what you want while the
code is still being debugged. Fast iteration beats capability at this stage.

**2. Recommended context length: 256 tokens.**
Attention cost grows with the square of context length. 256 is long enough to show that
attention is doing something real, cheap enough to train repeatedly. Raise it to 512 once the
pipeline is proven.

**3. Recommended batch size: 32 sequences × 256 tokens = 8,192 tokens per step.**
This is a starting point, not a measurement. The trainer will print real VRAM usage, and we
tune from there. If it OOMs, halve it and use gradient accumulation to keep the effective batch.

**4. Recommended training configuration (starting point, to be tuned against real loss curves):**

| Setting | Value | Why |
|---|---|---|
| optimizer | AdamW | Standard for transformers; decoupled weight decay |
| learning rate | 3e-4 | Conventional for a model this small |
| lr schedule | linear warmup (100 steps) → cosine decay | Warmup avoids an early-training blowup |
| weight decay | 0.1 | Applied to weight matrices, not biases/norms |
| gradient clipping | 1.0 | Cheap insurance against loss spikes |
| precision | fp32 first, then try AMP (bf16/fp16) | Get it *correct* before making it fast |
| eval interval | every 250 steps | Catch overfitting on a small dataset |
| checkpoint interval | every 500 steps + on best val loss | |

**5. Expected limitations — stated plainly:**
- 4 GiB VRAM is the hard ceiling. Models past ~50 M parameters need real memory tricks.
- A 75 W laptop GPU throttles under sustained load; long runs will be slower than short ones.
- We are training on a **tiny dataset**, so the first models will memorise rather than
  generalise. That is expected and is in fact the point of the Phase-7 overfitting experiment.
- A 4 M-parameter model trained on a small corpus will produce **locally plausible but largely
  incoherent text**. It will not answer questions. That is not a bug.
- No tokens/sec, VRAM, or loss number exists yet for this machine. **Not measured.**

**6. Scaling path.** Only move to the next tier once the current one trains cleanly end to end:
`4 M → 14 M → 34 M → reassess`. The jump past ~50 M needs mixed precision and probably
gradient accumulation, and should be planned then, with measurements in hand.

**7. Android (much later).** A 4–14 M parameter model quantised to INT8 lands in the tens of
megabytes, which is unremarkable for a phone. The open question is the runtime, not the size —
see `ANDROID.md`. **Nothing about Android has been tested.**

---

## 4. Measured benchmark — Phase 4

**Run:** 2026-09-13, `.venv/Scripts/python.exe scripts/check_model.py --config configs/small.yaml`

**Model:** `small` — 13,989,888 parameters (measured, matches formula exactly).
**Workload:** real training steps — forward, backward, gradient clipping, AdamW update — in fp32
on random token ids. That is a *compute* benchmark: the model learns nothing from noise, but the
cost of a step does not depend on what the tokens are.
**Conditions:** GPU at 67 °C before starting; 605 MiB of VRAM already in use by other
applications, leaving 3,306 MiB free. 3 warmup steps, then 10 timed steps per row.

| Kernels | Batch | Tokens/step | Peak allocated | Peak reserved | s/step | Tokens/s |
|---|---|---|---|---|---|---|
| manual | 16 | 8,192 | 3,268 MiB | 3,470 MiB | 0.968 | 8,467 |
| manual | 8 | 4,096 | 1,727 MiB | 1,950 MiB | 0.286 | 14,341 |
| manual | 4 | 2,048 | 958 MiB | 1,104 MiB | 0.132 | 15,531 |
| **fused** | **16** | **8,192** | **2,381 MiB** | **2,554 MiB** | **0.301** | **27,194** |
| fused | 8 | 4,096 | 1,281 MiB | 1,480 MiB | 0.158 | 25,949 |
| fused | 4 | 2,048 | 733 MiB | 874 MiB | 0.084 | 24,368 |

### What this establishes

- **The chosen `small` config fits at batch 16 × 512 with fused kernels**, using 2,554 MiB
  reserved — about 750 MiB under what was free at the time.
- **Fused kernels are 3.2× faster and use 26% less memory** at the configured batch size. The
  config default is `fused_kernels: true` because of this measurement.
- **Estimated** time for the configured 20,000 steps: 20,000 × 0.301 s ≈ **1.7 hours** —
  *before* thermal throttling, which a 13-step benchmark cannot show. Treat it as a lower bound.

### A Windows trap this run exposed

`manual` at batch 16 reserved **3,470 MiB when only 3,306 MiB was free — and did not raise an
out-of-memory error.** It ran 3.4× slower per token than `manual` at batch 8 instead.

The likely explanation (**inferred, not proven**): on Windows the NVIDIA driver's *CUDA Sysmem
Fallback Policy* lets allocations spill into shared system RAM over PCIe rather than failing.
The throughput collapse is the signature of that. The consequence matters for training:
**on this machine, running out of VRAM does not crash — it silently makes training several times
slower.** So the trainer must watch reserved memory against the card's total and warn loudly,
rather than rely on an OOM error that may never come.

## 5. Measured during a real training run — Phase 7

**Run:** 2026-09-13, `scripts/prove_learning.py --config configs/synthetic.yaml`, 600 steps,
same model and batch as section 4.

| Measurement | Value |
|---|---|
| Peak VRAM reserved | 2,564 MiB (section 4 benchmark: 2,554 MiB) |
| Throughput, per 10-step interval | 23,313 – 27,972 tokens/s |
| Throughput, steps 510–600 | ~24,500 tokens/s |
| GPU temperature / utilisation, mid-run | 86 °C / 99% (single `nvidia-smi` reading) |
| Training time, 600 steps | 217.2 s |

Late-run throughput was about 10% below the 27,194 tokens/s of the 13-step benchmark in
section 4. Together with the 86 °C reading this is **consistent with thermal throttling**, which
the short benchmark could not reveal. It is one run and one temperature reading, not a
characterisation. The VRAM-spill warning did not fire: peak usage stayed well inside the budget.

**Revised estimate** for the `small` config's 20,000 steps at ~24,500 tokens/s:
20,000 × 8,192 / 24,500 ≈ 6,690 s ≈ **1.9 hours**, plus evaluation and checkpoint time.
*Estimated* — a 20,000-step run may throttle further.

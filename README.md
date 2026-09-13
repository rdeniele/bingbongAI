# BingBongAI

A language model built from scratch — own tokenizer, own transformer, own weights, own training
loop — that runs entirely offline on a laptop, and eventually on an Android phone.

**No pretrained weights. No cloud inference APIs. Every weight in this model starts as random
noise and is trained here.**

## What "from scratch" means here

| Built by us | Allowed as a tool |
|---|---|
| Tokenizer (trained on our data) | Python |
| Transformer architecture | PyTorch (tensors, autograd, CUDA) |
| Training loop | NumPy |
| Checkpointing | |
| Text generation / sampling | |
| Retrieval and memory systems | |

Explicitly **not** used: GPT / Llama / Mistral / Qwen / Gemma / Phi / Falcon / BLOOM / DeepSeek
weights or tokenizers; OpenAI / Anthropic / Gemini / OpenRouter / Together APIs; any hosted
inference.

## Status

| Phase | What it is | State |
|---|---|---|
| 1 | Inspect hardware, scaffold project | **Done** — see [HARDWARE.md](HARDWARE.md) |
| 2 | Build the tokenizer | **Code done, 67 tests pass.** Needs a real corpus — see below |
| 3 | Build the transformer | **Done** — [src/model/](src/model/), 13,989,888 params verified |
| 4 | Verify the forward pass | **Done** — 107 tests pass; GPU benchmark in [HARDWARE.md](HARDWARE.md) §4 |
| 5 | Create a tiny dataset | **Done** — synthetic corpus, tokenized train/val split, batch loader |
| 6 | Train | **Done** — trainer with checkpoints, exact resume, VRAM-spill warning |
| 7 | Prove the model learned | **Done** — val loss 9.07 → 0.48, completions 0/33 → 33/33. See [TRAINING.md](TRAINING.md) |
| 8 | Text generation | **Done** — temperature, top-k, seeded + greedy modes, streaming, `--explain`. See [ARCHITECTURE.md](ARCHITECTURE.md) §9 |
| 9 | Offline chat | Not started |
| 10 | Local knowledge retrieval | Not started |
| 11 | Local memory | Not started |
| 12 | Quantisation | Not started |
| 13 | Android | Not started |

**Target model: `small`** — 6 layers, `d=384`, context 512, vocab 8192, **13,989,888 parameters**
(arithmetic; verified against code in Phase 3). See [configs/small.yaml](configs/small.yaml).

**BingBongAI has learned something from random initialization.** Trained for 600 steps
(217 s on the RTX 3050) on a synthetic corpus of 33 patterns, validation loss fell from
**9.0726 to 0.4812** and greedy completion of those patterns went from **0/33 to 33/33**.
Full record: [experiments/phase7_learning_proof.json](experiments/phase7_learning_proof.json).

That proves the pipeline works; it is not a useful model. It has only ever seen those 33
sentences. **It has not been trained on real text yet** — that needs a corpus.

## Setup

Verified working on this machine: **PyTorch 2.14.0+cu126, CUDA available, RTX 3050 Laptop
(4.0 GiB, compute 8.6)**. The virtualenv lives on D: alongside the project.

```bash
.venv/Scripts/python.exe scripts/inspect_hardware.py
```

```bash
.venv/Scripts/python.exe -m pytest tests/ -q
```

```bash
.venv/Scripts/python.exe scripts/train_tokenizer.py --config configs/small.yaml
```

```bash
.venv/Scripts/python.exe scripts/check_model.py --config configs/small.yaml
```

```bash
.venv/Scripts/python.exe scripts/prove_learning.py --config configs/synthetic.yaml
```

```bash
.venv/Scripts/python.exe scripts/generate.py --prompt "The color of snow is"
```

```bash
.venv/Scripts/python.exe scripts/generate.py --prompt "The capital of Japan is" --explain
```

## Layout

```
configs/      training + model configuration (YAML)
data/raw/     source text, including data/raw/knowledge/ for your own documents
data/processed/   tokenised datasets
data/tokenizer/   trained tokenizer vocabulary
checkpoints/  saved model + optimizer state
src/tokenizer/    encode/decode, tokenizer training
src/model/        embeddings, attention, feed-forward, blocks, the LM
src/training/     dataset, dataloader, trainer, checkpointing
src/inference/    generation, chat loop
src/knowledge/    dataset prep from your documents, retrieval
src/utils/        config loading, logging
scripts/      command-line entry points
tests/        automated tests
```

## Documents

- [HARDWARE.md](HARDWARE.md) — what this machine is and what it can actually train
- [ARCHITECTURE.md](ARCHITECTURE.md) — how the model works, tensor by tensor
- [TRAINING.md](TRAINING.md) — how training works and what runs have actually been done
- [KNOWLEDGE.md](KNOWLEDGE.md) — the two separate ways to give BingBongAI knowledge
- [ANDROID.md](ANDROID.md) — the phone deployment question, unanswered for now

## Ground rule

Nothing in this repository reports a result that was not actually measured. Untested things say
"Not tested yet". Projections say "Estimated".

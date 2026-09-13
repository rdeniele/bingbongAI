# TRAINING.md

How BingBongAI learns, and a log of every training run actually performed.

> **Status: implemented (Phases 5–6) and proven (Phase 7).** The trainer is in
> [src/training/](src/training/). BingBongAI has been trained from random weights on the
> synthetic dataset and demonstrably learned it — see the run log at the bottom. It has **not**
> been trained on real text yet. Every number in the run log is copied from
> `runs/<name>/metrics.jsonl` or `experiments/*.json`, never typed from memory.

## What training actually is

The model is handed a sequence and asked to predict the next token at every position at once:

```
text:    "The cat sat on the mat"
tokens:  [ The ][ cat ][ sat ][ on ][ the ][ mat ]

input:   [ The ][ cat ][ sat ][ on ][ the ]
target:  [ cat ][ sat ][ on ][ the ][ mat ]
```

Every position is a training example, so one sequence of 512 tokens gives 512 predictions
rather than one. The causal mask is what makes this safe: position 3 predicting position 4
cannot peek at position 4.

In code this is a single slice ([src/training/dataset.py](src/training/dataset.py)):
`x = tokens[s : s+T]`, `y = tokens[s+1 : s+T+1]`. The target is the input shifted by one.
There is no other labelling.

## The data pipeline

```
data/raw/<corpus>/*.txt
   │  scripts/train_tokenizer.py     learn BPE merges from the corpus
   ▼
data/tokenizer/<name>/tokenizer.json
   │  scripts/prepare_data.py        encode every document, append <|eos|>,
   │                                 split each document 90% train / 10% validation
   ▼
data/processed/<name>_train.bin      flat uint16 token ids
data/processed/<name>_val.bin
data/processed/<name>_meta.json      counts + tokenizer fingerprint
   │  scripts/train.py               random 512-token windows, batches of 16
   ▼
checkpoints/<name>/*.pt   and   runs/<name>/metrics.jsonl
```

Documents are joined into one long stream, not padded. A window may cross a document boundary;
the `<|eos|>` between them is how the model learns that what follows is unrelated.

The tokenizer fingerprint is recorded when data is prepared and checked again at training time,
so token files can never be silently paired with a different vocabulary.

## The loop

```
for each step:
    1. sample a batch of windows          (B, T) inputs, (B, T) targets
    2. forward pass                       -> logits (B, T, V)
    3. cross-entropy loss                 -> scalar
    4. loss.backward()                    -> d(loss)/d(weight) for every parameter
    5. clip the global gradient norm to 1.0
    6. optimizer.step()                   -> every weight moves against its gradient
    7. optimizer.zero_grad()
    8. periodically: evaluate on validation windows, save checkpoints
```

**What backpropagation does, concretely:** `loss.backward()` applies the chain rule from the loss
back through the head, every block and the embeddings, leaving in each parameter's `.grad` the
partial derivative of the loss with respect to that number — how much the loss would rise if it
were nudged up. Nothing is updated at that point.

**What the optimizer does with it:** plain gradient descent would do `w ← w − lr·grad`. AdamW
keeps two running averages per weight — of the gradient and of its square — and steps by
`lr · m / (√v + ε)`, so every weight effectively gets its own step size. Weight decay (the "W")
shrinks weight matrices slightly toward zero each step; biases and LayerNorm parameters are
excluded, since decaying a LayerNorm gain just shrinks the signal.

**Learning-rate schedule:** linear warmup to the peak over `warmup_steps`, then cosine decay to
`min_learning_rate`. Warmup exists because AdamW's variance estimate is unreliable for the first
few steps, and large early updates can wreck a freshly initialised network.

**Gradient accumulation:** `gradient_accumulation_steps: N` runs N micro-batches before each
update, dividing each loss by N. A test proves 2 × 4 produces the same update as 1 × 8. This is
the escape hatch if a batch does not fit in VRAM.

## Configuration

All model and training parameters live in `configs/*.yaml`, loaded through
[src/utils/config.py](src/utils/config.py). **No hyperparameter is hardcoded in `src/`.**

| Config | Architecture | Data | Purpose |
|---|---|---|---|
| [synthetic.yaml](configs/synthetic.yaml) | small, 13.99 M | `data/raw/synthetic` | Phase-7 learning proof |
| [small.yaml](configs/small.yaml) | small, 13.99 M | `data/raw` | **The real model** — waiting on a corpus |
| [tiny.yaml](configs/tiny.yaml) | tiny, 4.27 M | `data/raw` | Fast debugging |

## Commands

```bash
.venv/Scripts/python.exe scripts/train_tokenizer.py --config configs/synthetic.yaml
```

```bash
.venv/Scripts/python.exe scripts/prepare_data.py --config configs/synthetic.yaml
```

```bash
.venv/Scripts/python.exe scripts/train.py --config configs/synthetic.yaml
```

```bash
.venv/Scripts/python.exe scripts/train.py --config configs/synthetic.yaml --resume checkpoints/synthetic/latest.pt --max-steps 1000
```

```bash
.venv/Scripts/python.exe scripts/prove_learning.py --config configs/synthetic.yaml
```

On resume, the **architecture and tokenizer must match** the checkpoint (checked, with a clear
error). **Training settings come from the config**, which is how a run is extended — raise
`max_steps` and resume. Changing `max_steps` also reshapes the cosine schedule from that point.
`Ctrl+C` saves `latest.pt` before exiting.

## Checkpoints

Written to `checkpoints/<name>/`:

| File | What it is |
|---|---|
| `latest.pt` | Most recent state. Resume from this. |
| `best.pt` | Lowest validation loss so far. Use this for generation. |
| `step_NNNNNN.pt` | The last `keep_last` periodic checkpoints. Older ones are deleted. |

Each is **168 MB** for the small model (measured file size): weights, plus AdamW's two moment
buffers, which are each the same size as the weights. A checkpoint holds:

| Contents | Why |
|---|---|
| model weights | the model itself |
| optimizer state | AdamW's moments — resuming without them causes a visible loss spike |
| step, best validation loss | the schedule and `best.pt` continue correctly |
| batch sampler RNG state | a resumed run draws the same batches an uninterrupted one would |
| torch + Python RNG state | reproducibility |
| full config + model config | a checkpoint describes itself; architecture is checked on load |
| tokenizer fingerprint | weights can never be loaded against the wrong vocabulary |

Saves are **atomic** (write to a temp file, then rename), so a crash mid-save cannot corrupt the
previous checkpoint. Loading uses `torch.load(weights_only=True)`, which refuses to unpickle
anything but tensors and plain containers — a `.pt` file is a pickle, and unrestricted
unpickling can execute arbitrary code.

**Verified:** [tests/test_checkpoint.py](tests/test_checkpoint.py) stops a run at step 15,
restores it into a fresh trainer built with a *different* seed, finishes to step 30, and asserts
the loss at every later step equals a run that never stopped. That only passes if weights,
optimizer moments, step, schedule position and sampler RNG are all restored. A separate test
does a resume on CUDA.

## The Phase-7 experiment: proving the model learns

A synthetic corpus of 33 rigid patterns, each ending in a word fully determined by what precedes
it ([src/training/synthetic.py](src/training/synthetic.py)):

```
The color of snow is white.        After three comes four.
A lion roars.                      The capital of Japan is Tokyo.
```

200 independently shuffled passes over the 33 sentences, 6,600 lines, 45,401 tokens.

The claim is narrow and falsifiable: **starting from random weights, training makes this exact
model learn.** It does not claim intelligence.

| Check | Pass condition |
|---|---|
| Random start | Step-0 loss close to `ln(8192)` = 9.01 |
| Loss falls | Down to roughly the order-entropy floor (below) |
| Patterns learned | Greedy completion correct on all 33 patterns, after training and not before |
| Generation changes | Free text from the same prompt goes from noise to the corpus's patterns |

### A correction to the original plan: the loss cannot reach zero

An earlier version of this document said the model "must reach near-zero loss". **That was
wrong for this dataset**, and it matters for reading the curve.

Sentence *order* is shuffled, so nothing — not even a perfect model — can know which sentence
comes next. Each sentence carries about `ln(33)` = 3.50 nats of unavoidable surprise, spread over
its tokens. Measured with the real tokenizer, that works out to an **estimated floor of ~0.508
nats per token**. A model that has learned everything learnable should level off *near* 0.5,
not at 0.

The estimate is deliberately pessimistic: shuffling is *without replacement* within each pass,
so a model that notices which sentences it has recently seen can beat it slightly. A final loss
a little under 0.508 is therefore expected, not suspicious.

---

## Training run log

### Run 1 — Phase 7 learning proof · 2026-09-13 · ✅ passed

**Command:** `scripts/prove_learning.py --config configs/synthetic.yaml`
**Record:** [experiments/phase7_learning_proof.json](experiments/phase7_learning_proof.json)

| Setting | Value |
|---|---|
| Model | `small` architecture, **13,989,888** parameters, random init, seed 1337 |
| Tokenizer | byte-level BPE trained on the corpus: 484 tokens (model has 8,192 output rows) |
| Data | 40,861 train / 4,540 validation tokens |
| Hardware | RTX 3050 Laptop, CUDA, fp32, fused kernels |
| Batch | 16 × 512 = 8,192 tokens per step |
| Schedule | AdamW, lr 3e-4 → 3e-5, warmup 50, 600 steps (~4.9 M tokens, ~120 passes over the train split) |

**Result — before vs after:**

| Measurement | Before (random) | After 600 steps |
|---|---|---|
| Validation loss | **9.0726** | **0.4812** |
| Completions correct, prompt in context | **0 / 33** | **33 / 33** |
| Completions correct, bare prompt | **0 / 33** | **33 / 33** |
| Mean probability on the correct answer (in context) | 0.000122 | 0.990035 |
| Mean probability on the correct answer (bare) | 0.000119 | 0.9698 |

0.000122 is exactly what random guessing over 8,192 outputs gives (1/8192 = 0.000122). After
training the model puts 99% of its probability on the right word.

**Validation loss curve** (fixed validation windows, so points are directly comparable):

| Step | 0 | 50 | 100 | 150 | 200 | 250 | 300 | 350 | 400 | 450 | 500 | 550 | 600 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Val loss | 9.0726 | 1.4605 | 0.8798 | 0.8536 | 0.7703 | 0.6032 | 0.5206 | 0.5040 | 0.4938 | 0.4865 | 0.4821 | 0.4819 | 0.4812 |

Two distinct phases are visible. By step 100 the loss has fallen to ~0.88: the model has learned
the vocabulary and sentence templates but not yet which word goes with which subject. A plateau
follows (steps 100–200), then a second drop to ~0.5 as the specific associations lock in. The
curve flattens at 0.48 — just under the estimated 0.508 floor, as predicted above.

**Generation, greedy, same prompt:**

| Prompt | Before | After |
|---|---|---|
| `The color of snow is` | `>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>` | ` white.⏎The color of the sky is blue.⏎The color of a cloud is grey.⏎…` |
| `A lion` | ` moo moo moo moo moo moo moo …` | ` roars.⏎The color of the sun is yellow.⏎…` |
| `After three comes` | ` black black black black …` (then broken bytes) | ` four.⏎The capital of Japan is Tokyo.⏎…` |

After finishing a sentence, greedy decoding always starts the next one with "The color of…" or
"The capital of…". That is the correct greedy behaviour, not a failure: 16 of the 33 sentences
start with "The", so it is the single most likely continuation, and greedy decoding always takes
the single most likely token. Phase 8's sampling will vary this.

**Cost:** 217.2 s of training for 600 steps. Peak VRAM reserved 2,564 MiB. Throughput per
10-step interval ranged 23,313–27,972 tokens/s; steps 510–600 averaged ~24,500, below the
27,194 measured in the short Phase-4 benchmark. The GPU was observed at 86 °C and 99%
utilisation mid-run, consistent with thermal throttling.

**What went wrong on the way — three attempts before this record:**

1. **Crashed before training.** The untrained model chose token id 2777 during the "before"
   measurement. The model has 8,192 output rows but the synthetic tokenizer only 484, so that id
   has no text and `decode` raised. *Fix:* generation now only considers ids the tokenizer can
   decode; regression test added.
2. **Trained fully (val 0.4797), then crashed while printing.** The random model's output
   decodes to U+FFFD replacement characters, which the Windows cp1252 console cannot print. The
   results file had not been written yet and was lost. *Fix:* console output escapes unprintable
   characters, and the record is now written **before** the report is printed.
3. **The recorded run above.** Same seed as attempt 2, final val 0.4812 versus 0.4797. Same
   start (9.0726 both times), slightly different end. The likely cause is non-deterministic GPU
   kernels (fused attention and the fused AdamW update are not guaranteed bit-exact run to run);
   exact reproducibility is only verified on CPU. The difference does not change any conclusion.

**Resume verified on GPU:** `train.py --resume checkpoints/synthetic/latest.pt --max-steps 620`
continued from step 600 with train loss 0.4662 at step 610 — no spike — and val 0.4798 at 620.
The first attempt at this crashed (`RNG state must be a torch.ByteTensor`): checkpoints were being
loaded directly onto the GPU, RNG state included, which the CPU test suite could not catch.
*Fix:* checkpoints always load to CPU; a CUDA-only regression test was added.

Note that `checkpoints/synthetic/` now holds the 620-step model, while the experiment record
describes the model at step 600.

### Conclusion

Random weights → training → loss falls from 9.07 to 0.48 → the model completes all 33 patterns
it completed none of before → its free generation changes from noise to the corpus's structure.
**BingBongAI's model, tokenizer, data pipeline, optimizer and training loop work end to end.**

What this does *not* show: any ability on text it has not seen. Every validation sentence also
appears in training, so on this corpus validation loss measures memorisation. Generalisation
can only be measured on a real corpus.

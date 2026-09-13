# ARCHITECTURE.md

How BingBongAI's language model works.

> **Status: implemented (Phase 3) and verified (Phase 4).** Code in [src/model/](src/model/).
> Shapes, causal masking, parameter counts and initial loss described below are asserted in
> [tests/test_model.py](tests/test_model.py) and [tests/test_attention.py](tests/test_attention.py).
> Generation (section 9) is implemented and measured on the Phase-7 synthetic checkpoint.

## Notation

**Primary target: the `small` config** ([configs/small.yaml](configs/small.yaml)).
`tiny` is kept as a fast debugging config.

| Symbol | Meaning | **small** | tiny |
|---|---|---|---|
| `B` | batch size | 16 | 32 |
| `T` | context length (tokens per sequence) | **512** | 256 |
| `V` | vocabulary size | **8192** | 4096 |
| `D` | embedding dimension (`embedding_dim`) | **384** | 256 |
| `H` | number of attention heads | **6** | 4 |
| `Dh` | dimension per head = `D / H` | **64** | 64 |
| `L` | number of transformer blocks | **6** | 4 |

---

## 0. Tokenization: how text becomes numbers

The model never sees text. Before anything else, a string is turned into a list of integers by
the tokenizer in [src/tokenizer/tokenizer.py](src/tokenizer/tokenizer.py) — **implemented from
scratch, trained on our own corpus, no pretrained vocabulary.**

The design is **byte-level BPE**. Start with the 256 possible byte values as the base
vocabulary, then repeatedly find the most frequent adjacent pair of tokens in the corpus and
merge it into a new token. Common words collapse to a single token; rare ones stay as pieces.

```
"The color of snow is white."
    ↓  pre-tokenize (split on word boundaries, keep leading spaces)
["The", " color", " of", " snow", " is", " white", "."]
    ↓  UTF-8 bytes, then replay the learned merges
[261, 268, 262, 404, 263, 407, 46]
```

That is **real measured output** from the trained tokenizer, not an illustration. Seven tokens
for seven words.

The property that matters is that this is **lossless and closed**: every byte value 0–255 is
already in the vocabulary, so *any* input — emoji, Filipino, Python source, a corrupt byte —
encodes and decodes exactly. `decode(encode(x)) == x` is asserted over 21 adversarial samples in
[tests/test_tokenizer.py](tests/test_tokenizer.py).

A consequence worth stating plainly: `<|unk|>` exists as a reserved id, but `encode()` can never
emit it. There is no out-of-vocabulary input. That is the guarantee byte-level buys.

## The whole pipeline

```
token ids                      (B, T)   integers in [0, V)
   │
   ├─ token embedding          (B, T, D)
   ├─ + positional embedding   (B, T, D)
   ▼
transformer block × L          (B, T, D)   shape never changes through the stack
   │
   ├─ final layer norm         (B, T, D)
   ▼
language model head            (B, T, V)   one score per vocabulary entry, per position
   ▼
logits → softmax → next-token probabilities
```

The key fact: **the shape `(B, T, D)` is invariant through the entire stack.** Every block takes
`(B, T, D)` and returns `(B, T, D)`. That is why blocks can be stacked arbitrarily.

---

## 1. Token embeddings — `src/model/embeddings.py`

**Input:** `(B, T)` integer token ids. **Output:** `(B, T, D)` floats.

A learned lookup table of shape `(V, D)`. Row `i` is the vector for token `i`. Embedding is
literally an indexing operation — no matrix multiply — but the table is a trained parameter, so
the model learns what each token "means" as a direction in `D`-dimensional space.

Initialised from `N(0, 0.02)`. **This is where "random initialisation" begins.**

## 2. Positional information

Self-attention is *permutation-equivariant*: with no positional signal, "dog bites man" and
"man bites dog" produce identical attention patterns. Position has to be injected.

**Choice: learned positional embeddings** — a second table of shape `(T_max, D)`, added to the
token embeddings.

Why learned rather than sinusoidal or RoPE: it is the simplest thing that works, it is trivially
inspectable, and at `T=512` it costs 196,608 parameters — 1.4% of the small model. Its real
limitation is that it **cannot extrapolate past `T_max`**. RoPE is the upgrade path once
context length starts mattering; that trade is revisited, not assumed away.

## 3. Causal self-attention — `src/model/attention.py`

The core mechanism. For each position, the model asks: *which earlier positions should I read
from, and what do I take from them?*

**Input** `(B, T, D)` → **output** `(B, T, D)`.

```
x                          (B, T, D)
  │ one linear layer D → 3D, then split
  ▼
q, k, v                    each (B, T, D)
  │ reshape to heads: (B, T, H, Dh) → transpose → (B, H, T, Dh)
  ▼
scores = q @ kᵀ / √Dh      (B, H, T, T)   ← how much each position attends to each other
  │ causal mask: set scores[i, j] = -inf where j > i
  ▼
weights = softmax(scores)  (B, H, T, T)   each row sums to 1
  ▼
out = weights @ v          (B, H, T, Dh)
  │ transpose back, merge heads
  ▼
(B, T, D) → output projection D → D → (B, T, D)
```

**The `1/√Dh` scaling** keeps the dot products from growing with `Dh`. Without it, large scores
push softmax into a near-one-hot regime where gradients vanish.

**The causal mask** is what makes this a *language* model. Position `i` may only attend to
positions `≤ i`. Set to `-inf` before softmax, those entries become exactly 0 after it. If this
mask is wrong, the model sees the future, the training loss looks fantastic, and generation is
garbage — so [tests/test_attention.py](tests/test_attention.py) asserts that changing tokens at
positions `≥ j` leaves every output at positions `< j` unchanged, checked at every split point on
both kernels. To prove that test is not vacuous, it was run against a deliberately sabotaged
all-ones mask: it detected the leak.

**Multiple heads** let the model run `H` independent attention patterns at once (one may track
syntax, another long-range reference) at no extra cost, since `H × Dh = D`.

## 4. Feed-forward network — `src/model/feed_forward.py`

**Input** `(B, T, D)` → **output** `(B, T, D)`.

```
(B, T, D) → Linear D → 4D → (B, T, 4D) → GELU → Linear 4D → D → (B, T, D)
```

Applied to each position **independently** — no mixing across the sequence. Attention moves
information *between* positions; the feed-forward network *processes* what arrived. The 4×
expansion is convention. This holds two thirds of the parameters in each block.

## 5. Transformer block — `src/model/transformer_block.py`

Pre-normalisation layout:

```
x = x + attention(layer_norm(x))
x = x + feed_forward(layer_norm(x))
```

**Residual connections** (`x + ...`) mean each sublayer learns a *correction*, not a replacement.
They also give gradients a direct path to early layers, which is what makes deep stacks
trainable at all.

**Layer normalisation** normalises each position's `D`-vector to zero mean and unit variance,
then applies learned scale and shift. **Pre-LN** (normalise *before* the sublayer) rather than
post-LN, because it trains stably without a carefully tuned warmup.

## 6. Language model head — `src/model/language_model.py`

**Input** `(B, T, D)` → **output** `(B, T, V)` logits.

A linear projection `D → V` producing, at every position, one raw score per vocabulary entry.
Position `i`'s logits are the model's prediction for token `i+1`.

**Weight tying:** the head reuses the token embedding matrix transposed. It saves `V × D`
parameters (3.15 M of the small model's 13.99 M — 22.5%) and usually helps quality. The
intuition: the vector that *represents* a token is a sensible vector to *score* it with.

## 7. Loss

Cross-entropy between the logits at position `i` and the actual token at position `i+1`:

```
logits  (B, T, V)  →  reshape (B*T, V)
targets (B, T)     →  reshape (B*T,)
loss = cross_entropy(logits, targets)     scalar
```

A useful sanity check: an untrained model should produce loss ≈ `ln(V)`, because random weights
give a near-uniform distribution over the vocabulary. **If step-0 loss is not close to that,
initialization is broken.**

**Measured** on the `small` model (`scripts/check_model.py`, random tokens): step-0 loss
**9.0833** against `ln(8192)` = **9.0109** — a difference of +0.07. The small excess is expected:
logits at init are not exactly zero, just small, and any spread in them costs a little loss.

## 7b. Two implementations of attention and LayerNorm

Each of attention and LayerNorm has a **manual** path (the formula written out, line for line —
this is the specification) and a **fused** path (PyTorch's `scaled_dot_product_attention` and
`layer_norm` kernels). Tests assert the two produce the same outputs *and the same gradients*.
The choice is `model.fused_kernels` in the config.

This is not a style choice. **Measured** on the RTX 3050 Laptop, `small` config, batch 16 × 512:

| Kernels | Peak VRAM reserved | Time / step | Tokens / s |
|---|---|---|---|
| manual | 3,470 MiB | 0.968 s | 8,467 |
| **fused** | **2,554 MiB** | **0.301 s** | **27,194** |

The manual attention path materialises the `(B, H, T, T)` score matrix — 25.2 M entries per
layer at this size — and keeps it for backpropagation. The fused kernel computes the same
function without storing it. On a 4 GiB card that is the difference between fitting and not.
Configs default to `fused_kernels: true`; set it to `false` to train on the written-out math.

## 8. Parameter count

By formula: embeddings + `L × (12D² + 13D)` + final norm, with the head tied.

**small** (`V=8192, D=384, L=6, T=512`) — the primary target:

| Component | Parameters | Share |
|---|---|---|
| Token embedding `V × D` | 3,145,728 | 22.5% |
| Positional embedding `T × D` | 196,608 | 1.4% |
| 6 transformer blocks | 10,646,784 | 76.1% |
| Final layer norm | 768 | — |
| LM head (tied) | 0 | — |
| **Total** | **13,989,888 ≈ 13.99 M** | |

**tiny** (`V=4096, D=256, L=4, T=256`) — fast debugging config:

| Component | Parameters |
|---|---|
| Token embedding | 1,048,576 |
| Positional embedding | 65,536 |
| 4 transformer blocks | 3,159,040 |
| Final layer norm | 512 |
| **Total** | **4,273,664 ≈ 4.27 M** |

**Verified.** The implementation's `sum(p.numel())` equals these hand-derived totals exactly,
for both configs, per component. [tests/test_model.py](tests/test_model.py) asserts it, so if
the architecture or this document ever drift apart, the test suite fails.

---

## 9. Generation: how BingBongAI writes, one token at a time — `src/inference/generate.py`

Training teaches the model to predict the next token. Writing is that single prediction,
repeated, with the model reading its own output:

```
ids = encode("The capital of Japan is")
repeat:
    logits = model(ids[-512:])          (1, T, V)   run the whole model
    scores = logits[0, -1]              (V,)        keep ONLY the last position
    next   = choose(scores)             one id      greedy or sampled
    stop if next is <|eos|>
    ids.append(next)                    the choice becomes part of the next input
```

Three facts follow directly from that loop:

- **One forward pass produces one token.** 60 tokens of output means 60 full runs of the model.
- **Nothing plans ahead, and nothing is revised.** Each token is chosen knowing only the tokens
  before it. Once appended, it is part of the input forever.
- **The model's only memory is the token sequence.** There is no hidden state carried between
  steps. A test proves it: generating 10 tokens and then 5 more from the result gives exactly the
  same 15 tokens as generating 15 at once. Past 512 tokens, the oldest are dropped and the model
  cannot see them at all.

### Choosing the token

`sample_next_token` turns the `(V,)` score vector into one id:

1. **Restrict to real tokens.** The output layer has 8,192 rows; the synthetic tokenizer only 484.
   Ids with no text are never eligible. (Phase 7 found this the hard way.)
2. **Temperature `T`** divides the scores before softmax: `p_i = exp(z_i/T) / Σ exp(z_j/T)`.
   `T = 1` is the model's own distribution; `T < 1` sharpens it; `T > 1` flattens it. `T = 0`
   is treated as exact greedy argmax.
3. **Top-k** keeps the `k` highest scores and sets the rest to −∞, so their probability is
   exactly 0. It cuts off the long tail of individually unlikely tokens.
4. **Sample** one id from what remains, using a private seeded `torch.Generator`.

**Determinism.** Greedy has no randomness. Sampling is reproducible from a seed; if none is
given, one is drawn from the OS and *reported*, so any output can be regenerated. Generation
uses its own generator and never touches PyTorch's global random state (tested). Sampling
happens on CPU; logits computed on GPU may differ from CPU logits in the last bits, so a seed
reproduces on the same device, not necessarily across devices.

**Streaming.** A character can span several tokens — 🤖 is four UTF-8 bytes, which a byte-level
tokenizer may emit one at a time. `IncrementalDecoder` holds text back while it ends in an
incomplete character, so the screen never shows garbage that later "turns into" the right symbol.

### Watching it happen: `--explain`

```bash
.venv/Scripts/python.exe scripts/generate.py --prompt "The capital of Japan is" --temperature 0.8 --top-k 40 --seed 7 --max-new-tokens 12 --explain
```

**Real output** from the synthetic checkpoint (step 620), first five steps:

```
step  chosen      p(model)  p(sampled)   top candidates
   1   Tokyo         0.977       0.996    Tokyo:0.98   is:0.00   an:0.00   Peru:0.00
   2  .              0.999       1.000   .:1.00   Paris:0.00   orange:0.00
   3  \n             1.000       1.000   \n:1.00   orange:0.00   six:0.00
   4  The            0.482       0.523   The:0.48  After:0.26  A:0.25
   5   color         0.730       0.777    color:0.73   capital:0.27
```

Step 1 is a memorised fact, so the model is nearly certain. **Step 4 is where a new sentence
starts, and the model is genuinely uncertain — for a good reason.** In the corpus, 16 of 33
sentences start with "The" (0.485), 9 with "After" (0.273) and 8 with "A" (0.242). The model's
0.48 / 0.26 / 0.25 matches those frequencies. It did not just memorise sentences; it learned how
often each kind occurs. `p(sampled)` is higher than `p(model)` for the top choices because
temperature 0.8 sharpens the distribution.

### Measured: what the decoding settings actually do

`scripts/sampling_sweep.py` on the synthetic checkpoint. **Accuracy:** the 33 patterns × 3
seeds, prompt in context. **Valid lines:** complete lines from 20 free generations of 48 tokens
that are real corpus sentences. **Record:**
[experiments/phase8_sampling.json](experiments/phase8_sampling.json).

| Setting | Accuracy | Valid lines | Distinct outputs | Sentence starts The / After / A |
|---|---|---|---|---|
| greedy | 33/33 | 6/6 | 1/1 | 1.00 / 0.00 / 0.00 |
| T = 0.5 | 99/99 | 132/133 | 20/20 | 0.65 / 0.22 / 0.14 |
| T = 0.8, top-k 40 | 99/99 | 139/139 | 20/20 | 0.58 / 0.24 / 0.19 |
| **T = 1.0** | **99/99** | **142/144** | **20/20** | **0.51 / 0.28 / 0.21** |
| T = 1.5 | 89/99 | 99/140 | 20/20 | 0.46 / 0.29 / 0.23 |
| T = 1.5, top-k 5 | 94/99 | 135/146 | 20/20 | 0.47 / 0.27 / 0.25 |
| T = 2.5 | 5/99 | 2/82 | 20/20 | 0.38 / 0.17 / 0.06 |
| *corpus* | | | | *0.485 / 0.273 / 0.242* |

What the numbers show:

- **Greedy collapses to the single most likely path.** Every sentence it wrote started with "The"
  (the corpus has 48.5%). Correct, but it can only ever write one thing.
- **T = 1.0 reproduces the data.** Sentence starts of 0.51 / 0.28 / 0.21 against a corpus of
  0.485 / 0.273 / 0.242, with 142 of 144 lines valid. Sampling from the model's own distribution
  gives back the distribution it was trained on.
- **T < 1 exaggerates the majority.** At 0.5, "The" rises to 65%: sharpening makes the already
  likely likelier.
- **T > 1 breaks the patterns.** At 1.5, validity falls to 71% (99/140) and fragments and
  hybrids appear: *"A bird sings. dog barks."*, *"… of an orange Tokyo."* At 2.5 almost nothing
  is valid (2/82).
- **Top-k repairs high temperature.** At T = 1.5, adding top-k 5 lifts validity from 71% to 92%
  (135/146) and accuracy from 89 to 94 of 99, by removing the tail tokens that produced most of
  the hybrids — while keeping the more even sentence-start mix. The errors it still makes are
  among near-miss candidates: *"After four comes nine."*, *"The capital of Peru is Paris is
  yellow."*

Even at T = 1.0 there were invalid lines: *"The coal is black."* and *"The color of Japan is
green."* Sampling occasionally picks a lower-probability token and then continues plausibly from
it. That is the trade for variety.

**Defaults** in `scripts/generate.py`: temperature 0.8, top-k 40 — 99/99 accuracy and 139/139 valid
lines in this sweep, with more variety than greedy. These defaults are tuned on 33 memorised
sentences; they should be re-measured once the model is trained on real text.

### Measured: generation speed

Greedy, 60 new tokens, prompt "The color of", 3 runs after a warm-up, small model (13.99 M):

| Device | Runs (tokens/s) | Mean |
|---|---|---|
| RTX 3050 Laptop (CUDA) | 172.6, 221.2, 226.9 | **206.9** |
| Ryzen 7 4800H (CPU) | 48.6, 52.7, 53.1 | **51.5** |

The first call on CUDA is much slower: a cold `generate.py` run measured **49 tok/s** for 20
tokens, because it includes one-off GPU kernel setup. The table excludes that by warming up
first.

Every step re-runs all 6 layers over the whole window, so each new token costs more than the one
before. A **key/value cache** — keeping each layer's `k` and `v` from earlier steps and computing
only the new token — is the standard fix and the obvious first optimisation for Phase 12. It is
not implemented yet; these numbers are the baseline it will be measured against.

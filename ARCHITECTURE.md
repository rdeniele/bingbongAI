# ARCHITECTURE.md

How BingBongAI's language model works.

> **Status: implemented (Phase 3) and verified (Phase 4).** Code in [src/model/](src/model/).
> Shapes, causal masking, parameter counts and initial loss described below are asserted in
> [tests/test_model.py](tests/test_model.py) and [tests/test_attention.py](tests/test_attention.py).
> The model has **not been trained** — every weight is still random.

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

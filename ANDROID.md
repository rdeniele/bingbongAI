# ANDROID.md

Running BingBongAI offline on a phone.

> **Status: not started, deliberately.** Android work begins only after desktop inference works
> (Phase 13). Nothing in this document has been tested. Every number below is arithmetic or an
> open question.

## The target

```
Android phone, airplane mode ON -> open BingBongAI -> chat -> answers
```

No backend. No network permission needed. No pretrained model substituted in because it was
easier to deploy -- if BingBongAI cannot run on the phone, the answer is to fix BingBongAI, not
to replace it with someone else's model.

## Why this is plausible

The model is *small*. Rough arithmetic (**Estimated**, derived from parameter counts):

| Model | Params | fp32 | fp16 | int8 |
|---|---|---|---|---|
| tiny | 4.27 M | ~17 MB | ~8.5 MB | ~4.3 MB |
| small | 13.99 M | ~56 MB | ~28 MB | ~14 MB |
| medium | 33.87 M | ~135 MB | ~68 MB | ~34 MB |

Those are weight sizes only -- runtime also needs the KV cache and activations. Even so, a model
this size is unremarkable for a modern phone. **Size is not the hard part.**

## What the hard part actually is

The runtime: how a custom PyTorch model gets executed on Android. The candidates each carry a
real cost that has to be measured, not assumed:

| Option | The appeal | The catch |
|---|---|---|
| **ExecuTorch** | PyTorch's own on-device runtime; direct export path from our code | Newer tooling; the export step has to actually accept our architecture |
| **ONNX Runtime Mobile** | Mature, well-documented Android story | Requires a clean ONNX export; custom ops can be awkward |
| **TorchScript / PyTorch Mobile** | Simplest export path | Being superseded by ExecuTorch |
| **Write the inference in C++/Kotlin ourselves** | Total control, no framework dependency, most in the spirit of the project | By far the most work -- we would be reimplementing attention and matmul |

None of these has been tried. Picking one before the model architecture is settled would be
guessing.

## Sequencing

1. Desktop inference works and is measured.
2. Quantisation is done and **benchmarked** (Phase 12). Quantisation is not assumed to help.
3. *Then* export is attempted, and whichever runtime accepts our model is the one we use.
4. A minimal Android app: text in, text out, no network permission in the manifest.

## What will not happen

- Substituting a downloaded pretrained model because mobile deployment is easier with it.
- Any cloud fallback, "just for the hard questions".
- Claiming a phone performance number that was not measured on an actual phone.

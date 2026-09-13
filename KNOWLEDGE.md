# KNOWLEDGE.md

The two -- separate, non-interchangeable -- ways to give BingBongAI knowledge.

> **Status: plan, not implementation.** Phases 10 and 11. Nothing here is built.

## The distinction that matters most

**Dropping a file into `data/raw/knowledge/` does NOT teach the neural network anything.**
It puts a file in a folder. Nothing more. This document exists largely to keep that honest,
because conflating the two mechanisms is the easiest way to fool yourself about what your model
actually knows.

There are two genuinely different mechanisms:

| | **A. Training knowledge** | **B. Retrieval knowledge** |
|---|---|---|
| What changes | the model's **weights** | the **prompt** the model is given |
| When | during a training run | at conversation time |
| Cost | a full training run | a local search, milliseconds |
| Updating a fact | retrain | edit the file |
| Where it lives | spread across millions of weights | in a file, readable |
| Can it be wrong | yes, and you cannot inspect why | yes, but you can read the source |
| Does the model "know" it | it absorbed the statistical pattern | no -- it is reading it off the page |

Both are legitimate. The rule is that the system never presents B as if it were A.

## A. Training knowledge -- `src/knowledge/prepare_dataset.py`

```
your documents -> cleaning -> tokenizer -> token ids -> training run -> changed weights
```

The honest caveat: teaching a fact by training is *unreliable at small scale*. A 4M-parameter
model trained on a handful of documents learns the **style and vocabulary** of your writing long
before it learns any particular **fact** from it. Expect early models to produce text that
sounds like your notes while confidently stating things your notes never said.

For actual factual recall on this hardware, **retrieval is the mechanism that works.**

## B. Retrieval knowledge -- `src/knowledge/retrieval.py`

```
question -> local search over your documents -> best-matching passages
         -> placed into the model's context -> model answers with them in view
```

Fully offline. **No cloud embedding API** -- that would break the core premise of the project.

**Starting approach: BM25**, a classic keyword-relevance ranking. It scores a passage by how
many of the question's rare words it contains, discounting words that appear everywhere. It is a
few dozen lines of plain Python, needs no model, no training and no GPU, and is genuinely strong
on small document collections.

Its weakness is real and worth naming: it matches **words, not meaning**. Ask about "car" and a
document that only says "automobile" scores zero. The upgrade is locally-computed embeddings --
which, to stay from-scratch, means *our own* trained encoder, not a downloaded one. That is a
later decision, made with BM25's measured failures in hand rather than guessed at in advance.

## Local memory -- Phase 11

Conversation memory, stored locally as plain readable files. Design constraints, fixed now:

- **Local only.** Never transmitted anywhere.
- **Inspectable.** Plain text or JSON you can open and read.
- **Deletable.** A command to wipe it, and deleting the file is always a valid way to do that.
- **Honest.** Memory is retrieval, not learning. Injecting "the user's favourite language is
  Python" into the context is not the model *remembering* -- it is the system reminding it.

## Where knowledge lives

```
data/raw/knowledge/     <- your documents go here (this alone does nothing on its own)
data/processed/         <- tokenised training data, if you chose route A
data/memory/            <- conversation memory (gitignored, private)
```

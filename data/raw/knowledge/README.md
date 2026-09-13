# Your knowledge goes here

Put plain-text files in this folder (`programming.txt`, `notes.txt`, `projects.txt`, ...).

**Putting a file here does not train the model.** It puts a file in a folder.

Two separate things can then be done with it, and they are genuinely different -- see
[../../../KNOWLEDGE.md](../../../KNOWLEDGE.md):

- **Training** (`scripts/prepare_data.py`) turns these files into training data and changes the
  model's weights during a training run.
- **Retrieval** (Phase 10) searches these files at conversation time and shows the model the
  relevant passages. The weights do not change.

Neither is built yet.

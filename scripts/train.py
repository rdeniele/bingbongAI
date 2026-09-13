"""Train BingBongAI.

    python scripts/train.py --config configs/synthetic.yaml
    python scripts/train.py --config configs/synthetic.yaml --resume checkpoints/synthetic/latest.pt

Prerequisites, in order:
    python scripts/train_tokenizer.py --config <config>
    python scripts/prepare_data.py    --config <config>

Outputs:
    checkpoints/<name>/latest.pt      most recent state (resume from this)
    checkpoints/<name>/best.pt        lowest validation loss so far
    checkpoints/<name>/step_*.pt      the last `keep_last` periodic checkpoints
    runs/<name>/metrics.jsonl         every logged measurement, one JSON per line
    runs/<name>/log.txt               the console output

ON RESUME
    The architecture and tokenizer must match the checkpoint exactly; that is
    checked. Training settings come from --config, which is how a finished run is
    extended: raise max_steps in the config and resume. (Changing max_steps also
    reshapes the cosine schedule from that point on.)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from src.model.language_model import BingBongLM  # noqa: E402
from src.tokenizer.tokenizer import BPETokenizer  # noqa: E402
from src.training.checkpoint import CheckpointError  # noqa: E402
from src.training.dataloader import BatchLoader  # noqa: E402
from src.training.dataset import TokenDataset, read_meta  # noqa: E402
from src.training.trainer import Trainer, TrainingSettings  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logging import RunLogger  # noqa: E402


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise SystemExit("config requests device 'cuda' but CUDA is not available")
    return torch.device(requested)


def build_trainer(config, device: torch.device, logger: RunLogger, max_steps: int | None = None):
    """Assemble tokenizer, data, model and trainer from a config. Shared with prove_learning.py."""
    settings = TrainingSettings.from_config(config)
    if max_steps is not None:
        settings = TrainingSettings(**{**settings.__dict__, "max_steps": max_steps})

    tokenizer = BPETokenizer.load(config.tokenizer.vocab_path)
    meta = read_meta(config.data.meta_path)
    if meta["tokenizer_fingerprint"] != tokenizer.fingerprint():
        raise SystemExit(
            f"token files in {config.data.meta_path} were made with tokenizer "
            f"{meta['tokenizer_fingerprint']}, but {config.tokenizer.vocab_path} is "
            f"{tokenizer.fingerprint()}. Re-run scripts/prepare_data.py."
        )
    if tokenizer.vocab_size > config.tokenizer.vocab_size:
        raise SystemExit(
            f"tokenizer vocab {tokenizer.vocab_size} exceeds model vocab {config.tokenizer.vocab_size}"
        )

    T = config.model.context_length
    train_set = TokenDataset(config.data.train_path, T)
    val_set = TokenDataset(config.data.val_path, T)

    # Seed before building the model: the random initialization is reproducible.
    torch.manual_seed(settings.seed)
    model = BingBongLM.from_config(config).to(device)

    train_loader = BatchLoader(train_set, settings.batch_size, device, seed=settings.seed)
    val_loader = BatchLoader(val_set, settings.batch_size, device, seed=settings.seed + 1)

    trainer = Trainer(
        model, settings, train_loader, val_loader, device,
        config_dict=config.to_dict(),
        tokenizer_fingerprint=tokenizer.fingerprint(),
        checkpoint_dir=Path(config.paths.checkpoint_dir) / config.name,
        logger=logger,
    )
    dataset_tokens = {"train": train_set.num_tokens, "val": val_set.num_tokens}
    return trainer, tokenizer, dataset_tokens


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BingBongAI.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None, help="checkpoint to continue from")
    parser.add_argument("--max-steps", type=int, default=None, help="override training.max_steps")
    parser.add_argument("--device", default=None, help="override training.device (auto|cuda|cpu)")
    args = parser.parse_args()

    config = load_config(args.config)
    device = resolve_device(args.device or config.training.get("device", "auto"))
    logger = RunLogger(Path(config.paths.get("run_dir", "runs")) / config.name)

    trainer, _, dataset_tokens = build_trainer(config, device, logger, args.max_steps)
    if args.resume:
        try:
            trainer.resume(args.resume)
        except CheckpointError as error:
            raise SystemExit(f"cannot resume: {error}") from None

    trainer.describe(dataset_tokens)
    try:
        summary = trainer.fit()
    except KeyboardInterrupt:
        path = trainer.save("latest.pt")
        logger.info(f"\nInterrupted at step {trainer.step}. Saved {path}. Resume with:")
        logger.info(f"  python scripts/train.py --config {args.config} --resume {path}")
        raise SystemExit(130) from None
    finally:
        logger.close()

    print()
    print(f"  Finished at step {summary['step']}")
    print(f"  Final train loss: {summary['final_train_loss']:.4f}")
    print(f"  Final val loss:   {summary['final_val_loss']:.4f}")
    print(f"  Best val loss:    {summary['best_val_loss']:.4f}")
    print(f"  Wall time:        {summary['wall_seconds_this_session']:.1f}s "
          f"for {summary['steps_this_session']} steps")
    print(f"  Checkpoints:      {trainer.checkpoint_dir}")


if __name__ == "__main__":
    main()

"""Chat with BingBongAI in the terminal. Fully offline.

    python scripts/chat.py
    python scripts/chat.py --checkpoint checkpoints/small/best.pt --temperature 0.7

No network access is used or needed: the model, tokenizer and conversation all
live on this machine. tests/test_chat.py verifies a full chat turn with the
network forcibly disabled.

Commands inside the chat:
    exit / quit     leave
    /reset          forget the conversation
    /history        print the conversation so far
    /context        show EXACTLY what the model saw for its last reply
    /settings       show decoding settings
    /temp X         set temperature (0 = greedy)
    /topk K         set top-k (0 = off)
    /seed N         fixed seed for reproducible replies (/seed off to go random)
    /help           this list
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from src.inference.chat import ChatSession  # noqa: E402
from src.inference.generate import GenerationConfig  # noqa: E402
from src.inference.model_loader import load_for_inference  # noqa: E402
from src.training.checkpoint import CheckpointError  # noqa: E402

DEFAULT_CHECKPOINT = "checkpoints/synthetic/best.pt"

BANNER = """\
================================
        BingBongAI
      Offline AI Assistant
================================"""

HELP = """\
  exit / quit   leave            /history    show the conversation
  /reset        forget it all    /context    show exactly what the model sees
  /settings     show settings    /temp X  /topk K  /seed N|off   change decoding"""


def say(text: str = "", end: str = "\n") -> None:
    """Print without crashing on characters the Windows console cannot show."""
    try:
        print(text, end=end, flush=True)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "ascii"
        print(text.encode(encoding, errors="backslashreplace").decode(encoding), end=end, flush=True)


def describe_settings(session: ChatSession) -> str:
    g = session.generation
    if g.greedy:
        mode = "greedy (deterministic)"
    else:
        mode = (f"temperature {g.temperature}, top-k {g.top_k if g.top_k else 'off'}, "
                f"seed {g.seed if g.seed is not None else 'random'}")
    return (f"  decoding: {mode}\n"
            f"  reply limit: {g.max_new_tokens} tokens | context window: "
            f"{session.model.cfg.context_length} tokens")


def handle_command(session: ChatSession, line: str) -> bool:
    """Run a /command. Returns False if the chat should end."""
    parts = line.split()
    command, args = parts[0].lower(), parts[1:]
    g = session.generation

    try:
        if command in {"/help", "/?"}:
            say(HELP)
        elif command == "/reset":
            session.reset()
            say("  (conversation cleared)")
        elif command == "/history":
            if not session.history:
                say("  (empty)")
            for m in session.history:
                who = "You" if m.role == "user" else "BingBongAI"
                say(f"  {who}: {m.text}")
        elif command == "/context":
            info = session.last_prompt
            if info is None:
                say("  (nothing yet - send a message first)")
            else:
                say(f"  The last reply was generated from exactly this input:")
                say(f"  {info.tokens} tokens of {info.budget} available "
                    f"({info.turns_included} turns in view, {info.turns_dropped} dropped"
                    f"{', newest message truncated' if info.newest_message_truncated else ''})")
                say("  ---- begin model input ----")
                say(info.text)
                say("  ---- end model input (the model continued from here) ----")
        elif command == "/settings":
            say(describe_settings(session))
        elif command == "/temp":
            session.generation = replace(g, temperature=float(args[0]))
            say(describe_settings(session))
        elif command == "/topk":
            k = int(args[0])
            session.generation = replace(g, top_k=None if k == 0 else k)
            say(describe_settings(session))
        elif command == "/seed":
            seed = None if args[0].lower() == "off" else int(args[0])
            session.generation = replace(g, seed=seed)
            say(describe_settings(session))
        else:
            say(f"  unknown command {command}. Type /help.")
    except (IndexError, ValueError) as error:
        say(f"  could not apply {command}: {error or 'missing value'}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline chat with BingBongAI.")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40, help="0 disables top-k")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--system", default="", help="optional text placed before the conversation")
    parser.add_argument("--device", default="auto", help="auto | cuda | cpu")
    parser.add_argument("--verbose", action="store_true", help="print token/timing stats after each reply")
    args = parser.parse_args()

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))
    try:
        loaded = load_for_inference(args.checkpoint, device)
    except (CheckpointError, FileNotFoundError) as error:
        raise SystemExit(f"cannot load model: {error}") from None

    try:
        session = ChatSession(
            loaded.model, loaded.tokenizer,
            GenerationConfig(max_new_tokens=args.max_new_tokens,
                             temperature=args.temperature,
                             top_k=None if args.top_k == 0 else args.top_k,
                             seed=args.seed),
            system_prompt=args.system,
        )
    except ValueError as error:
        raise SystemExit(f"invalid setting: {error}") from None

    say(BANNER)
    say(f"\n  model: {loaded.checkpoint_path} ({loaded.config_name}, step {loaded.step}, "
        f"{loaded.model.num_parameters():,} parameters)")
    say(f"  running on: {torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'} "
        f"- no internet connection used")
    say(describe_settings(session))
    if loaded.config_name == "synthetic":
        say("\n  NOTE: this checkpoint was trained only on 33 synthetic sentences such as")
        say("  'The color of snow is white.' It has never seen a conversation, so it")
        say("  cannot understand questions yet. Expect it to answer with those patterns.")
    say("\n  Type /help for commands, exit to quit.")

    while True:
        try:
            line = input("\nYou: ")
        except (EOFError, KeyboardInterrupt):
            say()
            break
        text = line.strip()
        if not text:
            continue
        if text.lower() in {"exit", "quit"}:
            break
        if text.startswith("/"):
            handle_command(session, text)
            continue

        say("\nBingBongAI: ", end="")
        holder: dict = {}
        wrote = False
        try:
            for piece in session.stream_reply(text, turn=holder):
                say(piece, end="")
                wrote = True
        except KeyboardInterrupt:
            say("\n  (reply interrupted)")
            continue
        if not wrote:
            say("(no reply)", end="")
        say()

        if args.verbose and "turn" in holder:
            turn = holder["turn"]
            r, p = turn.result, turn.prompt
            say(f"  [{len(r.token_ids)} tokens, {r.tokens_per_second:.0f} tok/s, stop: {r.stop_reason}, "
                f"{'greedy' if r.seed is None else f'seed {r.seed}'} | prompt {p.tokens}/{p.budget} tokens, "
                f"{p.turns_included} turns in view, {p.turns_dropped} dropped]")

    say("Goodbye.")


if __name__ == "__main__":
    main()

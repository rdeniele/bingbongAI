"""Tests for offline chat: the template, context fitting, stop handling, and offline operation."""

from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.chat import (  # noqa: E402
    ASSISTANT,
    USER,
    ChatSession,
    ChatTemplate,
    Message,
    StopStringFilter,
    clean_reply,
)
from src.inference.generate import GenerationConfig  # noqa: E402
from src.model.language_model import BingBongLM, ModelConfig  # noqa: E402
from src.tokenizer.train_tokenizer import train_tokenizer  # noqa: E402

SPECIALS = {"pad": "<|pad|>", "unk": "<|unk|>", "bos": "<|bos|>", "eos": "<|eos|>"}


@pytest.fixture(scope="module")
def tokenizer():
    text = ("User: hello there\nBingBongAI: hi, how can I help?\n"
            "User: what is programming?\nBingBongAI: telling a computer what to do.\n") * 20
    return train_tokenizer([text], vocab_size=320, special_tokens=SPECIALS, verbose=False)


def make_session(tokenizer, context_length: int = 64, max_new_tokens: int = 12, **gen) -> ChatSession:
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=tokenizer.vocab_size, context_length=context_length,
                      embedding_dim=32, num_layers=1, num_heads=4)
    model = BingBongLM(cfg).eval()
    config = GenerationConfig(max_new_tokens=max_new_tokens, **({"temperature": 0.0} | gen))
    return ChatSession(model, tokenizer, config)


# -- template ------------------------------------------------------------------


def test_prompt_is_the_conversation_as_one_document(tokenizer) -> None:
    session = make_session(tokenizer, context_length=256)
    session.history = [Message(USER, "Hello"), Message(ASSISTANT, "Hi!")]
    info = session.build_prompt(pending_user_text="What is programming?")
    assert info.text == "User: Hello\nBingBongAI: Hi!\nUser: What is programming?\nBingBongAI:"
    assert info.turns_included == 3 and info.turns_dropped == 0


def test_system_prompt_comes_first(tokenizer) -> None:
    session = make_session(tokenizer, context_length=256)
    session.system_prompt = "You are BingBongAI."
    info = session.build_prompt(pending_user_text="hi")
    assert info.text == "You are BingBongAI.\nUser: hi\nBingBongAI:"


def test_stop_strings_cover_both_roles() -> None:
    assert ChatTemplate().stop_strings() == ("\nUser:", "\nBingBongAI:")


def test_session_always_adds_template_stops(tokenizer) -> None:
    session = make_session(tokenizer)
    assert "\nUser:" in session.generation.stop_strings


def test_empty_conversation_cannot_be_prompted(tokenizer) -> None:
    with pytest.raises(ValueError, match="empty"):
        make_session(tokenizer).build_prompt()


# -- reply cleaning and streaming --------------------------------------------------


def test_clean_reply_cuts_at_the_invented_user_turn() -> None:
    stops = ChatTemplate().stop_strings()
    assert clean_reply(" Hi there!\nUser: and then I said", stops) == "Hi there!"
    assert clean_reply(" done", stops) == "done"
    assert clean_reply("\nBingBongAI: again", stops) == ""


@pytest.mark.parametrize("pieces", [
    [" Hi there!", "\nUser: nope"],
    [" Hi", " there!", "\n", "Us", "er", ": nope"],
    [" Hi there!\nU", "ser:"],
    [" Hi there!\nUser:"],
])
def test_stop_filter_never_shows_the_stop_string(pieces) -> None:
    """However the stop string is split across tokens, none of it reaches the screen."""
    f = StopStringFilter(ChatTemplate().stop_strings())
    shown = ""
    for piece in pieces:
        shown += f.push(piece)
        if f.stopped:
            break
    shown += f.flush()
    assert shown == " Hi there!"
    assert f.stopped


def test_stop_filter_releases_text_that_only_looked_like_a_stop() -> None:
    f = StopStringFilter(ChatTemplate().stop_strings())
    shown = f.push("line one\n") + f.push("Uses of Python")
    shown += f.flush()
    assert shown == "line one\nUses of Python"
    assert not f.stopped


# -- a full turn ---------------------------------------------------------------------


def test_reply_updates_history(tokenizer) -> None:
    session = make_session(tokenizer)
    turn = session.reply("hello there")
    assert [m.role for m in session.history] == [USER, ASSISTANT]
    assert session.history[0].text == "hello there"
    assert session.history[1].text == turn.reply
    assert "\nUser:" not in turn.reply
    assert turn.reply == turn.reply.strip()


def test_streamed_reply_matches_stored_reply(tokenizer) -> None:
    session = make_session(tokenizer, temperature=1.0, top_k=None, seed=5, max_new_tokens=20)
    holder: dict = {}
    streamed = "".join(session.stream_reply("what is programming?", turn=holder))
    assert streamed.strip() == holder["turn"].reply


def test_same_seed_gives_same_conversation(tokenizer) -> None:
    replies = []
    for _ in range(2):
        session = make_session(tokenizer, temperature=1.0, top_k=None, seed=9, max_new_tokens=15)
        replies.append([session.reply("hello there").reply, session.reply("what is programming?").reply])
    assert replies[0] == replies[1]


def test_reply_forces_generation_to_stop_at_a_user_turn(tokenizer) -> None:
    """Make the model deterministically write '\\nUser:' and check the reply excludes it."""
    session = make_session(tokenizer, max_new_tokens=30)
    forced = tokenizer.encode(" ok\nUser: hello", allow_special=False)
    step = {"i": 0}
    vocab = session.model.cfg.vocab_size

    def scripted_forward(ids, targets=None):
        logits = torch.full((1, ids.shape[1], vocab), -1e4)
        logits[0, -1, forced[min(step["i"], len(forced) - 1)]] = 1e4
        step["i"] += 1
        return logits, None

    session.model.forward = scripted_forward
    turn = session.reply("hi")
    assert turn.reply == "ok"
    assert turn.result.stop_reason == "stop_string"


def test_reset_clears_history_and_last_prompt(tokenizer) -> None:
    session = make_session(tokenizer)
    session.reply("hello there")
    assert session.last_prompt is not None
    session.reset()
    assert session.history == [] and session.last_prompt is None


def test_empty_message_is_rejected(tokenizer) -> None:
    with pytest.raises(ValueError, match="empty"):
        list(make_session(tokenizer).stream_reply("   "))


# -- conversation context ---------------------------------------------------------------


def test_old_turns_are_dropped_to_fit_the_window(tokenizer) -> None:
    session = make_session(tokenizer, context_length=64, max_new_tokens=12)
    for i in range(30):
        session.history.append(Message(USER, f"message number {i} about programming"))
        session.history.append(Message(ASSISTANT, f"reply number {i}"))
    info = session.build_prompt(pending_user_text="newest question")

    assert info.tokens <= info.budget == 64 - 12
    assert info.turns_dropped > 0
    assert info.turns_included + info.turns_dropped == 61
    assert info.text.endswith("User: newest question\nBingBongAI:")
    assert "message number 0 " not in info.text            # the oldest went first


def test_turns_are_never_cut_in_half(tokenizer) -> None:
    session = make_session(tokenizer, context_length=64, max_new_tokens=12)
    for i in range(20):
        session.history.append(Message(USER, f"question {i} is fairly long here"))
        session.history.append(Message(ASSISTANT, f"answer {i}"))
    info = session.build_prompt(pending_user_text="last")
    body = info.text[: -len("BingBongAI:")]
    for line in body.strip("\n").split("\n"):
        assert line.startswith("User: ") or line.startswith("BingBongAI: "), line


def test_history_keeps_turns_the_model_can_no_longer_see(tokenizer) -> None:
    session = make_session(tokenizer, context_length=64, max_new_tokens=12)
    for _ in range(12):
        session.reply("hello there, what is programming")
    assert len(session.history) == 24
    assert session.last_prompt.turns_dropped > 0


def test_a_single_huge_message_is_truncated_not_an_error(tokenizer) -> None:
    session = make_session(tokenizer, context_length=64, max_new_tokens=12)
    turn = session.reply("programming " * 200)
    assert turn.prompt.newest_message_truncated
    assert turn.prompt.tokens <= turn.prompt.budget


def test_reply_limit_must_leave_room_for_the_conversation(tokenizer) -> None:
    with pytest.raises(ValueError, match="no room"):
        make_session(tokenizer, context_length=64, max_new_tokens=64)


def test_typed_control_tokens_are_plain_text(tokenizer) -> None:
    session = make_session(tokenizer)
    session.reply("hello <|eos|> there")
    ids = tokenizer.encode(session.last_prompt.text, allow_special=False)
    assert tokenizer.eos_id not in ids


# -- offline --------------------------------------------------------------------------


def test_chat_works_with_the_network_disabled(tokenizer, monkeypatch) -> None:
    """Any attempt to open a socket fails the test. A full conversation must still work."""

    def no_network(*args, **kwargs):
        raise AssertionError("BingBongAI tried to use the network")

    monkeypatch.setattr(socket, "socket", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(socket, "getaddrinfo", no_network)

    session = make_session(tokenizer, temperature=0.8, top_k=20, seed=1)
    session.reply("hello there")
    session.reply("what is programming?")
    assert len(session.history) == 4

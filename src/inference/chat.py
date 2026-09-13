"""Offline chat: turning a next-token predictor into a conversation.

THE MODEL DOES NOT KNOW IT IS CHATTING
    BingBongAI is a text continuer. It has no built-in notion of "user",
    "assistant", "turn" or "question". Chat is a FORMAT wrapped around it: the
    conversation so far is written out as one plain document, ending exactly where
    the assistant's reply should begin, and the model continues that document.

        User: Hello
        BingBongAI: Hi! How can I help?
        User: What is programming?
        BingBongAI:                         <- the model continues from here

    Generation stops when the model starts writing the next "User:" line (it is
    continuing the document, so it will happily invent the user's side too), or
    at <|eos|>, or at the token limit. Whatever came before that stop is the reply.

WHY A PLAIN-TEXT TEMPLATE
    Dedicated role tokens (<|user|>, <|assistant|>) are cleaner, but adding tokens
    changes the vocabulary, which means retraining the tokenizer AND the model.
    "User: " / "BingBongAI: " work with the existing tokenizer today. The one hard
    requirement is that conversational TRAINING data uses exactly this template --
    that is what would teach the model what a reply looks like. So the template
    lives here, in one place, for training-data preparation to reuse.

    The known weakness: a user can type "\\nBingBongAI: ..." inside a message and
    fake a turn. With role tokens that is impossible, because users cannot type a
    control token (encode() uses allow_special=False). Acceptable for a local,
    single-user assistant; worth revisiting with role tokens later.

CONVERSATION CONTEXT
    The model sees at most `context_length` (512) tokens. Room is reserved for the
    reply (`max_new_tokens`), and the conversation is fitted into what remains:

      - the newest user message is always included
      - earlier turns are added newest-first while they fit, as WHOLE turns --
        never cut mid-message, since half a sentence misleads more than none
      - turns that do not fit are dropped from the prompt (not from history)
      - a single message too long to fit alone keeps only its final tokens

    Dropped turns are gone as far as the model is concerned. It is not "forgetting"
    in any gradual sense: those tokens are simply not in its input. Remembering
    things beyond the window is the job of local memory (Phase 11), which is a
    separate system, not the neural network.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from .generate import GenerationConfig, GenerationResult, stream_tokens

USER = "user"
ASSISTANT = "assistant"


@dataclass(frozen=True)
class ChatTemplate:
    user_prefix: str = "User: "
    assistant_prefix: str = "BingBongAI: "
    separator: str = "\n"

    def render_message(self, role: str, text: str) -> str:
        prefix = self.user_prefix if role == USER else self.assistant_prefix
        return f"{prefix}{text}{self.separator}"

    def reply_opening(self) -> str:
        """The text the prompt ends with: the assistant's turn, begun but empty.

        The trailing space is removed. With a space-prefixing tokenizer the model
        learned " word" tokens (space attached to the word), so ending the prompt
        on a bare space would force an unnatural token split at the very first word.
        """
        return self.assistant_prefix.rstrip(" ")

    def stop_strings(self) -> tuple[str, ...]:
        """Text that means the model has moved past its own turn."""
        return (
            self.separator + self.user_prefix.rstrip(" "),
            self.separator + self.assistant_prefix.rstrip(" "),
        )


@dataclass
class Message:
    role: str
    text: str


@dataclass
class PromptInfo:
    """Exactly what was fed to the model for one reply."""

    text: str
    tokens: int
    budget: int                   # tokens available for the conversation
    turns_included: int
    turns_dropped: int
    newest_message_truncated: bool


@dataclass
class ChatTurn:
    reply: str
    prompt: PromptInfo
    result: GenerationResult


class StopStringFilter:
    """Streams text while never showing any part of a stop string.

    "\\nUser:" arrives across several tokens. By the time it is complete, "\\nUser"
    would already be on screen. So text that could still turn out to be the start
    of a stop string is held back until it either completes one (then dropped) or
    diverges (then released).
    """

    def __init__(self, stop_strings: tuple[str, ...]) -> None:
        self.stop_strings = tuple(s for s in stop_strings if s)
        self.buffer = ""
        self.stopped = False

    def push(self, piece: str) -> str:
        if self.stopped:
            return ""
        self.buffer += piece
        cut = min((self.buffer.find(s) for s in self.stop_strings if s in self.buffer), default=-1)
        if cut >= 0:
            self.stopped = True
            out, self.buffer = self.buffer[:cut], ""
            return out

        hold = 0
        for stop in self.stop_strings:
            for length in range(min(len(stop) - 1, len(self.buffer)), 0, -1):
                if self.buffer.endswith(stop[:length]):
                    hold = max(hold, length)
                    break
        out = self.buffer[: len(self.buffer) - hold]
        self.buffer = self.buffer[len(self.buffer) - hold:]
        return out

    def flush(self) -> str:
        if self.stopped:
            return ""
        out, self.buffer = self.buffer, ""
        return out


def clean_reply(text: str, stop_strings: tuple[str, ...]) -> str:
    """Cut at the first stop string and trim surrounding whitespace."""
    cut = len(text)
    for stop in stop_strings:
        index = text.find(stop)
        if index >= 0:
            cut = min(cut, index)
    return text[:cut].strip()


class ChatSession:
    def __init__(
        self,
        model,
        tokenizer,
        generation: GenerationConfig,
        template: ChatTemplate | None = None,
        system_prompt: str = "",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.template = template or ChatTemplate()
        self.system_prompt = system_prompt
        self.history: list[Message] = []
        self.last_prompt: PromptInfo | None = None     # exactly what the last reply was generated from
        self.generation = generation  # validated below via the setter

    # -- settings -----------------------------------------------------------

    @property
    def generation(self) -> GenerationConfig:
        return self._generation

    @generation.setter
    def generation(self, config: GenerationConfig) -> None:
        context = self.model.cfg.context_length
        if config.max_new_tokens >= context:
            raise ValueError(
                f"max_new_tokens ({config.max_new_tokens}) must be smaller than the context "
                f"length ({context}), or there is no room left for the conversation"
            )
        stops = tuple(dict.fromkeys(config.stop_strings + self.template.stop_strings()))
        self._generation = GenerationConfig(
            max_new_tokens=config.max_new_tokens, temperature=config.temperature,
            top_k=config.top_k, seed=config.seed, stop_strings=stops,
        )

    def reset(self) -> None:
        self.history.clear()
        self.last_prompt = None

    def _count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, allow_special=False))

    # -- context ------------------------------------------------------------

    def build_prompt(self, pending_user_text: str | None = None) -> PromptInfo:
        """Fit the conversation into the context window, newest turns first."""
        messages = list(self.history)
        if pending_user_text is not None:
            messages.append(Message(USER, pending_user_text))
        if not messages:
            raise ValueError("nothing to reply to: the conversation is empty")

        t = self.template
        context = self.model.cfg.context_length
        budget = context - self.generation.max_new_tokens
        header = f"{self.system_prompt}{t.separator}" if self.system_prompt else ""
        opening = t.reply_opening()
        fixed = self._count(header) + self._count(opening)

        rendered = [t.render_message(m.role, m.text) for m in messages]
        costs = [self._count(r) for r in rendered]

        newest_truncated = False
        remaining = budget - fixed
        if costs[-1] > remaining:
            # Even the newest message alone does not fit: keep only its ending.
            keep = max(1, remaining)
            ids = self.tokenizer.encode(rendered[-1], allow_special=False)[-keep:]
            rendered[-1] = self.tokenizer.decode(ids)
            costs[-1] = self._count(rendered[-1])
            newest_truncated = True

        chosen = [rendered[-1]]
        used = costs[-1]
        for text, cost in zip(reversed(rendered[:-1]), reversed(costs[:-1])):
            if used + cost > remaining:
                break
            chosen.append(text)
            used += cost
        chosen.reverse()

        prompt = header + "".join(chosen) + opening
        return PromptInfo(
            text=prompt,
            tokens=self._count(prompt),
            budget=budget,
            turns_included=len(chosen),
            turns_dropped=len(messages) - len(chosen),
            newest_message_truncated=newest_truncated,
        )

    # -- a turn -------------------------------------------------------------

    def stream_reply(self, user_text: str, turn: dict | None = None) -> Iterator[str]:
        """Add the user's message, then yield the reply as it is generated.

        After the generator is exhausted, `turn["turn"]` holds the ChatTurn.
        """
        user_text = user_text.strip()
        if not user_text:
            raise ValueError("message is empty")

        prompt = self.build_prompt(pending_user_text=user_text)
        self.last_prompt = prompt
        self.history.append(Message(USER, user_text))

        result = GenerationResult(prompt=prompt.text, text="", token_ids=[], prompt_tokens=0,
                                  prompt_truncated=False, stop_reason="max_new_tokens",
                                  seed=None, seconds=0.0)
        stops = self.generation.stop_strings
        visible = StopStringFilter(stops)
        started_reply = False

        pieces = stream_tokens(self.model, self.tokenizer, prompt.text, self.generation,
                               result=result)
        try:
            for piece in pieces:
                out = visible.push(piece)
                if not started_reply:
                    out = out.lstrip()          # the model usually opens with a space
                    started_reply = bool(out)
                if out:
                    yield out
                if visible.stopped:
                    break
        finally:
            pieces.close()                      # finalises timing and result.text now

        tail = visible.flush()
        if tail:
            yield tail if started_reply else tail.lstrip()

        reply = clean_reply(result.text, stops)
        self.history.append(Message(ASSISTANT, reply))
        if turn is not None:
            turn["turn"] = ChatTurn(reply=reply, prompt=prompt, result=result)

    def reply(self, user_text: str) -> ChatTurn:
        holder: dict = {}
        for _ in self.stream_reply(user_text, turn=holder):
            pass
        return holder["turn"]

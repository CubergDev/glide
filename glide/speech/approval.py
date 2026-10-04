"""The yes a voice request needs before it may click and type.

Hands-free listening has no wake word and no speaker check, so any audible speech (a video playing nearby, a
colleague) can ask Glide for something. With `act` on, a request that becomes a computer task therefore runs as a dry
run first, which only describes its first move; the task runs for real only after the person has said the confirmation
phrase, or pressed the key that calls `VoiceLoop.confirm()`. This module is that window and nothing else:

- It opens only when a dry run has ended and its description has been heard, and it closes on the first answer of any
  kind: one yes, for one task. A different sentence, a stop, a late answer, a pause, Glide being talked over, the loop
  ending, or the time running out (`timeout_s`) is a no, and nothing is done.
- The phrase has to be the whole utterance, compared the way stop phrases are (`router.normalize`), and at least two
  words, so a stray "yes" or "okay" never confirms anything.
- Text the person sees says what to say and for how long. Nothing here is spoken: speaking needs the assistant.

It holds the task's words (the transcript of the request) only while the window is open, in memory.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from ..assistant.phrases import say
from ..assistant.router import normalize


@dataclass(frozen=True)
class Pending:
    """A previewed task waiting for its yes: what was asked, in what language, and until when."""

    text: str
    language: str | None
    deadline: float
    goal: str = ""  # the task that was previewed: the one the yes runs


class TaskApproval:
    def __init__(
        self,
        phrase: str,
        timeout_s: float,
        *,
        clock: Callable[[], float] = time.monotonic,
        show: Callable[[str], None] = lambda text: None,
    ) -> None:
        self.phrase = normalize(phrase)
        self._phrase_text = phrase.strip()
        self.timeout_s = timeout_s
        self._clock = clock
        self._show = show
        self._lock = threading.Lock()
        self._pending: Pending | None = None

    @property
    def waiting(self) -> bool:
        return self._pending is not None

    def open(self, text: str, language: str | None, goal: str = "") -> None:
        """Start the window for a task whose dry run has been heard, and tell the person what to say."""
        with self._lock:
            self._pending = Pending(text, language, self._clock() + self.timeout_s, goal)
        self._show(say("confirm_task", language, phrase=self._phrase_text, seconds=str(round(self.timeout_s))))

    def answer(self, utterance: str) -> Pending | None:
        """What was just said, while a window may be open: the task to run if it is exactly the phrase and in time.
        Whatever it was, the window is closed."""
        pending = self._take()
        if pending is None:
            return None
        if self._clock() <= pending.deadline and normalize(utterance) == self.phrase:
            return pending
        self._show(say("confirm_lapsed", pending.language))
        return None

    def confirm(self) -> Pending | None:
        """The keyed yes: the task to run if a window is open and in time."""
        pending = self._take()
        if pending is None:
            return None
        if self._clock() > pending.deadline:
            self._show(say("confirm_lapsed", pending.language))
            return None
        return pending

    def cancel(self) -> None:
        """A no: the person spoke over Glide, paused, stopped, or the loop is ending."""
        pending = self._take()
        if pending is not None:
            self._show(say("confirm_lapsed", pending.language))

    def expire(self) -> None:
        """Close a window whose time has run out (called often)."""
        with self._lock:
            pending = self._pending
            if pending is None or self._clock() <= pending.deadline:
                return
            self._pending = None
        self._show(say("confirm_lapsed", pending.language))

    def _take(self) -> Pending | None:
        with self._lock:
            pending, self._pending = self._pending, None
        return pending

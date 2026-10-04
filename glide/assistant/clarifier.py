"""The channel for the router's `clarify` route: one question, shown and spoken, then a wait for `answer_pending`.

`glide.routing.resolve` asks through the `Clarifier` protocol (`ask(question) -> str | None`). This is the assistant's
implementation, one per request. The reply comes ONLY from `Assistant.answer_pending`: a request made through
`handle_text` or `handle_audio` is a new request and never the answer. `ask` returns None, and the router then does
nothing and says what is needed, when the request is cancelled (a stop, or a newer request, which cancels the older
one), when the wait times out, or when the question could not be put.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

from ..computer.control import RunControl
from ..computer.models import Abort

DEFAULT_WAIT_S = 120.0  # how long a question waits for its answer before the request is left undone


class PendingQuestion:
    def __init__(
        self,
        present: Callable[[str], None],
        control: RunControl,
        *,
        wait_s: float = DEFAULT_WAIT_S,
        on_open: Callable[[PendingQuestion], None] = lambda q: None,
        on_close: Callable[[PendingQuestion], None] = lambda q: None,
    ) -> None:
        self._present = present  # shows and says the question
        self._control = control  # the request's: its cancel wakes the wait
        self._wait_s = wait_s
        self._on_open, self._on_close = on_open, on_close
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._reply: str | None = None
        self.question: str | None = None  # set while the question is open

    def ask(self, question: str) -> str | None:
        with self._lock:
            self._event.clear()
            self._reply = None
            self.question = question
        try:
            with self._control.closing_on_cancel(self._event.set):  # Abort now if already cancelled
                self._on_open(self)
                self._present(question)
                self._event.wait(self._wait_s)
        except Abort:
            return None
        finally:
            with self._lock:
                reply, self.question = self._reply, None
            self._on_close(self)
        return None if self._control.cancelled.is_set() else reply

    def answer(self, text: str) -> bool:
        """Give the question its answer. False when no question is open (or it already has one)."""
        with self._lock:
            if self.question is None or self._event.is_set():
                return False
            self._reply = text
            self._event.set()
        return True

"""Follow-up questions about one pinned item. Read-only: no action runner, classifier or dispatch is imported.

One `PointSession` owns one pin at a time and answers questions about it, typed or transcribed, each on a daemon
worker thread (`ask`), because the model call can take as long as its deadline and the person must be able to stop,
ask again or leave meanwhile. The rules:

- A newer question, `stop`, `repoint` and `close` all supersede the answer in flight. A superseded answer is dropped
  where it arrives, under the session's lock, so a late reply can never reach the screen, the speaker or the history.
- Superseding also cancels the run's `RunControl`: whatever the provider call registered with `closing_on_cancel` is
  closed, and the worker is abandoned, not joined. `close()` therefore returns at once even while a provider is still
  holding the connection. (Chains that call providers through `RunControl.interruptible`, on branch
  consolidation/cancel, make the worker itself end promptly; until then it ends on its own deadline and its result is
  discarded.)
- The last four exchanges (`HISTORY`) go with the next question, so "and the second one?" makes sense. They belong to
  the pin: `repoint` clears them, and so does `close`. They are never written anywhere, whatever the recording
  setting: a pinned item's text and the questions about it exist in memory for as long as the pin does.
- A pin is trusted for `HARD_TTL_S` seconds (point_ask.py). The model is told how old the snapshot is (`age_s`), the
  answer event carries that age for the answer card, and an expired pin ends the session instead of being answered
  about.
- Provider failures and anything unexpected show a fixed sentence. What a provider said or what was on the screen is
  never put in an event.

Events go to `emit(kind, **data)` on whatever thread they happen on: `thinking`, `answer` (text, uncertain, age_s,
model, seconds), `error` (text, closed), `stopped`.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from collections.abc import Callable

from ..computer.control import RunControl, controlled
from ..computer.models import Abort
from ..computer.writer import WriterError
from .point_ask import MAX_QUESTION, PointSelection, PointStopped, PointUnavailable

HISTORY = 4
PROVIDER_FAILED = "The answer provider failed or returned an incomplete answer. Please try again."
UNEXPECTED = "Could not answer about this point. Please try again."
EXPIRED = "The pinned item expired. Select the point again."
TOO_LONG = "Question is too long. Ask a shorter question."


def _ignore(*_args, **_kwargs) -> None:
    return None


class PointSession:
    def __init__(
        self,
        selection: PointSelection,
        writer,
        emit: Callable[..., None],
        *,
        speak: Callable[[str], None] = _ignore,
        cancel_speech: Callable[[], None] = _ignore,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._selection, self._writer, self._emit = selection, writer, emit
        self._speak, self._cancel_speech, self._clock = speak, cancel_speech, clock
        self._lock = threading.RLock()
        self._history: deque[dict] = deque(maxlen=HISTORY)
        self._generation = 0
        self._control: RunControl | None = None
        self._closed = False
        self.failed = False  # the last answer failed (a retry is allowed); cleared by the next question

    @property
    def available(self) -> bool:
        return not self._closed

    @property
    def history(self) -> list[dict]:
        with self._lock:
            return list(self._history)

    @property
    def selection(self) -> PointSelection:
        return self._selection

    def ask(self, text: str) -> bool:
        """Answer `text` about the pin on a worker thread. False when nothing was started."""
        if not isinstance(text, str) or not text.strip():
            return False
        if len(text) > MAX_QUESTION:
            self._emit("error", text=TOO_LONG, closed=False)
            return False
        text = text.strip()
        with self._lock:
            if self._closed:
                return False
            self._supersede()
            self.failed = False
            generation, selection, history = self._generation, self._selection, list(self._history)
            control = self._control = RunControl(str(uuid.uuid4()))
            self._emit("thinking", text="Reading the pinned item. No computer actions.")
        threading.Thread(
            target=self._answer, args=(generation, control, selection, text, history), name="glide-point-answer", daemon=True
        ).start()
        return True

    def stop(self) -> None:
        """Cancel the answer in flight and the readout. The pin stays."""
        with self._lock:
            if self._closed:
                return
            self._supersede()
            self._emit("stopped", text="Answer stopped.")

    def repoint(self, selection: PointSelection) -> bool:
        """Pin a new item: whatever was in flight about the old one is dropped and its exchanges are forgotten.

        False, and the new selection is closed, when the session has ended."""
        with self._lock:
            accepted = not self._closed
            discarded = self._selection if accepted else selection
            if accepted:
                self._selection = selection
                self._supersede()
                self._history.clear()
                self.failed = False
        discarded.close()
        return accepted

    def close(self) -> None:
        """End the session. Returns at once: the answer in flight, if any, is abandoned and its result discarded."""
        with self._lock:
            self._close_locked()
        self._selection.close()

    # -- internals -------------------------------------------------------------------------------

    def _supersede(self) -> None:
        """Whatever is in flight is no longer wanted. Called with the lock held."""
        self._generation += 1
        control, self._control = self._control, None
        if control is not None:
            control.cancel()
        self._cancel_speech()

    def _close_locked(self) -> None:
        self._closed = True
        self._supersede()
        self._history.clear()

    def _answer(self, generation: int, control: RunControl, selection: PointSelection, text: str, history: list[dict]) -> None:
        started = self._clock()
        try:
            with controlled(control):
                answer = selection.ask(self._writer, text, allow_model=True, history=history)
            control.check(wait=False)
        except Abort:
            return  # a newer question, Stop or close owns the screen and the speaker now
        except PointStopped:
            self._finish(generation, "error", text=EXPIRED, closed=True, end=True)
        except PointUnavailable as error:  # our own sentence: a text-only model, a bad question
            self._finish(generation, "error", text=str(error), closed=False, failed=True)
        except WriterError:  # never what the provider said: it may echo the request or the screen
            self._finish(generation, "error", text=PROVIDER_FAILED, closed=False, failed=True)
        except Exception:
            self._finish(generation, "error", text=UNEXPECTED, closed=False, failed=True)
        else:
            with self._lock:
                if self._closed or generation != self._generation:
                    return  # late: dropped before it reaches the history, the screen or the speaker
                self._history.append({"question": text, "answer": answer.text})
                self._control = None
                self._emit(
                    "answer",
                    text=answer.text,
                    uncertain=answer.uncertain,
                    age_s=round(selection.age_s(), 1),
                    model=answer.model,
                    seconds=round(self._clock() - started, 3),
                )
                self._speak(answer.text)

    def _finish(self, generation: int, kind: str, *, end: bool = False, failed: bool = False, **data) -> None:
        with self._lock:
            if self._closed or generation != self._generation:
                return
            self._control = None
            if failed:
                self.failed = True
            if end:
                self.failed = True
                self._close_locked()
            self._emit(kind, **data)
        if end:
            self._selection.close()

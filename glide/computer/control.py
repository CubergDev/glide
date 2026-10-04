"""Run-scoped, thread-safe interruption shared by models, runners and native input (D2).

A `RunControl` belongs to one task. It is made available to every layer through a context variable
(`controlled`, `current_control`) so no signature has to carry it.

What it adds over the older stop hook, and why it is adopted
------------------------------------------------------------
- `checkpoint()` is called before each step, each model reply is used, and inside every platform input
  call, so a stop is seen between two actions and between two characters of typed text.
- `dispatch(action, ...)` gates each real mutation: it takes a checkpoint first, then marks the control
  `in_flight` before the action runs. After a stop that arrives mid-action the runner therefore knows the
  outcome of the last write is unknown. AGENTS.md says an attempted action is not a verified effect and an
  unknown write is never replayed: the runner reads the screen once, read-only, reports "completion
  unknown", and stops. `in_flight` is cleared only by a fresh observation (`action_checked`).
- `cancel()` runs the callbacks registered with `closing_on_cancel`, so a blocked network read (a model
  request, a CDP websocket) is closed instead of being waited out.
- `interruptible(call)` runs a blocking call on a helper thread and returns the moment the control is cancelled,
  whether or not the thread can be woken. A connection that is already open is closed by its own
  `closing_on_cancel` callback; a request that is still waiting for its first byte cannot be closed from
  outside, so its thread is left to end on its own deadline and its answer is thrown away. The provider chains
  (glide/providers/chain.py) call every provider through it, so one cancel reaches every model, speech and
  classifier call made under the control, and no adapter has to know about threads.

How it composes with `assistant.tasks.abort_on`
------------------------------------------------
`abort_on(stop_event)` replaces `desktop.check_abort` process-wide while a task runs. It is kept, because
it needs nothing but an event: it works for a caller that has no `RunControl` and for code that runs on
another thread, where the context variable is not set. Both mechanisms raise the same `Abort` and are
polled at the same places, because the platform adapters' own `check_abort` calls `checkpoint()` as well
as watching the corner of the screen. `ComputerTask.stop()` therefore does both, and gives both the same
reason (`cancel(reason)`; `abort_on(..., control)`), so the outcome string does not depend on which of
them fires first. The two never disagree about whether to stop: either one set is enough.

Cancelling is not rolling back. Nothing here undoes an action that was already sent.
"""

from __future__ import annotations

import contextvars
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import Future
from contextlib import contextmanager, suppress
from dataclasses import dataclass

from .models import Abort

CANCELLED = "task cancelled"


@dataclass(frozen=True)
class TaskEvent:
    task_id: str
    kind: str
    text: str = ""
    question_id: str = ""
    outcome: str = ""
    model: str = ""
    role: str = ""
    elapsed_s: float = 0.0
    spoken_text: str = ""


class _Registration:
    """One `closing_on_cancel` block's callback: it runs only while the block is open, and the block cannot end under it."""

    def __init__(self, close: Callable[[], None]) -> None:
        self._close = close
        self._live = True
        self._lock = threading.RLock()  # re-entrant: a callback may end the block that owns it

    def run(self) -> None:
        with self._lock:
            if self._live:
                with suppress(Exception):
                    self._close()

    def retire(self) -> None:
        with self._lock:
            self._live = False


class RunControl:
    def __init__(self, task_id: str = "", emit: Callable[[TaskEvent], None] | None = None):
        self.in_flight = False
        self.task_id = task_id
        self.emit = emit or (lambda event: None)
        self.cancelled = threading.Event()
        self.reason = CANCELLED
        self.ready = threading.Event()
        self.ready.set()
        self._callbacks: list[_Registration] = []  # in the order they were made
        self._lock = threading.Lock()  # guards the reason, the flag and the list; never held while a callback runs

    def cancel(self, reason: str = "") -> None:
        """Stop the task. The first reason given is the one every later check reports, however many threads cancel at once.

        The reason and the flag change together under one lock, so a second canceller finds the flag already set.
        Each callback registered with `closing_on_cancel` then runs once, unless its block has exited by then, and
        exiting a block waits for its own callback if it is running, so a callback never runs after its block is over
        (a connection that is back in a pool is not shut down by it).
        """
        with self._lock:
            if not self.cancelled.is_set():
                self.reason = reason or CANCELLED
                self.cancelled.set()
            registrations = tuple(self._callbacks)
        self.ready.set()
        for registration in registrations:
            registration.run()

    def pause(self) -> None:
        self.ready.clear()

    def resume(self) -> None:
        self.ready.set()

    def check(self, *, wait: bool = True) -> None:
        while True:
            if self.cancelled.is_set():
                raise Abort(self.reason)
            if not wait or self.ready.wait(0.05):
                if self.cancelled.is_set():
                    raise Abort(self.reason)
                return

    @contextmanager
    def closing_on_cancel(self, close: Callable[[], None]) -> Iterator[None]:
        """For the block, a cancel calls `close()` from the cancelling thread. `Abort` at once if already cancelled.

        When the block exits, `close` is not called again, and if it is being called right now the exit waits for it.
        """
        registration = _Registration(close)
        with self._lock:
            self._callbacks.append(registration)
        try:
            self.check(wait=False)
            yield
        finally:
            registration.retire()
            with self._lock:
                self._callbacks.remove(registration)

    def interruptible[R](self, call: Callable[[], R]) -> R:
        """`call()`, run on a helper thread, or `Abort` the moment this control is cancelled, whichever is first.

        An answer that arrives after the cancel is dropped, never returned. The helper is a daemon thread that
        sees this control as the current one, so what it opens can register its own `closing_on_cancel`. It is
        abandoned, not killed, when the cancel wins: it ends when its connection closes or its deadline passes.
        """
        woke = threading.Event()
        with self.closing_on_cancel(woke.set):  # Abort here if the control was cancelled already
            outcome = spawn(call, self)
            outcome.add_done_callback(lambda _: woke.set())
            woke.wait()
        self.check(wait=False)
        return outcome.result()

    def event(self, kind: str, text: str = "", **kwargs) -> None:
        self.emit(TaskEvent(self.task_id, kind, text, **kwargs))


_CURRENT: contextvars.ContextVar[RunControl | None] = contextvars.ContextVar("run_control", default=None)


def current_control() -> RunControl | None:
    return _CURRENT.get()


def checkpoint(control: RunControl | None = None, *, wait: bool = True) -> None:
    active = control if control is not None else _CURRENT.get()
    if active is not None:
        active.check(wait=wait)


@contextmanager
def controlled(control: RunControl | None) -> Iterator[None]:
    token = _CURRENT.set(control)
    try:
        yield
    finally:
        _CURRENT.reset(token)


def spawn[R](call: Callable[[], R], control: RunControl | None = None) -> Future[R]:
    """Run `call()` on a daemon thread, under `control`, and return the Future of its outcome.

    The context variables of the caller are copied, so code in the thread finds the same current control (or
    `control`, when one is given) that code in the caller would. A daemon thread never delays the end of the process.
    """
    outcome: Future[R] = Future()
    context = contextvars.copy_context()

    def work() -> None:
        try:
            with controlled(control if control is not None else current_control()):
                outcome.set_result(call())
        except BaseException as error:  # whatever it was is handed to whoever waits
            outcome.set_exception(error)

    threading.Thread(target=context.run, args=(work,), name="glide-call", daemon=True).start()
    return outcome


@contextmanager
def linked(parent: RunControl | None) -> Iterator[RunControl]:
    """A fresh control that is cancelled whenever `parent` is, and can be cancelled alone without touching it.

    That is what a request raced against another needs: the loser is cancelled by itself, the user's cancel
    reaches both. Raises `Abort` at once if `parent` is cancelled already.
    """
    child = RunControl(parent.task_id if parent is not None else "")
    if parent is None:
        yield child
        return
    with parent.closing_on_cancel(lambda: child.cancel(parent.reason)):
        yield child


def dispatch(action, *args, **kwargs):
    """Gate an actual mutation, and retain uncertainty until fresh observation."""
    checkpoint()
    active = current_control()
    if active:
        active.in_flight = True
    return action(*args, **kwargs)

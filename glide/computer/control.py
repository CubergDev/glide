"""Run-scoped, thread-safe interruption shared by models, runners and native input."""

from __future__ import annotations

import contextvars
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass

from .models import Abort


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


class RunControl:
    def __init__(self, task_id: str = "", emit: Callable[[TaskEvent], None] | None = None):
        self.in_flight = False
        self.task_id = task_id
        self.emit = emit or (lambda event: None)
        self.cancelled = threading.Event()
        self.ready = threading.Event()
        self.ready.set()
        self._callbacks: set[Callable[[], None]] = set()
        self._lock = threading.Lock()

    def cancel(self) -> None:
        self.cancelled.set()
        self.ready.set()
        with self._lock:
            callbacks = tuple(self._callbacks)
        for close in callbacks:
            with suppress(Exception):
                close()

    def pause(self) -> None:
        self.ready.clear()

    def resume(self) -> None:
        self.ready.set()

    def check(self, *, wait: bool = True) -> None:
        while True:
            if self.cancelled.is_set():
                raise Abort("task cancelled")
            if not wait or self.ready.wait(0.05):
                if self.cancelled.is_set():
                    raise Abort("task cancelled")
                return

    @contextmanager
    def closing_on_cancel(self, close: Callable[[], None]) -> Iterator[None]:
        with self._lock:
            self._callbacks.add(close)
        try:
            self.check(wait=False)
            yield
        finally:
            with self._lock:
                self._callbacks.discard(close)

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


def dispatch(action, *args, **kwargs):
    """Gate an actual mutation, and retain uncertainty until fresh observation."""
    checkpoint()
    active = current_control()
    if active:
        active.in_flight = True
    return action(*args, **kwargs)

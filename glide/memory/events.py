"""Explicit, ephemeral host events; observers never control invocation success."""

import copy
import math
import re
import time
from collections import deque
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from threading import RLock
from typing import Any
from uuid import uuid4

from ._callbacks import _sync
from .contracts import Scope


@dataclass(frozen=True)
class Invocation:
    scope: Scope
    plan_id: str
    call_id: str


_invocation: ContextVar[Invocation | None] = ContextVar("glide_invocation", default=None)


@contextmanager
def invocation(scope: Scope, plan_id: str, call_id: str):
    """Supply host context, restoring it when a synchronous or asynchronous call exits."""
    if not isinstance(scope, Scope):
        raise TypeError("invocation scope must be host-owned Scope")
    token = _invocation.set(Invocation(scope, plan_id, call_id))
    try:
        yield _invocation.get()
    finally:
        _invocation.reset(token)


def current_invocation() -> Invocation | None:
    return _invocation.get()


@dataclass(frozen=True)
class Event:
    id: str
    kind: str
    source: str
    data: dict[str, Any]
    timestamp: float
    scope: Scope | None = None
    correlation_id: str | None = None


class EventBus:
    """Thread-safe synchronous delivery with bounded observer fault metadata."""

    def __init__(self):
        self._lock = RLock()
        self._subscribers: dict[str, tuple[Callable[[Event], Any], frozenset[str] | None]] = {}
        self.errors: deque[dict[str, str]] = deque(maxlen=128)

    def subscribe(self, callback: Callable[[Event], Any], *, kinds: frozenset[str] | None = None):
        if not callable(callback) or (kinds is not None and not isinstance(kinds, frozenset)):
            raise TypeError("subscriber must be callable with an optional frozenset of kinds")
        identifier = uuid4().hex
        with self._lock:
            self._subscribers[identifier] = (callback, kinds)

        def unsubscribe():
            with self._lock:
                self._subscribers.pop(identifier, None)

        return unsubscribe

    def publish(self, kind: str, source: str, data: dict, *, scope=None, correlation_id=None) -> Event:
        context = current_invocation()
        if context is not None:
            scope = context.scope if scope is None else scope
            correlation_id = context.call_id if correlation_id is None else correlation_id
        event = Event(uuid4().hex, kind, source, copy.deepcopy(data), time.time(), scope, correlation_id)
        with self._lock:
            subscribers = tuple(self._subscribers.items())
        for identifier, (callback, kinds) in subscribers:
            if kinds is not None and kind not in kinds:
                continue
            try:
                _sync(callback(copy.deepcopy(event)))
            except BaseException as error:
                with self._lock:
                    self.errors.append({"event_id": event.id, "subscriber_id": identifier, "error_type": type(error).__name__})
        return event


_SAFE_ID = re.compile(r"[A-Za-z0-9_.:-]{1,256}\Z")
_TOOL_NAME = re.compile(r"[A-Za-z0-9_.-]{1,128}\Z")
_PROGRESS_TOKEN = re.compile(r"[0-9a-f]{32}\Z")
_NUMERIC = frozenset({"duration", "duration_ms", "bytes", "count", "progress", "total", "generation", "pages"})
_STATUS = frozenset({"success", "error", "cancelled", "protocol_failure", "execution_failure", "skipped", "stale"})


class SQLiteEventSink:
    """Persist a metadata allowlist, never raw arguments, results or notifications."""

    def __init__(self, store, default_scope: Scope | None = None):
        if default_scope is not None and not isinstance(default_scope, Scope):
            raise TypeError("default_scope must be host-owned Scope")
        self.store, self.default_scope = store, default_scope
        self.errors: deque[dict[str, str]] = deque(maxlen=128)

    def __call__(self, event: Event) -> None:
        try:
            scope = event.scope if event.scope is not None else self.default_scope
            if scope is None:
                return
            payload: dict[str, Any] = {"timestamp": event.timestamp}
            for key, value in (("source", event.source), ("correlation_id", event.correlation_id)):
                if isinstance(value, str) and _SAFE_ID.fullmatch(value):
                    payload[key] = value
            name = event.data.get("name")
            if isinstance(name, str) and _TOOL_NAME.fullmatch(name):
                payload["name"] = name
            operation = event.data.get("operation")
            if isinstance(operation, str) and operation in {"tools/list", "tools/call"}:
                payload["operation"] = operation
            progress_token = event.data.get("progressToken")
            if isinstance(progress_token, str) and _PROGRESS_TOKEN.fullmatch(progress_token):
                payload["progressToken"] = progress_token
            for key in _NUMERIC:
                value = event.data.get(key)
                if type(value) in (int, float) and 0 <= value <= 1e15 and math.isfinite(value):
                    payload[key] = value
            for key in ("success", "connected"):
                if type(event.data.get(key)) is bool:
                    payload[key] = event.data[key]
            if isinstance(event.data.get("status"), str) and event.data["status"] in _STATUS:
                payload["status"] = event.data["status"]
            self.store.event(scope, event.kind, payload)
        except BaseException as error:
            self.errors.append({"event_id": event.id, "error_type": type(error).__name__})

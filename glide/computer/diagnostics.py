"""The recording surface the computer core calls: `event`, `exception`, `record_content`, `Diagnostics`.

This is the small, in-memory part. A `Diagnostics` is activated for one task (`runner.run` does it) and
keeps a bounded list of events and the last exception, with credentials removed. It writes nothing to disk,
so a run folder holds only what the runner chooses to put there (D3). Whether the task may record
content (goal text, typed text, page text, raw URLs) is `record_content`, off unless the caller opts in;
without it, content fields are replaced by a marker and URLs are cut down to scheme and host.

The call surface (names and signatures) follows variant-6's `diagnostics.py`, so call sites in `calls`,
`browser.cdp` and `runner` port unchanged. The execution phase replaces this module's body with the
file-backed recorder (events.jsonl, failure report); keep the names below.
"""

from __future__ import annotations

import contextlib
import contextvars
import os
import re
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

REDACTED = "[REDACTED]"
CONTENT_OMITTED = "[content recording disabled]"
MAX_STRING = 500
MAX_ITEMS = 40
MAX_DEPTH = 6
MAX_RECENT_EVENTS = 200

_current: contextvars.ContextVar[Diagnostics | None] = contextvars.ContextVar("diagnostics", default=None)
_secret_key = re.compile(
    r"(?:^|[-_ ])(?:authorization|api[-_ ]?key|access[-_ ]?key|secrets?|token|passwords?|passwd|cookies?|credentials?)(?:$|[-_ ])",
    re.I,
)
_credential_assignment = re.compile(
    r"(?i)(\b(?:authorization|api[-_ ]?key|access[-_ ]?key|secret|token|password|passwd|cookie)\b[\"']?\s*[=:]\s*)"
    r"(?:\"[^\"]*\"|'[^']*'|(?:bearer\s+)?[^\s,;}]+)"
)
_bearer = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+")
_url = re.compile(r"\b(?:https?|wss?)://[^\s\"'<>]+", re.I)
_content_keys = {
    "goal",
    "plan",
    "history",
    "prompt",
    "response",
    "input",
    "output",
    "body",
    "text",
    "page",
    "page_text",
    "screenshot",
    "image",
    "html",
    "content",
    "evidence",
    "transcript",
    "query",
    "params",
    "arguments",
    "args",
    "value",
    "label",
    "instructions",
    "schema",
    "configuration",
}


def _secrets() -> list[str]:
    """Values of credential-named environment variables, read only to remove them."""
    return sorted({v for k, v in os.environ.items() if _secret_key.search(k) and len(v) >= 6}, key=len, reverse=True)


def _host_only(match: re.Match[str]) -> str:
    try:
        parts = urlsplit(match.group(0))
        return f"{parts.scheme}://{parts.netloc.rsplit('@', 1)[-1]}"
    except ValueError:
        return "[invalid URL]"


def scrub_text(value: str, *, urls: bool = True, limit: int = MAX_STRING) -> str:
    """Remove credentials (and, by default, everything after a URL's host) from one line of text."""
    for secret in _secrets():
        value = value.replace(secret, REDACTED)
    if urls:
        value = _url.sub(_host_only, value)
    value = _credential_assignment.sub(lambda m: m.group(1) + REDACTED, value)
    value = _bearer.sub("Bearer " + REDACTED, value)
    value = " ".join(value.split())
    return value if len(value) <= limit else value[:limit] + "..."


def redact(value: Any) -> Any:
    """Credentials removed from strings and structured fields; nothing truncated."""
    secrets = _secrets()

    def scrub(item: Any, depth: int = 0) -> Any:
        if depth > 30:
            return "[nested value omitted]"
        if isinstance(item, str):
            for secret in secrets:
                item = item.replace(secret, REDACTED)
            item = _credential_assignment.sub(lambda m: m.group(1) + REDACTED, item)
            return _bearer.sub("Bearer " + REDACTED, item)
        if isinstance(item, dict):
            return {k: REDACTED if _secret_key.search(str(k)) else scrub(v, depth + 1) for k, v in item.items()}
        if isinstance(item, list | tuple):
            return [scrub(child, depth + 1) for child in item]
        return item

    return scrub(value)


def _bounded(value: Any, *, content: bool, depth: int = 0) -> Any:
    if depth > MAX_DEPTH:
        return "[nested value omitted]"
    if isinstance(value, BaseException):
        # The type is always safe; the message only when content may be recorded.
        text = scrub_text(str(value), urls=not content) if content else type(value).__name__
        return {"type": type(value).__name__, "message": text}
    if isinstance(value, Path):
        return scrub_text(str(value), urls=not content)
    if isinstance(value, str):
        return scrub_text(value, urls=not content)
    if isinstance(value, dict):
        result = {}
        for index, (key, child) in enumerate(value.items()):
            key = str(key)[:64]
            if index >= MAX_ITEMS:
                result["_omitted_fields"] = len(value) - MAX_ITEMS
                break
            if _secret_key.search(key):
                result[key] = REDACTED
            elif not content and (key.lower() in _content_keys or key.lower().endswith(("_text", "_content", "_params"))):
                result[key] = CONTENT_OMITTED
            else:
                result[key] = _bounded(child, content=content, depth=depth + 1)
        return result
    if isinstance(value, list | tuple | deque):
        return [_bounded(child, content=content, depth=depth + 1) for child in list(value)[:MAX_ITEMS]]
    if value is None or isinstance(value, bool | int | float):
        return value
    return f"<{type(value).__name__}>"  # never repr() an SDK object: it may print credentials


class Diagnostics:
    """Bounded, in-memory events for one task. `root` is where a file-backed recorder would write."""

    def __init__(self, root: Path | None = None, *, record_content: bool = False, task_id: str = "", kind: str = "task"):
        self.root = Path(root) if root is not None else None
        self.record_content = record_content
        self.task_id, self.kind = task_id, kind
        self.events: deque[dict[str, Any]] = deque(maxlen=MAX_RECENT_EVENTS)
        self.error: dict[str, Any] | None = None
        self.error_stage = ""
        self._lock = threading.Lock()

    @contextlib.contextmanager
    def activate(self):
        token = _current.set(self)
        try:
            yield self
        finally:
            _current.reset(token)

    def event(self, name: str, **details: Any) -> None:
        item = {
            "time": time.time(),
            "event": scrub_text(name, limit=128),
            "details": _bounded(details, content=self.record_content),
        }
        with self._lock:
            self.events.append(item)

    def exception(self, error: BaseException, *, stage: str = "") -> None:
        with self._lock:
            self.error = _bounded(error, content=self.record_content)
            self.error_stage = scrub_text(stage, limit=64)
        self.event("exception", stage=stage, exception=error)

    def finish(self, summary: dict[str, Any], *, error: BaseException | None = None) -> Path | None:
        """Close the task's recording. Returns the path of a written report: this recorder writes none."""
        if error is not None:
            with self._lock:
                self.error = _bounded(error, content=self.record_content)
        return None


def event(name: str, **details: Any) -> None:
    if recorder := _current.get():
        recorder.event(name, **details)


def exception(error: BaseException, *, stage: str = "") -> None:
    if recorder := _current.get():
        recorder.exception(error, stage=stage)


def record_content() -> bool:
    recorder = _current.get()
    return bool(recorder and recorder.record_content)

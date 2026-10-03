"""The recording surface of the computer core: `event`, `exception`, `record_content`, `Diagnostics`.

A `Diagnostics` is activated for one task (`runner.run` does it) and keeps the last events and the last exception in
memory, with credentials removed. What it may keep is decided by one switch, `record_content` (D3), which is off
unless the caller opts in:

- Off: goal text, typed text, page text, answers and the like are replaced by a marker, URLs are cut down to scheme
  and host, and an exception is recorded by its type alone. Nothing is written to disk, so a run folder holds only
  what the runner puts there.
- On: content is kept (credentials still removed) and the recording is also written to the run folder as private
  files, so a failure can be handed to someone who was not there: `events.jsonl` (appended as it happens, rotated
  at a size), `diagnostic.json` (the summary, the exception chain and the last events, bounded) and, for a run that
  did not end well, `failure.md`. Directories are 0700 and files 0600, written by rename so a reader never sees a
  half-written report.

Recording never touches a browser, a model, a microphone or the desktop, and a write that fails is noted
(`write_error`) and never raised: a diagnostic must not mask the error it describes. The report says what was
observed and does not guess at a cause.

`write_private` is the one way the run folder's own files (run.json) are written, so they are 0600 too.
"""

from __future__ import annotations

import contextlib
import contextvars
import importlib.metadata
import json
import os
import platform
import re
import sys
import threading
import uuid
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

REDACTED = "[REDACTED]"
CONTENT_OMITTED = "[content recording disabled]"
MAX_STRING = 4096
MAX_ITEMS = 40
MAX_DEPTH = 7
MAX_RECENT_EVENTS = 200  # kept in memory; the report carries the last MAX_REPORT_EVENTS of them
MAX_REPORT_EVENTS = 40
MAX_EVENT_BYTES = 16_384
MAX_EVENTS_BYTES = 2_000_000
MAX_REPORT_BYTES = 196_608
MAX_EXCEPTIONS = 8
MAX_FRAMES = 30
CORE_SUMMARY_KEYS = ("outcome", "failure_stage", "failure_code", "error_type")

_current: contextvars.ContextVar[Diagnostics | None] = contextvars.ContextVar("diagnostics", default=None)
_secret_key = re.compile(
    r"(?:^|[-_ ])(?:authorization|api[-_ ]?key|access[-_ ]?key|secrets?|token|passwords?|passwd|cookies?|credentials?)(?:$|[-_ ])",
    re.I,
)
_credential_assignment = re.compile(
    r"(?i)(\b(?:authorization|api[-_ ]?key|access[-_ ]?key|secret|token|password|passwd|cookie)\b[\"']?\s*[=:]\s*)"
    r"(?:\"[^\"]*\"|'[^']*'|(?:bearer\s+)?[^\s,;}]+)"
)
_cookie = re.compile(r"(?i)(\bcookie[\"']?\s*[=:]\s*)[^\r\n]+")
_bearer = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+")
_url = re.compile(r"\b(?:https?|wss?)://[^\s\"'<>]+", re.I)
_content_keys = {
    "goal",
    "plan",
    "plans",
    "history",
    "prompt",
    "prompts",
    "response",
    "responses",
    "input",
    "output",
    "body",
    "text",
    "page",
    "page_text",
    "page_content",
    "screenshot",
    "image",
    "html",
    "markup",
    "content",
    "evidence",
    "sources",
    "transcript",
    "dictation",
    "query",
    "search_query",
    "params",
    "parameters",
    "arguments",
    "args",
    "action_params",
    "actions",
    "value",
    "label",
    "field_value",
    "instructions",
    "schema",
    "configuration",
}


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def _secrets() -> list[str]:
    """Values of credential-named environment variables, read only to remove them."""
    return sorted({v for k, v in os.environ.items() if _secret_key.search(k) and len(v) >= 6}, key=len, reverse=True)


def _host_only(match: re.Match[str]) -> str:
    try:
        parts = urlsplit(match.group(0))
        return f"{parts.scheme}://{parts.netloc.rsplit('@', 1)[-1]}"
    except ValueError:
        return "[invalid URL]"


def _redact_url(match: re.Match[str]) -> str:
    """A URL without its user and password, and without the value of any query parameter that names a credential."""
    try:
        parts = urlsplit(match.group(0))
        netloc = f"{REDACTED}@{parts.netloc.rsplit('@', 1)[-1]}" if "@" in parts.netloc else parts.netloc
        query = urlencode([(k, REDACTED if _secret_key.search(k) else v) for k, v in parse_qsl(parts.query)])
        return urlunsplit((parts.scheme, netloc, parts.path, query, parts.fragment))
    except ValueError:
        return REDACTED


def _hide_credentials(value: str, secrets: list[str]) -> str:
    for secret in secrets:
        value = value.replace(secret, REDACTED)
    value = _url.sub(_redact_url, value)
    value = _cookie.sub(lambda m: m.group(1) + REDACTED, value)
    value = _credential_assignment.sub(lambda m: m.group(1) + REDACTED, value)
    return _bearer.sub("Bearer " + REDACTED, value)


def scrub_text(value: str, *, urls: bool = True, limit: int = MAX_STRING) -> str:
    """Remove credentials (and, by default, everything after a URL's host) from one line of text."""
    value = _hide_credentials(value, _secrets())
    if urls:
        value = _url.sub(_host_only, value)
    value = " ".join(value.split())
    return value if len(value) <= limit else value[:limit] + "..."


def redact(value: Any) -> Any:
    """Credentials removed from strings and structured fields; nothing truncated, nothing else changed."""
    secrets = _secrets()

    def scrub(item: Any, depth: int = 0) -> Any:
        if depth > 30:
            return "[nested value omitted]"
        if isinstance(item, str):
            return _hide_credentials(item, secrets)
        if isinstance(item, dict):
            return {
                _hide_credentials(str(k), secrets): REDACTED if _secret_key.search(str(k)) else scrub(v, depth + 1)
                for k, v in item.items()
            }
        if isinstance(item, list | tuple):
            return [scrub(child, depth + 1) for child in item]
        return item

    return scrub(value)


def _bounded(value: Any, *, content: bool, depth: int = 0) -> Any:
    """`value` made safe to keep: credentials gone, size bounded, and without `content` no content fields or URL paths."""
    if depth > MAX_DEPTH:
        return "[nested value omitted]"
    if isinstance(value, BaseException):
        # The type is always safe; the message only when content may be recorded.
        return {"type": type(value).__name__, "message": _message(value) if content else type(value).__name__}
    if isinstance(value, Path):
        return scrub_text(str(value), urls=not content)
    if isinstance(value, str):
        return scrub_text(value, urls=not content)
    if isinstance(value, dict):
        result = {}
        for index, (key, child) in enumerate(value.items()):
            key = scrub_text(str(key), limit=64)
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
        items = list(value)
        result = [_bounded(child, content=content, depth=depth + 1) for child in items[:MAX_ITEMS]]
        if len(items) > MAX_ITEMS:
            result.append({"_omitted_items": len(items) - MAX_ITEMS})
        return result
    if value is None or isinstance(value, bool | int | float):
        return value
    return f"<{type(value).__name__}>"  # never repr() an SDK object: it may print credentials


def _message(error: BaseException) -> str:
    try:
        return scrub_text(str(error), urls=False)
    except Exception:
        return "message could not be formatted"


def _chain(error: BaseException) -> list[dict[str, Any]]:
    """The exception, its causes and its context, with code locations but no traceback locals or source lines."""
    chain, seen, current = [], set(), error
    while current is not None and len(chain) < MAX_EXCEPTIONS and id(current) not in seen:
        seen.add(id(current))
        frames, tb = [], current.__traceback__
        while tb is not None:
            frames.append({"file": tb.tb_frame.f_code.co_filename, "line": tb.tb_lineno, "function": tb.tb_frame.f_code.co_name})
            tb = tb.tb_next
        item: dict[str, Any] = {"type": type(current).__name__, "message": _message(current), "frames": frames[-MAX_FRAMES:]}
        if len(frames) > MAX_FRAMES:
            item["omitted_frames"] = len(frames) - MAX_FRAMES
        for name in ("errno", "status_code", "status", "kind", "code", "request_id"):
            code = getattr(current, name, None)
            if isinstance(code, int | str) and not isinstance(code, bool):
                item[name] = scrub_text(code, limit=128) if isinstance(code, str) else code
        chain.append(item)
        reason = getattr(current, "reason", None)  # urllib's URLError keeps the OSError it wrapped here
        following = current.__cause__ or (None if current.__suppress_context__ else current.__context__)
        current = following or (reason if isinstance(reason, BaseException) else None)
    return chain


def _runtime() -> dict[str, Any]:
    """What the run was running on and with: versions and `GLIDE_*` settings, never a credential or another variable."""
    packages = {}
    for name in ("glide", "typesafe-sdk", "httpx", "websocket-client"):
        with contextlib.suppress(importlib.metadata.PackageNotFoundError):
            packages[name] = importlib.metadata.version(name)
    return {
        "python": sys.version.split()[0],
        "platform": platform.system(),
        "platform_release": platform.release(),
        "entrypoint": Path(sys.argv[0]).name if sys.argv and sys.argv[0] else "",
        "settings": {k: v for k, v in os.environ.items() if k.startswith("GLIDE_") and not _secret_key.search(k)},
        "packages": packages,
    }


def _failed(summary: dict[str, Any]) -> bool:
    """Whether a run's summary describes a run that did not do what was asked (a stop by the user does not)."""
    outcome = str(summary.get("outcome", "")).lower().replace("_", " ").strip()
    if outcome in {"done", "success", "succeeded"}:
        return summary.get("goal_achieved") is False
    return bool(outcome) and not outcome.startswith("aborted") and outcome not in {"cancelled", "canceled", "dry run", "dryrun"}


def _json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


def write_private(path: Path, data: str | bytes) -> None:
    """Write `path` whole and 0600: a temporary file in the same folder, then a rename."""
    data = data.encode("utf-8") if isinstance(data, str) else data
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _append(path: Path, data: bytes) -> None:
    descriptor = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
    with os.fdopen(descriptor, "ab") as stream:
        stream.write(data)


def _markdown(report: dict[str, Any]) -> str:
    summary, error = report["summary"], report["exception"]
    lines = [
        "# Failure context",
        "",
        "This report records observed facts. A cause has not been inferred.",
        "",
        f"- Outcome: `{summary.get('outcome', 'unknown')}`",
        f"- Stage: `{summary.get('failure_stage') or report['error_stage'] or 'not recorded'}`",
        f"- Task: `{report['task_id'] or 'not assigned'}`",
        f"- Finished: `{report['finished_at']}`",
        "",
    ]
    for index, entry in enumerate(error or ()):
        lines += [f"## {'Exception' if index == 0 else 'Caused by or during'}: {entry['type']}", "", "```text"]
        lines += [entry.get("message", "").replace("```", "'''"), "```", ""]
        lines += [f"- {Path(f['file']).name}:{f['line']} in {f['function']}" for f in entry["frames"]] + [""]
    if not error:
        lines += ["No exception was attached; see the failure code and the events.", ""]
    for title, value in (("Run summary", summary), ("Runtime", report["runtime"]), ("Recent events", report["recent_events"])):
        lines += [f"## {title}", "", "```json", json.dumps(value, ensure_ascii=False, indent=2).replace("```", "'''"), "```", ""]
    return "\n".join(lines)


class Diagnostics:
    """Bounded events and the last exception of one task; also files in `root` when `record_content` is set."""

    def __init__(self, root: Path | None = None, *, record_content: bool = False, task_id: str = "", kind: str = "task"):
        self.requested_root = Path(root).absolute() if root is not None else None
        self.root = self.requested_root
        self.record_content = record_content
        self.task_id, self.kind = task_id, kind
        self.started_at = _now()
        self.events: deque[dict[str, Any]] = deque(maxlen=MAX_RECENT_EVENTS)
        self.error: dict[str, Any] | None = None  # {"type", "message"}: the message only with content recording
        self.error_stage = ""
        self.write_error = ""  # the type of the last failed write; recording is best effort
        self.report_path: Path | None = None
        self._chain: list[dict[str, Any]] = []
        self._claimed = False
        self._finished = False
        self._lock = threading.RLock()

    @property
    def persists(self) -> bool:
        return self.record_content and self.root is not None

    @contextlib.contextmanager
    def activate(self):
        token = _current.set(self)
        try:
            yield self
        finally:
            _current.reset(token)

    def _claim(self) -> None:
        """Make the folder and take it by creating its events file. A folder whose events file exists is in use (an
        earlier run, or another recorder made at the same time), so this recorder gets one of its own beneath it."""
        while not self._claimed:
            self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
            self.root.chmod(0o700)
            try:
                os.close(os.open(self.root / "events.jsonl", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))
            except FileExistsError:
                self.root = self.requested_root / "diagnostics" / uuid.uuid4().hex
            else:
                self._claimed = True

    def event(self, name: str, **details: Any) -> None:
        item = {
            "time": _now(),
            "event": scrub_text(name, limit=256),
            "details": redact(_bounded(details, content=self.record_content)),
        }
        with self._lock:
            self.events.append(item)
            if self.persists and not self._finished:
                self._write_event(item)

    def _write_event(self, item: dict[str, Any]) -> None:
        try:
            data = (json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
            if len(data) > MAX_EVENT_BYTES:
                item = {**item, "details": {"truncated": True, "fields": [k[:64] for k in list(item["details"])[:20]]}}
                data = (json.dumps(item, ensure_ascii=False) + "\n").encode("utf-8")
            self._claim()
            path = self.root / "events.jsonl"
            if path.stat().st_size + len(data) > MAX_EVENTS_BYTES:
                path.replace(self.root / "events.previous.jsonl")
            _append(path, data)
        except Exception as error:
            self.write_error = type(error).__name__

    def exception(self, error: BaseException, *, stage: str = "") -> None:
        with self._lock:
            self._note(error)
            self.error_stage = scrub_text(stage, limit=64)
        self.event("exception", stage=stage, exception=error)

    def _note(self, error: BaseException) -> None:
        self.error = _bounded(error, content=self.record_content)
        self._chain = _chain(error) if self.persists else []

    def finish(self, summary: dict[str, Any], *, error: BaseException | None = None) -> Path | None:
        """Close the recording. Returns the report written (`failure.md` for a failed run, else `diagnostic.json`),
        or None when nothing is written: content recording is off, or the write failed."""
        with self._lock:
            if self._finished:
                return self.report_path
            self._finished = True
            if error is not None:
                self._note(error)
            if not self.persists:
                return None
            try:
                self._claim()
                failed = _failed(summary)
                report = self._report(summary)
                write_private(self.root / "diagnostic.json", _json(report))
                self.report_path = self.root / "diagnostic.json"
                if failed:
                    text = _markdown(report).encode("utf-8")[:MAX_REPORT_BYTES].decode("utf-8", errors="ignore")
                    write_private(self.root / "failure.md", text)
                    self.report_path = self.root / "failure.md"
            except Exception as write_error:
                self.write_error = type(write_error).__name__
            return self.report_path

    def _report(self, summary: dict[str, Any]) -> dict[str, Any]:
        report = {
            "schema": 1,
            "started_at": self.started_at,
            "finished_at": _now(),
            "task_id": self.task_id,
            "kind": self.kind,
            "artifact_root": str(self.requested_root),
            "summary": redact(_bounded(summary, content=True)),
            "runtime": redact(_bounded(_runtime(), content=True)),
            "exception": redact(self._chain) or None,
            "error_stage": self.error_stage,
            "recent_events": list(self.events)[-MAX_REPORT_EVENTS:],
            "diagnostic_write_error": self.write_error,
        }
        while len(_json(report)) > MAX_REPORT_BYTES and report["recent_events"]:
            report["recent_events"].pop(0)
            report["events_omitted_for_size"] = report.get("events_omitted_for_size", 0) + 1
        if len(_json(report)) > MAX_REPORT_BYTES:  # a huge summary or exception: keep the facts that say what happened
            core = {k: scrub_text(str(report["summary"][k]), limit=256) for k in CORE_SUMMARY_KEYS if k in report["summary"]}
            kept = [{"type": e["type"], "frames": e["frames"][-1:]} for e in report["exception"] or ()]
            report.update(summary=core, runtime={}, exception=kept or None, content_omitted_for_size=True)
        return report


def event(name: str, **details: Any) -> None:
    if recorder := _current.get():
        recorder.event(name, **details)


def exception(error: BaseException, *, stage: str = "") -> None:
    if recorder := _current.get():
        recorder.exception(error, stage=stage)


def record_content() -> bool:
    recorder = _current.get()
    return bool(recorder and recorder.record_content)

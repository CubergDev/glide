"""The wire format of the app protocol (version 1): one JSON object per line, as app/PROTOCOL.md specifies.

This module is the only place that knows the envelope. It encodes the messages the core sends, parses the commands
the app sends, and makes text safe to put in either. It holds no socket and no state.

Text that did not come from the core's own code (what was said, what a page showed, a task's goal, a provider's error)
is data. `clean_text` is the one door it goes through before it reaches the app: control characters and invisible
formatting characters (a right-to-left override can make one command read as another) are removed, whitespace is
collapsed to single spaces, and the length is bounded.

A parse error never carries the line: a line can be what the person said.
"""

from __future__ import annotations

import json
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

VERSION = 1
MAX_LINE_BYTES = 1_048_576  # the longest line either side accepts, without its newline
MAX_ID_CHARS = 64
MAX_TYPE_CHARS = 64
MAX_TEXT_CHARS = 4000  # a typed request longer than this is refused, never cut: a cut request means something else
MAX_COMMAND_CHARS = 400  # what an approval shows: enough to read, short enough that nothing hides in it
MAX_SUMMARY_CHARS = 200
MAX_REASON_CHARS = 200
MAX_CHANGES = 32

ASSISTANT_STATES = ("idle", "listening", "thinking", "acting", "speaking", "asking", "awaiting_approval", "error")
TASK_PHASES = (
    "started",
    "step",
    "attempted",
    "verified",
    "unverified",
    "reconcile_required",
    "completed",
    "failed",
    "stopped",
)
APPROVAL_KINDS = ("screen", "input", "app", "url", "file", "network", "spend", "other")
SETTING_KEYS = (
    "voice.hands_free",
    "voice.headset",
    "voice.silence_ms",
    "voice.language",
    "privacy.record_content",
    "computer.act_enabled",
    "roles.pin",
)


class FrameError(Exception):
    """A line that cannot be used. `code` is short and fixed; the message never echoes the line."""

    def __init__(self, code: str, message: str = "", *, fatal: bool = False) -> None:
        super().__init__(message or code)
        self.code = code
        self.fatal = fatal


class UnsupportedVersion(FrameError):
    def __init__(self) -> None:
        super().__init__("unsupported_version", f"this core speaks protocol version {VERSION}", fatal=True)


# -- text ---------------------------------------------------------------------------------------------


def clean_text(value: object, limit: int) -> str:
    """`value` as text that is safe to show: no control or formatting characters, single spaces, at most `limit` characters.

    A text that is cut ends in an ellipsis, so the person can tell. Anything that is not a string is made one first.
    """
    text = value if isinstance(value, str) else str(value)
    kept: list[str] = []
    for ch in text:
        category = unicodedata.category(ch)
        if ch.isspace():
            kept.append(" ")
        elif category[0] in "CZ" and ch != " ":  # Cc, Cf (bidi overrides, zero width), Cs, Co, Cn, Zl, Zp
            continue
        else:
            kept.append(ch)
    out = " ".join("".join(kept).split())
    if len(out) > limit:
        out = out[: max(0, limit - 1)].rstrip() + "\u2026"
    return out


# -- core to app --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Outgoing:
    """A message the core sends: its type, payload and the ids it answers or carries."""

    type: str
    data: Mapping[str, Any] | None = None
    reply_to: str | None = None
    id: str | None = None
    droppable: bool = False  # a slow client may miss it (levels, partial transcripts); never state, tasks, approvals


def encode(message: Outgoing) -> bytes:
    """The line for `message`, newline included. Always ASCII (non-ASCII is escaped), so no byte of it is a control character."""
    frame: dict[str, Any] = {"v": VERSION, "type": message.type}
    if message.id is not None:
        frame["id"] = message.id
    if message.reply_to is not None:
        frame["reply_to"] = message.reply_to
    if message.data is not None:
        frame["data"] = message.data
    line = json.dumps(frame, separators=(",", ":"), sort_keys=True, ensure_ascii=True).encode("ascii")
    if len(line) > MAX_LINE_BYTES:
        raise FrameError("line_too_long", "a message would exceed the line limit")
    return line + b"\n"


def hello(
    *, core_version: str, session_id: str, capabilities: list[str], recording_content: bool, reply_to: str | None = None
) -> Outgoing:
    return Outgoing(
        "hello",
        {
            "protocol": VERSION,
            "core_version": core_version,
            "session_id": session_id,
            "capabilities": list(capabilities),
            "recording_content": bool(recording_content),
        },
        reply_to=reply_to,
    )


def state(assistant: str, *, hands_free: bool, muted: bool, detail: str | None = None) -> Outgoing:
    data: dict[str, Any] = {"assistant": assistant, "hands_free": bool(hands_free), "muted": bool(muted)}
    if detail:
        data["detail"] = clean_text(detail, 60)
    return Outgoing("state", data)


def transcript(utterance_id: str, role: str, text: str, *, partial: bool, record_content: bool) -> Outgoing:
    """What was said. Content-free unless `record_content`: then there is no `text`, `redacted` is true and `chars` says how long
    it was (so the app can show that something was said, and how much). `chars` is a count, never a prefix."""
    data: dict[str, Any] = {"utterance_id": utterance_id, "role": role, "partial": bool(partial)}
    if record_content:
        data["text"] = clean_text(text, MAX_TEXT_CHARS)
        data["redacted"] = False
    else:
        data["redacted"] = True
        data["chars"] = min(len(text), 100_000)
    return Outgoing("transcript", data, droppable=partial)


def speech(phase: str, utterance_id: str | None = None) -> Outgoing:
    data: dict[str, Any] = {"phase": phase}
    if utterance_id is not None:
        data["utterance_id"] = utterance_id
    return Outgoing("speech", data)


def task(
    task_id: str, phase: str, *, step: int | None = None, summary: str | None = None, verified: bool | None = None
) -> Outgoing:
    data: dict[str, Any] = {"task_id": clean_text(task_id, MAX_ID_CHARS), "phase": phase}
    if step is not None:
        data["step"] = int(step)
    if summary:
        data["summary"] = clean_text(summary, MAX_SUMMARY_CHARS)
    if verified is not None:
        data["verified"] = bool(verified)
    return Outgoing("task", data)


def switch(*, role: str, from_slot: str, to_slot: str | None, kind: str, reason: str) -> Outgoing:
    return Outgoing(
        "switch",
        {
            "role": clean_text(role, 60),
            "from_slot": clean_text(from_slot, 120),
            "to_slot": None if to_slot is None else clean_text(to_slot, 120),
            "kind": clean_text(kind, 40),
            "reason": clean_text(reason, MAX_REASON_CHARS),
        },
    )


def approval_request(
    approval_id: str, kind: str, command: str, *, task_id: str | None = None, expires_in_s: float | None = None
) -> Outgoing:
    data: dict[str, Any] = {
        "approval_id": approval_id,
        "kind": kind if kind in APPROVAL_KINDS else "other",
        "command": clean_text(command, MAX_COMMAND_CHARS),
    }
    if task_id is not None:
        data["task_id"] = clean_text(task_id, MAX_ID_CHARS)
    if expires_in_s is not None:
        data["expires_in_s"] = float(expires_in_s)
    return Outgoing("approval_request", data)


def approval_closed(approval_id: str, outcome: str) -> Outgoing:
    """An addition inside version 1 (a new message type, which a receiver ignores if it does not know it): the core is no longer
    waiting for `approval_id`, and why. `outcome` is approved, denied, expired, cancelled or disconnected."""
    return Outgoing("approval_closed", {"approval_id": approval_id, "outcome": outcome})


def settings(revision: int, payload: Mapping[str, Any], *, reply_to: str | None = None) -> Outgoing:
    return Outgoing("settings", {"revision": int(revision), "settings": dict(payload)}, reply_to=reply_to)


def settings_result(ok: bool, revision: int, errors: list[tuple[str, str]], *, reply_to: str | None = None) -> Outgoing:
    return Outgoing(
        "settings_result",
        {
            "ok": bool(ok),
            "revision": int(revision),
            "errors": [{"key": clean_text(k, 80), "message": clean_text(m, MAX_REASON_CHARS)} for k, m in errors],
        },
        reply_to=reply_to,
    )


def error(code: str, message: str, *, fatal: bool = False, reply_to: str | None = None) -> Outgoing:
    return Outgoing(
        "error", {"code": code, "message": clean_text(message, MAX_REASON_CHARS), "fatal": bool(fatal)}, reply_to=reply_to
    )


def ping() -> Outgoing:
    return Outgoing("ping")


# -- app to core --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Command:
    """A parsed command of the app. `data` holds only the fields this core reads, checked for type and size."""

    type: str
    id: str | None = None
    data: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Unknown:
    """A well-formed line whose type this core does not know (a newer app). Ignored, and logged without its content."""

    type: str
    id: str | None = None


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _text(data: Mapping[str, Any], key: str, limit: int) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value or len(value) > limit:
        raise FrameError("malformed", f"{key} is missing or not usable")
    return value


def _optional_text(data: Mapping[str, Any], key: str, limit: int) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or len(value) > limit:
        raise FrameError("malformed", f"{key} is not usable")
    return value


def _hello(data: Mapping[str, Any]) -> dict[str, Any]:
    if not _is_int(data.get("protocol")):
        raise FrameError("malformed", "protocol is missing or not a number")
    return {
        "protocol": data["protocol"],
        "client": clean_text(data.get("client", ""), 60),
        "client_version": clean_text(data.get("client_version", ""), 40),
    }


def _text_input(data: Mapping[str, Any]) -> dict[str, Any]:
    value = data.get("text")
    if not isinstance(value, str):
        raise FrameError("malformed", "text is missing")
    if len(value) > MAX_TEXT_CHARS:
        raise FrameError("text_too_long", f"a request is at most {MAX_TEXT_CHARS} characters")
    text = " ".join(value.split())
    if not text:
        raise FrameError("malformed", "text is empty")
    return {"text": text}


def _approval_response(data: Mapping[str, Any]) -> dict[str, Any]:
    decision = data.get("decision")
    if decision not in ("approve", "deny"):
        raise FrameError("malformed", "decision must be approve or deny")
    return {"approval_id": _text(data, "approval_id", MAX_ID_CHARS), "decision": decision}


def _voice_control(data: Mapping[str, Any]) -> dict[str, Any]:
    action = data.get("action")
    if action not in ("mute", "unmute"):
        raise FrameError("malformed", "action must be mute or unmute")
    return {"action": action}


def _stop(data: Mapping[str, Any]) -> dict[str, Any]:
    return {"task_id": _optional_text(data, "task_id", MAX_ID_CHARS)}


def _settings_set(data: Mapping[str, Any]) -> dict[str, Any]:
    base = data.get("base_revision")
    changes = data.get("changes")
    if not _is_int(base) or not isinstance(changes, list) or len(changes) > MAX_CHANGES:
        raise FrameError("malformed", "base_revision or changes is not usable")
    checked: list[dict[str, Any]] = []
    for change in changes:
        if not isinstance(change, dict) or not isinstance(change.get("key"), str) or "value" not in change:
            raise FrameError("malformed", "a change needs a key and a value")
        checked.append({"key": change["key"][:80], "value": change["value"]})
    return {"base_revision": base, "changes": checked}


def _empty(data: Mapping[str, Any]) -> dict[str, Any]:
    return {}


_COMMANDS = {
    "hello": _hello,
    "text_input": _text_input,
    "interrupt": _empty,
    "stop": _stop,
    "approval_response": _approval_response,
    "voice_control": _voice_control,
    "settings_get": _empty,
    "settings_set": _settings_set,
    "pong": _empty,
}


def parse(line: bytes) -> Command | Unknown:
    """One line of the app, without its newline. Raises `FrameError` (never carrying the line) when it cannot be used,
    `UnsupportedVersion` when `v` is not ours."""
    if len(line) > MAX_LINE_BYTES:
        raise FrameError("line_too_long", "the line exceeds the limit")
    try:
        frame = json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError):
        raise FrameError("malformed", "the line is not JSON") from None
    if not isinstance(frame, dict):
        raise FrameError("malformed", "the line is not an object")
    version = frame.get("v")
    if not _is_int(version):
        raise FrameError("malformed", "v is missing")
    if version != VERSION:
        raise UnsupportedVersion
    kind = frame.get("type")
    if not isinstance(kind, str) or not kind or len(kind) > MAX_TYPE_CHARS:
        raise FrameError("malformed", "type is missing or not usable")
    ident = frame.get("id")
    if ident is not None and (not isinstance(ident, str) or len(ident) > MAX_ID_CHARS):
        raise FrameError("malformed", "id is not usable")
    check = _COMMANDS.get(kind)
    if check is None:
        return Unknown(clean_text(kind, 40), ident)
    data = frame.get("data", {})
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise FrameError("malformed", "data is not an object")
    return Command(kind, ident, check(data))

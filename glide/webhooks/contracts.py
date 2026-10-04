"""Versioned messages and the small readers every other module shares. Transport data cannot select agent methods.

Importing this module needs only pydantic, so the policy tests run without the `webhooks` extra."""

from __future__ import annotations

import json
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SLUG = r"^[a-z0-9][a-z0-9_-]{0,63}$"
ENV_NAME = r"[A-Z][A-Z0-9_]{0,99}"  # the name of an environment variable that holds a secret
MAX_CONTEXT_BYTES = 16384


class AuthError(ValueError):
    """A callback or token could not prove who sent it. Answered 401; the message is never shown to the sender."""


class TranslationError(ValueError):
    """A verified callback does not fit a call. Answered 400; the message quotes nothing from the payload."""


def call_id(source: str, event_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "glide-webhook:" + source + ":" + event_id))


def clip_bytes(text: str, limit: int) -> str:
    """`text` cut to at most `limit` UTF-8 bytes, never inside a character."""
    return text.encode("utf-8")[:limit].decode("utf-8", errors="ignore")


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON keys.")
        result[key] = value
    return result


def _no_constants(name):
    raise ValueError("NaN and Infinity are not JSON.")


def strict_json(raw: bytes) -> dict:
    """One JSON object: no duplicate keys, no NaN or Infinity. Anything else is a `TranslationError` quoting nothing."""
    try:
        value = json.loads(raw, object_pairs_hook=_unique_keys, parse_constant=_no_constants)
    except (ValueError, UnicodeError, RecursionError):
        raise TranslationError("Invalid JSON object.") from None
    if not isinstance(value, dict):
        raise TranslationError("Invalid JSON object.")
    return value


def _context_bytes(context: dict) -> int:
    return len(json.dumps(context, ensure_ascii=False, allow_nan=False).encode("utf-8"))


class Message(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class AgentCall(Message):
    schema_version: Literal["1.0"] = "1.0"
    id: str
    task_id: str
    agent_id: str = Field(pattern=SLUG)
    operation: Literal[
        "github.issue.triage",
        "github.pull_request.review",
        "github.comment.summarize",
        "gmail.mail.changed",
        "outlook.mail.changed",
        "agent.task.requested",
    ]
    source: str = Field(pattern=SLUG)
    event_id: str = Field(min_length=1, max_length=200)
    goal: str = Field(min_length=1, max_length=4096)
    context: dict[str, Any] = Field(default_factory=dict)
    allow_actions: bool = False

    @field_validator("id", "task_id")
    @classmethod
    def uuid_id(cls, value):
        if str(uuid.UUID(value)) != value:
            raise ValueError("A canonical UUID is required.")
        return value

    @field_validator("context")
    @classmethod
    def bounded_context(cls, value):
        try:
            size = _context_bytes(value)
        except (ValueError, TypeError, RecursionError):
            raise ValueError("Invalid context.") from None
        if size > MAX_CONTEXT_BYTES:
            raise ValueError("Context exceeds 16 KiB.")
        return value

    @model_validator(mode="after")
    def action_policy(self):
        if self.operation != "agent.task.requested" and self.allow_actions:
            raise ValueError("Provider notifications cannot enable computer actions.")
        return self


def build_call(
    *, source_id: str, event_id: str, agent_id: str, operation: str, goal: str, context: dict, allow_actions: bool = False
) -> AgentCall:
    """The call for one verified event, identified by its source and event. What the contract refuses is a
    `TranslationError` that quotes nothing."""
    if _context_bytes(context) > MAX_CONTEXT_BYTES:
        raise TranslationError("Webhook context is too large.")
    identity = call_id(source_id, event_id)
    try:
        return AgentCall(
            id=identity,
            task_id=identity,
            agent_id=agent_id,
            operation=operation,
            source=source_id,
            event_id=event_id,
            goal=goal,
            context=context,
            allow_actions=allow_actions,
        )
    except ValueError:
        raise TranslationError("Invalid webhook call.") from None


class LeaseMessage(Message):
    lease_token: str = Field(min_length=32, max_length=200)


class Completion(LeaseMessage):
    """A worker's result. `summary` may carry content derived from the task (a report, a page title), so the
    queue stores it only under `record_content`. `note` is a short worker-authored status phrase that never
    carries task content (for example "Desktop run not approved"); the queue always keeps it."""

    outcome: Literal["completed", "blocked", "cancelled", "failed", "uncertain"]
    summary: str = Field(default="", max_length=4096)
    note: str = Field(default="", max_length=200)


class AgentEvent(LeaseMessage):
    kind: Literal[
        "accepted",
        "question",
        "model_completed",
        "classifier_completed",
        "action_checked",
        "completed",
        "blocked",
        "cancelled",
        "status",
        "progress",
    ]
    text: str = Field(default="", max_length=4096)
    question_id: str = Field(default="", max_length=100)
    model: str = Field(default="", max_length=200)
    role: str = Field(default="", max_length=100)
    outcome: str = Field(default="", max_length=200)
    elapsed_s: float = Field(default=0, ge=0, le=86400, allow_inf_nan=False)


class ResolveMessage(Message):
    """An operator's reconciliation of an `uncertain` task. It never reruns or requeues anything."""

    outcome: Literal["completed", "failed", "cancelled"]
    summary: str = Field(default="", max_length=4096)

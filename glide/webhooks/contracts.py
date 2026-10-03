"""Versioned messages. Transport data cannot select arbitrary agent methods.

Importing this module needs only pydantic, so the policy tests run without the `webhooks` extra."""

from __future__ import annotations

import json
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SLUG = r"^[a-z0-9][a-z0-9_-]{0,63}$"


class AuthError(ValueError):
    pass


class TranslationError(ValueError):
    pass


def call_id(source: str, event_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "glide-webhook:" + source + ":" + event_id))


def strict_json(raw: bytes) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise TranslationError("Duplicate JSON keys.")
            result[key] = value
        return result

    try:
        value = json.loads(raw, object_pairs_hook=unique, parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
    except (ValueError, UnicodeError, RecursionError):
        raise TranslationError("Invalid JSON object.") from None
    if not isinstance(value, dict):
        raise TranslationError("Invalid JSON object.")
    return value


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
            encoded = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
        except (ValueError, TypeError, RecursionError):
            raise ValueError("Invalid context.") from None
        if len(encoded) > 16384:
            raise ValueError("Context exceeds 16 KiB.")
        return value

    @model_validator(mode="after")
    def action_policy(self):
        if self.operation != "agent.task.requested" and self.allow_actions:
            raise ValueError("Provider notifications cannot enable computer actions.")
        return self


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

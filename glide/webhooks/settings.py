"""Explicit provider configuration; credentials are supplied by environment variables or a `SecretSource`.

Everything is off until switched on (D5): the file needs `"enabled": true` and each source needs its own
`"enabled": true`. A source that is listed but not enabled answers 404."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator

from .contracts import SLUG, Message, strict_json
from .secret_sources import EnvSecrets, SecretSource


class VerificationKey(Message):
    kid: str = Field(min_length=1, max_length=100)
    algorithm: Literal["RS256", "EdDSA"]
    public_key: str = Field(min_length=32, max_length=8192)


class AgentAuth(Message):
    issuer: str = Field(min_length=1, max_length=500)
    audience: str = Field(min_length=1, max_length=500)
    keys: tuple[VerificationKey, ...] = Field(min_length=1, max_length=8)
    max_token_seconds: int = Field(default=3600, ge=60, le=3600)

    @model_validator(mode="after")
    def unique_keys(self):
        if len({key.kid for key in self.keys}) != len(self.keys):
            raise ValueError("Duplicate verification key identifiers.")
        return self


class Source(Message):
    id: str = Field(pattern=SLUG)
    enabled: bool = False
    provider: Literal["github", "standard", "gmail", "outlook", "mcp"]
    agent_id: str = Field(pattern=SLUG)
    key_envs: tuple[str, ...] = Field(default=(), max_length=4)
    repositories: tuple[str, ...] = ()
    allowed_senders: tuple[str, ...] = ()
    allow_actions: bool = False
    mailbox: str = ""
    subscription: str = ""
    audience: str = ""
    service_account: str = ""
    mailbox_id: str = ""
    tenant_id: str = ""
    subscription_id: str = ""
    subscription_expires_at: str = ""
    gmail_max_age_seconds: int = Field(default=86400, ge=300, le=604800)

    @field_validator("key_envs")
    @classmethod
    def secret_references(cls, values):
        if any(not re.fullmatch(r"[A-Z][A-Z0-9_]{0,99}", value) for value in values):
            raise ValueError("Invalid secret environment name.")
        return values

    @model_validator(mode="after")
    def provider_config(self):
        if self.provider in {"github", "standard", "outlook"} and not self.key_envs:
            raise ValueError("A signing/client-state secret must be configured.")
        if self.provider == "github" and not self.repositories:
            raise ValueError("GitHub requires a repository allowlist.")
        if self.provider == "gmail" and not all((self.mailbox, self.subscription, self.audience, self.service_account)):
            raise ValueError("Gmail requires mailbox, subscription, audience and push service-account bindings.")
        if self.provider == "outlook":
            if not all((self.mailbox_id, self.tenant_id, self.subscription_id, self.subscription_expires_at)):
                raise ValueError("Outlook requires mailbox, tenant, subscription and expiration bindings.")
            expiry = datetime.fromisoformat(self.subscription_expires_at.replace("Z", "+00:00"))
            if expiry.tzinfo is None:
                raise ValueError("Subscription expiration must include its timezone.")
        if self.provider != "standard" and self.allow_actions:
            raise ValueError("Only explicit standard task requests can allow computer actions.")
        return self

    def secrets(self, source: SecretSource | None = None) -> tuple[str, ...]:
        source = source or EnvSecrets()
        values = tuple(source.get(name) for name in self.key_envs)
        if not values or any(len(value.encode("utf-8")) < 32 for value in values):
            raise ValueError("Webhook credentials are missing or too short.")
        return values


class ServerSettings(Message):
    enabled: bool = False
    # D3: the queue keeps no issue/comment/mail excerpts, task goals after completion, worker summaries or event
    # text unless this is switched on.
    record_content: bool = False
    database: str = ".glide-webhooks/queue.sqlite3"
    sources: tuple[Source, ...] = Field(min_length=1, max_length=64)
    agent_auth: AgentAuth
    body_limit: int = Field(default=262144, ge=1024, le=1048576)
    timestamp_tolerance: int = Field(default=300, ge=30, le=300)
    source_requests_per_minute: int = Field(default=120, ge=1, le=10000)
    requests_per_minute: int = Field(default=1800, ge=1, le=100000)
    max_pending: int = Field(default=1000, ge=1, le=100000)
    max_events: int = Field(default=100000, ge=1, le=1000000)
    lease_seconds: int = Field(default=60, ge=15, le=300)
    allowed_hosts: tuple[str, ...] = Field(default=("localhost", "127.0.0.1"), min_length=1, max_length=64)

    @model_validator(mode="after")
    def unique_sources(self):
        if len({source.id for source in self.sources}) != len(self.sources):
            raise ValueError("Duplicate webhook source identifiers.")
        if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,252}", host) for host in self.allowed_hosts):
            raise ValueError("Configure explicit callback hosts.")
        return self


def load_settings(path: Path) -> ServerSettings:
    try:
        # JSON validation preserves strict tuple/type semantics at this boundary.
        raw = path.read_bytes()
        strict_json(raw)
        return ServerSettings.model_validate_json(raw)
    except (OSError, ValueError):
        raise ValueError("Invalid webhook configuration. See glide/webhooks/README.md.") from None

"""Shared builders for the webhook tests. No sockets, no real keys: every value is synthetic."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime, timedelta

from glide.webhooks.contracts import AgentCall
from glide.webhooks.settings import ServerSettings

SECRET = "fixture-webhook-key-" + "a" * 40
PLACEHOLDER_PEM = "-----BEGIN PUBLIC KEY-----\n" + "A" * 64 + "\n-----END PUBLIC KEY-----\n"


def settings_data(
    *, enabled=True, sources_enabled=True, record_content=False, database="queue.sqlite3", pem=PLACEHOLDER_PEM, **extra
):
    """A complete, valid configuration dict with synthetic bindings (no real account, host or key)."""
    expires = (datetime.now(UTC) + timedelta(days=1)).isoformat()
    data = {
        "enabled": enabled,
        "record_content": record_content,
        "database": database,
        "sources": [
            {
                "id": "gh",
                "enabled": sources_enabled,
                "provider": "github",
                "agent_id": "team",
                "key_envs": ["GITHUB_WEBHOOK_SECRET"],
                "repositories": ["team/repo"],
            },
            {
                "id": "gmail",
                "enabled": sources_enabled,
                "provider": "gmail",
                "agent_id": "team",
                "mailbox": "box@example.invalid",
                "subscription": "projects/p1/subscriptions/s1",
                "audience": "urn:fixture:gmail-audience",
                "service_account": "push@p1.iam.gserviceaccount.invalid",
            },
            {
                "id": "outlook",
                "enabled": sources_enabled,
                "provider": "outlook",
                "agent_id": "team",
                "key_envs": ["GRAPH_CLIENT_STATE"],
                "mailbox_id": "11111111-1111-4111-8111-111111111111",
                "tenant_id": "22222222-2222-4222-8222-222222222222",
                "subscription_id": "33333333-3333-4333-8333-333333333333",
                "subscription_expires_at": expires,
            },
            {
                "id": "standard",
                "enabled": sources_enabled,
                "provider": "standard",
                "agent_id": "team",
                "key_envs": ["STANDARD_WEBHOOK_KEY"],
                "allow_actions": True,
            },
        ],
        "agent_auth": {
            "issuer": "urn:fixture:issuer",
            "audience": "glide-agents",
            "keys": [{"kid": "one", "algorithm": "EdDSA", "public_key": pem}],
        },
    }
    data.update(extra)
    return data


def make_settings(**kwargs) -> ServerSettings:
    return ServerSettings.model_validate_json(json.dumps(settings_data(**kwargs)))


def make_call(
    operation="github.issue.triage",
    *,
    allow_actions=False,
    context=None,
    goal="Summarize the supplied issue metadata.",
    agent="laptop",
    key=None,
):
    identity = str(uuid.uuid5(uuid.NAMESPACE_URL, "synthetic-webhook:" + (key or operation)))
    return AgentCall(
        id=identity,
        task_id=identity,
        agent_id=agent,
        operation=operation,
        source="fixture",
        event_id="fixture-event",
        goal=goal,
        context={"number": 7} if context is None else context,
        allow_actions=allow_actions,
    )


class Hdrs:
    """Just enough of starlette's `Headers` (case-insensitive, `getlist`, `raw`, dict() of the first values)
    for the auth and translation tests, so they run without the `webhooks` extra."""

    def __init__(self, data=None, *, raw=None):
        pairs = list(raw) if raw is not None else [(k.encode(), v.encode()) for k, v in (data or {}).items()]
        self.raw = [(k.lower(), v) for k, v in pairs]

    def getlist(self, name):
        key = name.lower().encode()
        return [v.decode() for k, v in self.raw if k == key]

    def keys(self):
        return list(dict.fromkeys(k.decode() for k, _ in self.raw))

    def __getitem__(self, name):
        values = self.getlist(name)
        if not values:
            raise KeyError(name)
        return values[0]

    def get(self, name, default=None):
        values = self.getlist(name)
        return values[0] if values else default

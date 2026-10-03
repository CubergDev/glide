"""Authenticated callback ingress and the agent queue API. No desktop imports.

The service refuses to build unless the configuration says `"enabled": true`, and a source answers 404 unless it is
enabled too (D5). Callback bytes are verified exactly as received, before anything is parsed.
"""

from __future__ import annotations

import asyncio
import hashlib
import sqlite3
import threading
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import Field, ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.responses import JSONResponse, PlainTextResponse

from .auth import AgentVerifier, GoogleVerifier, bearer, delivery_id, header, verify_github, verify_standard, webhook_keys
from .contracts import AgentEvent, AuthError, Completion, LeaseMessage, Message, TranslationError, strict_json
from .mail import translate_gmail, translate_outlook, validate_outlook_token
from .settings import ServerSettings, Source
from .store import DeliveryConflict, LeaseConflict, QueueFull, QueueStore
from .translation import translate_github, translate_standard

JSON_TYPES = {"application/json", "application/cloudevents+json"}
WORKER_SCOPES = {"claim": "agent:claim", "heartbeat": "agent:report", "complete": "agent:report", "events": "agent:report"}


class ResolveMessage(Message):
    outcome: Literal["completed", "failed", "cancelled"]
    summary: str = Field(default="", max_length=4096)


class Limiter:
    """A sliding one-minute window in a fixed, configured bucket: attacker-chosen peers or paths never allocate state."""

    def __init__(self, limit: int):
        self.limit, self.recent, self.lock = limit, deque(), threading.Lock()

    def admit(self) -> None:
        now = time.monotonic()
        with self.lock:
            while self.recent and self.recent[0] <= now - 60:
                self.recent.popleft()
            if len(self.recent) >= self.limit:
                raise HTTPException(429, "Request rate exceeded.", headers={"Retry-After": "60"})
            self.recent.append(now)


def _expired(source: Source) -> bool:
    return datetime.now(UTC) >= datetime.fromisoformat(source.subscription_expires_at.replace("Z", "+00:00"))


async def read_body(request: Request, limit: int) -> bytes:
    """The raw body, bounded in size and time, before anything is decoded."""
    if request.headers.get("content-encoding", "identity") != "identity":
        raise HTTPException(415, "Compressed request bodies are not supported.")
    lengths = request.headers.getlist("content-length")
    if len(lengths) > 1 or (lengths and not (lengths[0].isascii() and lengths[0].isdigit())):
        raise TranslationError("Invalid body length.")
    if lengths and int(lengths[0]) > limit:
        raise HTTPException(413, "Request body exceeds the limit.")
    parts, total = [], 0
    async with asyncio.timeout(5):
        async for chunk in request.stream():
            total += len(chunk)
            if total > limit:
                raise HTTPException(413, "Request body exceeds the limit.")
            parts.append(chunk)
    return b"".join(parts)


def create_app(settings: ServerSettings, *, store=None, google_verifier=None, secrets=None, check_only=False) -> FastAPI:
    """Build the service. `check_only` validates everything (keys, secrets, bindings) and returns without a store."""
    if not settings.enabled:
        raise ValueError('Webhooks are disabled. Set "enabled": true in the configuration to run the service.')
    sources = {s.id: s for s in settings.sources if s.enabled}
    keys = {}
    for source in sources.values():
        if source.provider in {"github", "outlook", "standard"}:
            secret_values = source.secrets(secrets)
            keys[source.id] = webhook_keys(secret_values) if source.provider == "standard" else secret_values
    verifier = AgentVerifier(settings.agent_auth)
    google = google_verifier
    if google is None and any(s.provider == "gmail" for s in sources.values()):
        google = GoogleVerifier()
    global_limit = Limiter(settings.requests_per_minute)
    source_limits = {sid: Limiter(settings.source_requests_per_minute) for sid in sources}

    @asynccontextmanager
    async def lifespan(app):
        if check_only:
            yield
            return
        path = Path(settings.database)
        owned = store is None
        app.state.queue = store or QueueStore(
            path, settings.max_pending, settings.max_events, record_content=settings.record_content
        )
        try:
            yield
        finally:
            if owned:
                app.state.queue.close()

    app = FastAPI(
        title="Glide callbacks",
        version="1.0",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        dependencies=[Depends(global_limit.admit)],
    )
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.allowed_hosts)
    if check_only:
        return app

    @app.middleware("http")
    async def secure_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers.update(
            {"cache-control": "no-store", "x-content-type-options": "nosniff", "x-request-id": str(uuid.uuid4())}
        )
        return response

    def answer(detail: str, status: int, **headers):
        return lambda request, error: JSONResponse({"detail": detail}, status, headers=headers or None)

    app.add_exception_handler(
        AuthError, answer("Invalid callback or bearer authentication.", 401, **{"WWW-Authenticate": "Bearer"})
    )
    app.add_exception_handler(PermissionError, answer("Insufficient agent scope.", 403))
    app.add_exception_handler(TranslationError, answer("Invalid callback payload.", 400))
    app.add_exception_handler(TimeoutError, answer("Request body timed out.", 408))
    app.add_exception_handler(DeliveryConflict, answer("Delivery identity conflicts with stored content.", 409))
    app.add_exception_handler(LeaseConflict, answer("Message lease is unavailable.", 409))
    app.add_exception_handler(QueueFull, answer("Callback queue is full.", 503, **{"Retry-After": "5"}))
    app.add_exception_handler(sqlite3.Error, answer("Callback storage is unavailable.", 503, **{"Retry-After": "5"}))

    @app.get("/healthz")
    def health():
        return {"status": "ok"}

    # -- provider callbacks -------------------------------------------------------------------------------------

    @app.post("/webhooks/{provider}/{source_id}", status_code=202)
    async def callback(request: Request, provider: str, source_id: str):
        source = sources.get(source_id)
        if source is None or source.provider != provider:
            raise HTTPException(404, "Callback not found.")
        source_limits[source_id].admit()
        if source.provider == "outlook" and "validationToken" in request.query_params:
            tokens = request.query_params.getlist("validationToken")
            if len(tokens) != 1:
                raise TranslationError("Invalid validation token.")
            if _expired(source):
                raise AuthError("Subscription expired.")
            return PlainTextResponse(validate_outlook_token(tokens[0]))
        if source.provider == "gmail":
            await run_in_threadpool(
                google.verify, bearer(request.headers), audience=source.audience, service_account=source.service_account
            )
        raw = await read_body(request, settings.body_limit)
        if request.headers.get("content-type", "").split(";")[0].casefold() not in JSON_TYPES:
            raise HTTPException(415, "Callbacks require JSON.")
        try:
            deliveries = translate(source, request, raw)
        except (ValueError, KeyError, TypeError, RecursionError) as error:
            if isinstance(error, AuthError | TranslationError):
                raise
            raise TranslationError("Invalid callback payload.") from None
        return {"deliveries": await run_in_threadpool(request.app.state.queue.enqueue_many, deliveries)}

    def translate(source: Source, request: Request, raw: bytes) -> list[dict]:
        """Verify, then translate. Returns delivery records; [] for a valid event that is ignored."""
        digest = hashlib.sha256(raw).hexdigest()
        who = {"source_id": source.id, "agent_id": source.agent_id}
        record = settings.record_content
        if source.provider == "github":
            verify_github(raw, header(request.headers, "x-hub-signature-256"), keys[source.id])
            delivery_id(header(request.headers, "x-github-delivery"))
            # GitHub signs only the body, not its delivery or event headers, so identity comes from the signed bytes.
            call = translate_github(
                strict_json(raw),
                header(request.headers, "x-github-event"),
                event_id=digest,
                repositories=source.repositories,
                allowed_senders=source.allowed_senders,
                record_content=record,
                **who,
            )
            event_id, dedupe = digest, True
            if call is None:  # an unsigned event label must not poison receipts for a real delivery of the same body
                return []
        elif source.provider == "standard":
            event_id = verify_standard(raw, request.headers, keys[source.id], settings.timestamp_tolerance)
            call = translate_standard(
                strict_json(raw), event_id=event_id, allow_actions=source.allow_actions, record_content=record, **who
            )
            dedupe = False
        elif source.provider == "gmail":
            payload = strict_json(raw)
            message = payload.get("message")
            event_id = message.get("messageId") if isinstance(message, dict) else None
            call = translate_gmail(
                payload,
                event_id=event_id,
                mailbox=source.mailbox,
                subscription=source.subscription,
                max_age_s=source.gmail_max_age_seconds,
                record_content=record,
                **who,
            )
            dedupe = False
        else:  # outlook
            if _expired(source):
                raise AuthError("Subscription expired.")
            calls = translate_outlook(
                strict_json(raw),
                event_id=digest,
                subscription_id=source.subscription_id,
                mailbox_id=source.mailbox_id,
                tenant_id=source.tenant_id,
                client_states=keys[source.id],
                record_content=record,
                **who,
            )
            return [
                {
                    "source": source.id,
                    "event_id": c.event_id,
                    "digest": hashlib.sha256(c.event_id.encode()).hexdigest(),
                    "call": c.model_dump(),
                }
                for c in calls
            ]
        return [{"source": source.id, "event_id": event_id, "digest": digest, "call": call.model_dump(), "dedupe_body": dedupe}]

    # -- the agent queue API ------------------------------------------------------------------------------------

    def agent(scope: str | None = None):
        """Authenticate before anything else is read: the bearer, its issuer, audience, agent id and scope."""

        def check(request: Request, agent_id: str):
            action = request.url.path.rsplit("/", 1)[-1]
            needed = scope or ("agent:read" if request.method == "GET" else WORKER_SCOPES.get(action, "agent:read"))
            return verifier.verify(bearer(request.headers), agent_id, needed)

        return Depends(check)

    WORKER, OPERATOR = agent(), agent("agent:resolve")

    async def message_of(request: Request, model):
        try:
            return model.model_validate_json(await read_body(request, settings.body_limit))
        except ValidationError:
            raise HTTPException(422, "Invalid message payload.") from None

    def queue(request: Request) -> QueueStore:
        return request.app.state.queue

    @app.post("/v1/agents/{agent_id}/claim")
    def claim(request: Request, agent_id: str, principal=WORKER):
        return queue(request).claim(agent_id, principal.subject, settings.lease_seconds)

    @app.get("/v1/agents/{agent_id}/messages")
    def messages(request: Request, agent_id: str, principal=WORKER):
        return queue(request).messages(agent_id)

    @app.get("/v1/agents/{agent_id}/messages/{message_id}")
    def one_message(request: Request, agent_id: str, message_id: str, principal=WORKER):
        item = queue(request).get(agent_id, message_id)
        if item is None:
            raise HTTPException(404, "Message not found.")
        return item

    @app.post("/v1/agents/{agent_id}/messages/{message_id}/heartbeat")
    async def heartbeat(request: Request, agent_id: str, message_id: str, principal=WORKER):
        body = await message_of(request, LeaseMessage)
        return await run_in_threadpool(
            queue(request).heartbeat,
            agent_id,
            principal.subject,
            message_id,
            body.lease_token,
            settings.lease_seconds,
            include_server_time=True,
        )

    @app.post("/v1/agents/{agent_id}/messages/{message_id}/complete")
    async def complete(request: Request, agent_id: str, message_id: str, principal=WORKER):
        body = await message_of(request, Completion)
        return await run_in_threadpool(
            queue(request).finish,
            agent_id,
            principal.subject,
            message_id,
            body.lease_token,
            body.outcome,
            summary=body.summary,
            note=body.note,
        )

    @app.post("/v1/agents/{agent_id}/messages/{message_id}/events")
    async def events(request: Request, agent_id: str, message_id: str, principal=WORKER):
        body = await message_of(request, AgentEvent)
        return await run_in_threadpool(
            queue(request).publish,
            agent_id,
            principal.subject,
            message_id,
            body.lease_token,
            body.model_dump(exclude={"lease_token"}),
        )

    @app.post("/v1/agents/{agent_id}/messages/{message_id}/resolve")
    async def resolve(request: Request, agent_id: str, message_id: str, principal=OPERATOR):
        body = await message_of(request, ResolveMessage)
        return await run_in_threadpool(queue(request).resolve, agent_id, message_id, body.outcome, summary=body.summary)

    return app

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
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.responses import JSONResponse, PlainTextResponse

from .auth import AgentVerifier, GoogleVerifier, bearer, delivery_id, header, verify_github, verify_standard, webhook_keys
from .contracts import AgentEvent, AuthError, Completion, LeaseMessage, ResolveMessage, TranslationError, strict_json
from .mail import translate_gmail, translate_outlook, validate_outlook_token
from .settings import ServerSettings, Source
from .store import DeliveryConflict, LeaseConflict, QueueFull, QueueStore
from .translation import translate_github, translate_standard

JSON_TYPES = {"application/json", "application/cloudevents+json"}
# The service's own permission names: what a worker token needs to claim a message and to report on one.
WORKER_SCOPES = {"claim": "agent:claim", "report": "agent:report"}


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


# What each failure is answered with: the same fixed words every time, never anything from the request.
_ERROR_ANSWERS = (
    (AuthError, 401, "Invalid callback or bearer authentication.", {"WWW-Authenticate": "Bearer"}),
    (PermissionError, 403, "Insufficient agent scope.", {}),
    (TranslationError, 400, "Invalid callback payload.", {}),
    (TimeoutError, 408, "Request body timed out.", {}),
    (DeliveryConflict, 409, "Delivery identity conflicts with stored content.", {}),
    (LeaseConflict, 409, "Message lease is unavailable.", {}),
    (QueueFull, 503, "Callback queue is full.", {"Retry-After": "5"}),
    (sqlite3.Error, 503, "Callback storage is unavailable.", {"Retry-After": "5"}),
)


def _delivery(source: Source, event_id: str, digest: str, call, **extra) -> dict:
    return {"source": source.id, "event_id": event_id, "digest": digest, "call": call.model_dump(), **extra}


def _github(source: Source, request: Request, raw: bytes, keys, settings: ServerSettings) -> list[dict]:
    verify_github(raw, header(request.headers, "x-hub-signature-256"), keys)
    delivery_id(header(request.headers, "x-github-delivery"))
    # GitHub signs only the body, not its delivery or event headers, so identity comes from the signed bytes.
    digest = hashlib.sha256(raw).hexdigest()
    call = translate_github(
        strict_json(raw),
        header(request.headers, "x-github-event"),
        source_id=source.id,
        agent_id=source.agent_id,
        event_id=digest,
        repositories=source.repositories,
        allowed_senders=source.allowed_senders,
        record_content=settings.record_content,
    )
    if call is None:  # an unsigned event label must not poison receipts for a real delivery of the same body
        return []
    return [_delivery(source, digest, digest, call, dedupe_body=True)]


def _standard(source: Source, request: Request, raw: bytes, keys, settings: ServerSettings) -> list[dict]:
    event_id = verify_standard(raw, request.headers, keys, settings.timestamp_tolerance)
    call = translate_standard(
        strict_json(raw),
        source_id=source.id,
        agent_id=source.agent_id,
        event_id=event_id,
        allow_actions=source.allow_actions,
        record_content=settings.record_content,
    )
    return [_delivery(source, event_id, hashlib.sha256(raw).hexdigest(), call)]


def _gmail(source: Source, request: Request, raw: bytes, keys, settings: ServerSettings) -> list[dict]:
    payload = strict_json(raw)
    message = payload.get("message")
    event_id = message.get("messageId") if isinstance(message, dict) else None
    call = translate_gmail(
        payload,
        source_id=source.id,
        agent_id=source.agent_id,
        event_id=event_id,
        mailbox=source.mailbox,
        subscription=source.subscription,
        max_age_s=source.gmail_max_age_seconds,
        record_content=settings.record_content,
    )
    return [_delivery(source, event_id, hashlib.sha256(raw).hexdigest(), call)]


def _outlook(source: Source, request: Request, raw: bytes, keys, settings: ServerSettings) -> list[dict]:
    if source.expired():
        raise AuthError("Subscription expired.")
    calls = translate_outlook(
        strict_json(raw),
        source_id=source.id,
        agent_id=source.agent_id,
        subscription_id=source.subscription_id,
        mailbox_id=source.mailbox_id,
        tenant_id=source.tenant_id,
        client_states=keys,
        record_content=settings.record_content,
    )
    return [_delivery(source, c.event_id, hashlib.sha256(c.event_id.encode()).hexdigest(), c) for c in calls]


# Verify, then translate, per provider. A provider with no entry (an enabled `mcp` source: its receiver is not
# built) has nothing to verify with, so a body for it is refused and nothing is queued.
_TRANSLATORS = {"github": _github, "standard": _standard, "gmail": _gmail, "outlook": _outlook}


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
    # Two fixed buckets. `ingress` is for traffic anyone can send (health checks, callbacks, failed bearer checks);
    # `agent_limit` is for requests that already passed bearer verification. Unauthenticated floods can therefore
    # never make a worker's heartbeat or completion answer 429 (a lost lease becomes an uncertain run).
    global_limit = Limiter(settings.requests_per_minute)
    agent_limit = Limiter(settings.requests_per_minute)
    ingress = Depends(global_limit.admit)
    source_limits = {sid: Limiter(settings.source_requests_per_minute) for sid in sources}

    @asynccontextmanager
    async def lifespan(app):
        if check_only:
            yield
            return
        path = Path(settings.database)
        owned = store is None
        app.state.queue = store or QueueStore(
            path,
            settings.max_pending,
            settings.max_events,
            record_content=settings.record_content,
            receipt_retention_s=settings.receipt_retention_days * 86400,
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

    for error, status, detail, headers in _ERROR_ANSWERS:
        app.add_exception_handler(
            error, lambda request, exc, d=detail, s=status, h=headers: JSONResponse({"detail": d}, s, headers=h or None)
        )

    @app.get("/healthz", dependencies=[ingress])
    def health():
        return {"status": "ok"}

    # -- provider callbacks -------------------------------------------------------------------------------------

    @app.post("/webhooks/{provider}/{source_id}", status_code=202, dependencies=[ingress])
    async def callback(request: Request, provider: str, source_id: str):
        source = sources.get(source_id)
        if source is None or source.provider != provider:
            raise HTTPException(404, "Callback not found.")
        source_limits[source_id].admit()
        if source.provider == "outlook" and "validationToken" in request.query_params:
            tokens = request.query_params.getlist("validationToken")
            if len(tokens) != 1:
                raise TranslationError("Invalid validation token.")
            if source.expired():
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
            translator = _TRANSLATORS.get(source.provider)
            if translator is None:
                raise TranslationError("This source has no receiver.")
            deliveries = translator(source, request, raw, keys.get(source.id), settings)
        except (ValueError, KeyError, TypeError, RecursionError) as error:
            if isinstance(error, AuthError | TranslationError):
                raise
            raise TranslationError("Invalid callback payload.") from None
        return {"deliveries": await run_in_threadpool(request.app.state.queue.enqueue_many, deliveries)}

    # -- the agent queue API ------------------------------------------------------------------------------------

    def agent(scope: str):
        """Authenticate before anything else is read: the bearer, its issuer, audience, agent id and scope."""

        def check(request: Request, agent_id: str):
            try:
                principal = verifier.verify(bearer(request.headers), agent_id, scope)
            except AuthError:
                global_limit.admit()  # a failed attempt is anyone's traffic: it spends the ingress bucket, not the agents'
                raise
            agent_limit.admit()
            return principal

        return Depends(check)

    READER, OPERATOR = agent("agent:read"), agent("agent:resolve")
    CLAIMER, REPORTER = agent(WORKER_SCOPES["claim"]), agent(WORKER_SCOPES["report"])

    async def message_of(request: Request, model):
        try:
            return model.model_validate_json(await read_body(request, settings.body_limit))
        except ValidationError:
            raise HTTPException(422, "Invalid message payload.") from None

    def queue(request: Request) -> QueueStore:
        return request.app.state.queue

    @app.post("/v1/agents/{agent_id}/claim")
    def claim(request: Request, agent_id: str, principal=CLAIMER):
        return queue(request).claim(agent_id, principal.subject, settings.lease_seconds)

    @app.get("/v1/agents/{agent_id}/messages")
    def messages(
        request: Request,
        agent_id: str,
        status: Literal["pending", "leased", "completed", "blocked", "cancelled", "failed", "uncertain"] | None = None,
        limit: int = Query(50, ge=1, le=100),
        principal=READER,
    ):
        """Newest first. `status=uncertain` finds the rows that block the agent, however many newer ones are queued."""
        return queue(request).messages(agent_id, limit=limit, status=status)

    @app.get("/v1/agents/{agent_id}/messages/{message_id}")
    def one_message(request: Request, agent_id: str, message_id: str, principal=READER):
        item = queue(request).get(agent_id, message_id)
        if item is None:
            raise HTTPException(404, "Message not found.")
        return item

    @app.post("/v1/agents/{agent_id}/messages/{message_id}/heartbeat")
    async def heartbeat(request: Request, agent_id: str, message_id: str, principal=REPORTER):
        body = await message_of(request, LeaseMessage)
        return await run_in_threadpool(
            queue(request).heartbeat,
            agent_id,
            principal.subject,
            message_id,
            body.lease_token,
            settings.lease_seconds,
        )

    @app.post("/v1/agents/{agent_id}/messages/{message_id}/complete")
    async def complete(request: Request, agent_id: str, message_id: str, principal=REPORTER):
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
    async def events(request: Request, agent_id: str, message_id: str, principal=REPORTER):
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

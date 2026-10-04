"""What can go wrong with a provider, in one vocabulary every adapter maps into.

A chain (chain.py) decides what to do from `kind` alone, so adapters must convert their transport's
exceptions with `from_exception` or `from_status` and never let a raw httpx or SDK error escape.
Messages carry the status and a short snippet of what the server said. They never carry a key, a
header, or a request body.

This module is also the one place an error reply is made safe to show. `status_error` takes the status, the body
and headers of a failed HTTP reply, and the secrets and request texts that must not appear, and returns the
ProviderError: the key and anything that looks like an encoded image or token cut out, the quoted request cut out,
and the reply reduced to the message the server meant. Every HTTP adapter (llm.py, stt.py, tts.py) goes through it.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

import httpx

# Another provider may well succeed after these. A `bad_request` is our own fault (a malformed
# request fails everywhere), and `cancelled` is the caller's own decision (the user interrupted): neither says
# anything about the provider, so both are raised at once instead of being tried on the next one.
FAILOVER_KINDS = frozenset({"auth", "rate_limit", "timeout", "transport", "server", "unsupported", "content"})
SNIPPET = 300
# What a provider says when the account behind a key has nothing left to spend. The words are the providers' own
# error codes, not claims about any model. Billing is its own failure: it is not a rate limit that clears in
# seconds, and nothing the server said is worth repeating.
SPENT_WORDS = ("credit_balance_exhausted", "insufficient_quota", "insufficient_credits", "billing_hard_limit_reached")
SPENT_STATUS = 402
CANCELLED = "cancelled"
MIN_ECHO = 10  # a shorter request text than this is too common a string to cut out of an error reply
NO_MESSAGE = "(no message in the reply)"
_LONG_BLOB = re.compile(r"[A-Za-z0-9+/=_-]{80,}")  # base64 of an image, or any other long token a server may echo
_BEARER = re.compile(r"(?i)bearer\s+\S+")


class ProviderError(Exception):
    """One provider could not do what was asked.

    `kind` is one of: auth (key refused), rate_limit, timeout, transport (could not connect or the
    connection dropped), server (5xx), unsupported (model or feature this provider lacks),
    content (a reply that could not be used, such as invalid JSON), bad_request (our fault),
    stream (a stream died after it had started), exhausted (every provider failed), cancelled (the caller
    interrupted the call: the connection was closed on purpose, which is not a fault of the provider).
    """

    def __init__(
        self, message: str, *, kind: str, provider: str = "", status: int | None = None, retry_after: float | None = None
    ):
        super().__init__(message)
        self.kind = kind
        self.provider = provider
        self.status = status
        self.retry_after = retry_after


class AllProvidersFailed(ProviderError):
    """Every provider in a chain failed; `errors` holds (slot name, error) in the order tried."""

    def __init__(self, role: str, errors: list[tuple[str, ProviderError]]):
        tried = "; ".join(f"{name}: {err.kind}" for name, err in errors) or "none was available"
        super().__init__(f"every {role} provider failed ({tried})", kind="exhausted")
        self.role = role
        self.errors = errors


def cancelled(provider: str = "", reason: str = "") -> ProviderError:
    """The error for a call that was interrupted. It is never retried, never failed over and never a SwitchEvent."""
    return ProviderError(
        f"{provider or 'provider'} call cancelled" + (f" ({reason})" if reason else ""), kind=CANCELLED, provider=provider
    )


def snippet(body: object) -> str:
    """The start of a server's reply, on one line, for an error message."""
    text = body if isinstance(body, str) else repr(body)
    return " ".join(text.split())[:SNIPPET]


def spent(status: int, body: object = "") -> bool:
    """Whether a refusal says the account has no credit or quota left (a 402, or a 429 that names it)."""
    if status == SPENT_STATUS:
        return True
    text = body if isinstance(body, str) else repr(body)
    return status in (400, 403, 429) and any(word in text.lower() for word in SPENT_WORDS)


def redact(text: str, secrets: Iterable[str], mark: str = "[redacted]", *, min_len: int = 1) -> str:
    """`text` with each secret replaced by `mark`. A secret shorter than `min_len` is left alone: cut out of a message
    it would mangle ordinary words."""
    for secret in secrets:
        if secret and len(secret) >= min_len:
            text = text.replace(secret, mark)
    return text


def scrub(text: str, secrets: Iterable[str], mark: str = "[redacted]") -> str:
    """`text` without the secrets and without anything that looks like an encoded image or token."""
    return _LONG_BLOB.sub("[...]", redact(text, secrets, mark))


def retry_after(headers: Mapping[str, str]) -> float | None:
    """Seconds the server asked us to wait: `retry-after-ms` (OpenAI sends it), or `retry-after` as seconds or a date."""
    try:
        if milliseconds := headers.get("retry-after-ms"):
            seconds = float(milliseconds) / 1000
        elif value := headers.get("retry-after"):
            try:
                seconds = float(value)
            except ValueError:
                when = parsedate_to_datetime(value)
                seconds = (when if when.tzinfo else when.replace(tzinfo=UTC)).timestamp() - datetime.now(UTC).timestamp()
        else:
            return None
    except (TypeError, ValueError):
        return None
    return max(seconds, 0.0) if math.isfinite(seconds) else None


def _message(data: object, *, nested: bool = False) -> str | None:
    """The human message inside a decoded error reply: `error`, `detail` or `message`, which a server nests, gives as
    a string, or (FastAPI) lists as `{loc, msg}` entries whose `input` is left out because it can be the request."""
    if isinstance(data, str):
        return data
    if isinstance(data, list):
        parts = []
        for item in data:
            if isinstance(item, dict):
                where = ".".join(map(str, item["loc"])) if isinstance(item.get("loc"), list) else ""
                parts.append(f"{where}: {item.get('msg', '')}".strip(": "))
        return "; ".join(parts) or None
    if isinstance(data, dict):
        for key in ("error", "detail", "message", *(("status",) if nested else ())):
            if key in data and (found := _message(data[key], nested=True)):
                return found
    return None


def what_it_said(body: str) -> str:
    """The message of an error reply. A JSON reply is reduced to that message and nothing else: the rest of it may be a
    quoted request, headers or credentials, which no error message may carry. A plain-text reply is kept as it is. A
    bearer token that was quoted is cut out of either."""
    try:
        data = json.loads(body)
    except ValueError:
        return _BEARER.sub("Bearer [redacted]", body)
    message = _message(data)
    return _BEARER.sub("Bearer [redacted]", message) if message is not None else NO_MESSAGE


def echoed(text: str, request_texts: Iterable[str], mark: str = "[redacted]") -> str:
    """`text` with each piece of the request that it quotes cut out: a request body must not reach an error message.

    A quote may differ from the request in case and in how whitespace falls, and a server may quote it inside a JSON
    string, where quotes and newlines are escaped, so all of those forms are cut. A paraphrase is not.
    """
    for piece in sorted({p for p in request_texts if len(p) >= MIN_ECHO}, key=len, reverse=True):
        for form in (piece, json.dumps(piece)[1:-1]):
            words = form.split()
            if words:
                text = re.sub(r"\s+".join(map(re.escape, words)), mark, text, flags=re.IGNORECASE)
    return text


def status_error(
    status: int,
    body: str,
    *,
    provider: str = "",
    headers: Mapping[str, str] | None = None,
    secrets: Iterable[str] = (),
    request_texts: Iterable[str] = (),
    mark: str = "[redacted]",
) -> ProviderError:
    """The ProviderError for a failed HTTP reply, made safe to show.

    The key (`secrets`, replaced by `mark`) and anything that looks like an encoded image or token are cut out of the
    body, then the quoted request (`request_texts`), and the rest is reduced to what the server meant (`what_it_said`).
    A spent account is judged on the whole reply and comes back as `from_status` makes it: fixed text, none of the
    reply in it.
    """
    body = echoed(scrub(body, secrets, mark), request_texts, mark)
    said = body if spent(status, body) else what_it_said(body)
    return from_status(status, said, provider=provider, retry_after=retry_after(headers or {}))


def from_status(status: int, body: object = "", *, provider: str = "", retry_after: float | None = None) -> ProviderError:
    """The error for an HTTP status.

    A spent account is `auth`: this key will not work again for a long while, so the chain rests the slot and
    another provider may answer. Its message is fixed text, with none of the reply in it.
    """
    if spent(status, body):
        who = provider or "provider"
        return ProviderError(
            f"{who} has no credit or quota left (HTTP {status}); add credit to that account or use another provider",
            kind="auth",
            provider=provider,
            status=status,
        )
    detail = f"{provider or 'provider'} answered {status}: {snippet(body)}".rstrip(": ")
    if status in (401, 403):
        kind = "auth"
    elif status in (408, 504):
        kind = "timeout"
    elif status == 429:
        kind = "rate_limit"
    elif status >= 500:
        kind = "server"
    elif status == 404:
        kind = "unsupported"  # an unknown model id or endpoint: another provider may have it
    else:
        kind = "bad_request"
    return ProviderError(detail, kind=kind, provider=provider, status=status, retry_after=retry_after)


def from_exception(exc: BaseException, *, provider: str = "") -> ProviderError:
    """The error for an exception raised while talking to a provider over httpx."""
    if isinstance(exc, ProviderError):
        return exc
    if isinstance(exc, httpx.TimeoutException):
        return ProviderError(f"{provider or 'provider'} timed out", kind="timeout", provider=provider)
    if isinstance(exc, httpx.HTTPStatusError):
        response = exc.response
        return status_error(response.status_code, response.text, provider=provider, headers=response.headers)
    if isinstance(exc, httpx.TransportError):
        return ProviderError(
            f"{provider or 'provider'} could not be reached ({type(exc).__name__})", kind="transport", provider=provider
        )
    return ProviderError(f"{provider or 'provider'} failed ({type(exc).__name__})", kind="transport", provider=provider)

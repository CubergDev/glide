"""What can go wrong with a provider, in one vocabulary every adapter maps into.

A chain (chain.py) decides what to do from `kind` alone, so adapters must convert their transport's
exceptions with `from_exception` or `from_status` and never let a raw httpx or SDK error escape.
Messages carry the status and a short snippet of what the server said. They never carry a key, a
header, or a request body.
"""

from __future__ import annotations

import httpx

# Another provider may well succeed after these. A `bad_request` is our own fault (a malformed
# request fails everywhere), so it is raised at once instead of being tried on the next provider.
FAILOVER_KINDS = frozenset({"auth", "rate_limit", "timeout", "transport", "server", "unsupported", "content"})
SNIPPET = 300
# What a provider says when the account behind a key has nothing left to spend. The words are the providers' own
# error codes, not claims about any model. Billing is its own failure: it is not a rate limit that clears in
# seconds, and nothing the server said is worth repeating.
SPENT_WORDS = ("credit_balance_exhausted", "insufficient_quota", "insufficient_credits", "billing_hard_limit_reached")
SPENT_STATUS = 402


class ProviderError(Exception):
    """One provider could not do what was asked.

    `kind` is one of: auth (key refused), rate_limit, timeout, transport (could not connect or the
    connection dropped), server (5xx), unsupported (model or feature this provider lacks),
    content (a reply that could not be used, such as invalid JSON), bad_request (our fault),
    stream (a stream died after it had started), exhausted (every provider failed).
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
        after = response.headers.get("retry-after")
        return from_status(
            response.status_code,
            response.text,
            provider=provider,
            retry_after=float(after) if after and after.replace(".", "", 1).isdigit() else None,
        )
    if isinstance(exc, httpx.TransportError):
        return ProviderError(
            f"{provider or 'provider'} could not be reached ({type(exc).__name__})", kind="transport", provider=provider
        )
    return ProviderError(f"{provider or 'provider'} failed ({type(exc).__name__})", kind="transport", provider=provider)

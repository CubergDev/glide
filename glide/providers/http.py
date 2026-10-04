"""What every HTTP adapter (llm.py, stt.py, tts.py) does the same way: send, read, and turn a failure into a ProviderError.

- `open_response` sends one request and returns the response with its body unread, registered with the current
  `RunControl` while it waits for its headers (interrupt.py), so a cancel cuts the request that has no response yet.
- `streaming` is `open_response` as a block that also registers the open response and always closes it.
- `read_body` reads a body in pieces, so a deadline holds against a server that dribbles bytes forever and an error
  reply is never read in full.
- `translate` and `translated` turn what httpx raises into a ProviderError. The new error is raised outside the
  `except` block, on purpose: inside it the httpx exception would stay attached as `__context__`, and that exception
  holds the request, and so the key header. A connection that a cancel closed is the cancel, never a transport fault.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable, Iterator
from functools import partial

import httpx

from . import interrupt
from .errors import ProviderError, from_exception

HTTPX_ERRORS = (httpx.HTTPError, httpx.InvalidURL, httpx.StreamError)


def open_response(
    client: httpx.Client, provider: str, method: str, *, timeout: float | httpx.Timeout, **request: object
) -> httpx.Response:
    """Send one request (`url`, `headers`, `json`, ... as `httpx.Client.build_request` takes them) and return the
    response, its body unread; the caller closes it. httpx errors are raised as they are."""
    call = interrupt.Call(client)
    built = client.build_request(method, timeout=timeout, extensions=call.extensions, **request)
    with interrupt.closing(call.abort, provider):  # a model's answer is awaited here, before any header
        return client.send(built, stream=True)


@contextlib.contextmanager
def streaming(
    client: httpx.Client, provider: str, method: str, *, timeout: float | httpx.Timeout, **request: object
) -> Iterator[httpx.Response]:
    """`open_response` for a block: while it runs a cancel closes the open response, and it is closed on the way out."""
    response = open_response(client, provider, method, timeout=timeout, **request)
    try:
        with interrupt.closing(partial(interrupt.abort_response, response), provider):
            yield response
    finally:
        response.close()


def read_body(
    response: httpx.Response,
    *,
    provider: str,
    limit: int | None = None,
    deadline: float | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> bytes:
    """The body, in pieces. With `limit`, reading stops once that many bytes are in (the last piece may overshoot it);
    with `deadline` (on `clock`), a body still arriving after it is a "timeout"."""
    out = bytearray()
    for chunk in response.iter_bytes():
        out += chunk
        if limit is not None and len(out) >= limit:
            break
        if deadline is not None and clock() > deadline:
            raise ProviderError(f"{provider} timed out", kind="timeout", provider=provider)
    return bytes(out)


def translate[T](provider: str, call: Callable[[], T], *, also: tuple[type[BaseException], ...] = ()) -> T:
    """`call()`, with an httpx error (and any of `also`) raised as a ProviderError that holds nothing of the request."""
    failure: ProviderError | None = None
    try:
        return call()
    except (*HTTPX_ERRORS, *also) as exc:
        failure = from_exception(exc, provider=provider)
    interrupt.check(provider)  # a connection closed by a cancel is not a transport fault
    raise failure


def translated[T](provider: str, items: Iterator[T]) -> Iterator[T]:
    """`items`, with an httpx error raised while one is being made turned into a ProviderError as `translate` does.

    Closing this closes `items`, so a connection is given back the moment the consumer stops.
    """
    try:
        while True:
            failure: ProviderError | None = None
            try:
                item = next(items)
            except StopIteration:
                return
            except HTTPX_ERRORS as exc:
                failure = from_exception(exc, provider=provider)
            if failure is not None:
                interrupt.check(provider)
                raise failure
            yield item
    finally:
        close = getattr(items, "close", None)
        if close is not None:
            close()

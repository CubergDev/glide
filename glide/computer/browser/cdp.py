"""Minimal Chrome DevTools Protocol client.

This is the *only* module in the browser backend that talks to the browser.
Everything above it deals in plain dicts, so the backend can be swapped the way
`macos.py` is swapped for a Linux port in the original.

Why CDP instead of pixels: the DOM already knows the text, the role, the label
and the click point. Reading it is an exact answer in single-digit milliseconds.
OCR is a lossy guess that costs hundreds of milliseconds and needs Screen
Recording permission.
"""

from __future__ import annotations

import contextlib
import ipaddress
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

import websocket

from ..control import checkpoint
from ..diagnostics import event


class CDPError(RuntimeError):
    pass


def local_debugger_url(url: str, port: int) -> str:
    """A page's debugger URL, only when it points back at this Chrome's own loopback port.

    The port was free when it was picked, but another process could take it before Chrome
    does and answer /json/list itself. The session then connects nowhere but here.
    """
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "ws" or parsed.hostname != "127.0.0.1" or parsed.port != port:
        raise CDPError(f"debugger URL is not this Chrome's loopback port {port}: {url!r}")
    return url


@contextlib.contextmanager
def _traced(kind: str, **details):
    """Trace one browser request: started, then completed or failed with its seconds.

    Yields a dict the body may add to (the HTTP status); it is reported on completion.
    """
    started = time.perf_counter()
    extra: dict[str, Any] = {}
    event(f"browser_{kind}_started", provider="cdp", **details)
    try:
        yield extra
    except BaseException as error:
        event(f"browser_{kind}_failed", provider="cdp", **details, elapsed_s=time.perf_counter() - started, exception=error)
        raise
    event(f"browser_{kind}_completed", provider="cdp", **details, elapsed_s=time.perf_counter() - started, **extra)


def is_loopback(host: str | None) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host or "").is_loopback
    except ValueError:
        return False


# An opener that names no proxy at all: `http_proxy` in the environment must not carry a request to the
# debugging port of a browser on this machine somewhere else, or make the endpoint answer to someone else.
_DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _get_json(url: str, timeout: float = 5.0) -> Any:
    direct = is_loopback(urllib.parse.urlsplit(url).hostname)
    open_url = _DIRECT.open if direct else urllib.request.urlopen
    with _traced("http", method="GET", endpoint=url, timeout_s=timeout) as done, open_url(url, timeout=timeout) as resp:
        done["status_code"] = getattr(resp, "status", None)
        return json.loads(resp.read())


def _connect(ws_url: str, *, timeout: float, **options: Any) -> websocket.WebSocket:
    """A websocket to `ws_url`, which never goes through a proxy when it is a loopback address.

    websocket-client reads `http_proxy` from the environment unless told otherwise, and only honours a
    no-proxy list next to a proxy host. So a loopback address is given itself as the (never used) proxy
    host, and itself in the no-proxy list: the connection is direct, and a failure of that rule can only
    reach the same loopback host.
    """
    host = urllib.parse.urlsplit(ws_url).hostname
    if is_loopback(host):
        options.update(http_proxy_host=host, http_proxy_port=1, http_no_proxy=[host])
    return websocket.create_connection(ws_url, timeout=timeout, **options)


class Session:
    """One flat CDP session over websocket. Sync, because the loop is sync."""

    def __init__(
        self,
        ws_url: str,
        *,
        origin: str | None = None,
        timeout: float = 30.0,
        navigation_timeout: float | None = None,
        max_size: int | None = 64 * 1024 * 1024,
        suppress_origin: bool = False,
    ):
        self.ws_url = ws_url
        self.timeout = timeout
        self.navigation_timeout = max(timeout, navigation_timeout) if navigation_timeout is not None else timeout
        self._id = 0
        with _traced("socket", endpoint=ws_url, timeout_s=timeout):
            self._ws = _connect(ws_url, timeout=timeout, max_size=max_size, origin=origin, suppress_origin=suppress_origin)
        self.calls = 0

    # -- plumbing ----------------------------------------------------------
    def call(self, method: str, params: dict | None = None, *, session_id: str | None = None) -> dict:
        timeout = self.navigation_timeout if method == "Page.navigate" else self.timeout
        with _traced("request", method=method, endpoint=self.ws_url, request_count=self._id + 1, timeout_s=timeout):
            return self._call(method, params, session_id=session_id)

    def _call(self, method: str, params: dict | None = None, *, session_id: str | None = None) -> dict:
        checkpoint()
        self._id += 1
        msg_id = self._id
        message = {"id": msg_id, "method": method, "params": params or {}}
        if session_id:
            message["sessionId"] = session_id
        self._ws.send(json.dumps(message))
        # Connecting and small control requests keep their short deadline. Navigation can wait on remote TLS/HTTP.
        deadline = time.monotonic() + (self.navigation_timeout if method == "Page.navigate" else self.timeout)
        while True:
            checkpoint(wait=False)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise CDPError(f"{method}: request deadline exceeded")
            self._ws.settimeout(min(0.1, remaining))
            try:
                raw = self._ws.recv()
            except websocket.WebSocketTimeoutException:
                continue
            if not raw:
                raise CDPError("websocket closed")
            data = json.loads(raw)
            if data.get("id") != msg_id:
                continue  # an event, not our reply
            self.calls += 1
            if "error" in data:
                raise CDPError(f"{method}: {data['error']}")
            return dict(data.get("result") or {})

    def evaluate(self, expression: str, *, await_promise: bool = False) -> Any:
        result = self.call(
            "Runtime.evaluate",
            {"expression": expression, "returnByValue": True, "awaitPromise": await_promise},
        )
        if "exceptionDetails" in result:
            raise CDPError(f"JS error: {result['exceptionDetails'].get('text')}")
        return (result.get("result") or {}).get("value")

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._ws.close()

    def __enter__(self) -> Session:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

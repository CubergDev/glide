"""What an adapter does so that a cancel reaches its connection (D2).

The cancel itself is `glide.computer.control.RunControl`: the control that is current while a provider is called
(`current_control()`) is the one the user's stop cancels. This module is the adapter's side of it, small helpers
over `RunControl.closing_on_cancel` and `checkpoint`, and nothing more:

- `closing(close)` registers how to close a connection for as long as it is open. The cancelling thread runs
  `close`, which wakes whatever is blocked reading it.
- `check()` is the checkpoint between two requests, so a cancelled call is never sent again.
- `abort_response(response)` is the `close` for an httpx response whose headers have arrived. It shuts the socket
  down first, because closing a socket that another thread is blocked reading does not wake that thread on every
  platform, and shutting it down does.
- `Call(client)` is the `close` for a request that is still waiting for its headers, which is where a model spends
  most of a non-streaming call: no response exists yet to close. It finds the one connection that carries this
  request in httpcore's pool and shuts that down, and nothing else (a client's other requests are not touched). It
  reads two private attributes of httpcore 1.0.x (`_pool`, `_connection`/`_network_stream`) and does nothing if
  they are not there; tests/test_cancel_wire.py fails if a new httpcore moves them, over a real loopback socket.

An adapter raises only `ProviderError`, so a cancel leaves here as kind "cancelled" (errors.py). What the closed
connection makes httpx raise afterwards is still mapped by the adapter as usual; the chain (chain.py) turns anything
raised once the control is cancelled into the same "cancelled", so a closed connection is never a transport fault.
"""

from __future__ import annotations

import contextlib
import socket
from collections.abc import Callable, Iterator

import httpx

from glide.computer.control import current_control
from glide.computer.models import Abort

from .errors import cancelled


def check(provider: str = "") -> None:
    """Raise `ProviderError(kind="cancelled")` if the current control has been cancelled. A no-op without one."""
    control = current_control()
    if control is not None and control.cancelled.is_set():
        raise cancelled(provider, control.reason)


@contextlib.contextmanager
def closing(close: Callable[[], None], provider: str = "") -> Iterator[None]:
    """For the block, a cancel of the current control calls `close()` from the cancelling thread.

    A control already cancelled closes at once and raises `ProviderError(kind="cancelled")`. Without a control
    nothing is registered and the block runs as it is.
    """
    control = current_control()
    if control is None:
        yield
        return
    try:
        with control.closing_on_cancel(close):
            yield
    except Abort:
        with contextlib.suppress(Exception):
            close()
        raise cancelled(provider, control.reason) from None


def _shutdown(sock: object) -> None:
    if sock is not None:
        with contextlib.suppress(Exception):
            sock.shutdown(socket.SHUT_RDWR)


def abort_response(response: httpx.Response) -> None:
    """Close an open httpx response from any thread, waking a read that is blocked on it."""
    stream = response.extensions.get("network_stream")
    _shutdown(stream.get_extra_info("socket") if stream is not None else None)
    response.close()


class Call:
    """One httpx request, as far as a cancel is concerned. Pass `extensions` to the request, `abort` to `closing`."""

    def __init__(self, client: httpx.Client) -> None:
        self._client = client
        self._token = object()
        self.extensions: dict[str, object] = {"glide_call": self._token}  # httpcore carries these to its own Request

    def abort(self) -> None:
        """Shut down the connection that carries this request, whether or not its headers have arrived."""
        with contextlib.suppress(Exception):  # private attributes of httpcore: if they moved, there is nothing to shut
            pool = self._client._transport._pool
            for queued in list(pool._requests):
                if queued.request.extensions.get("glide_call") is self._token and queued.connection is not None:
                    _shutdown(queued.connection._connection._network_stream.get_extra_info("socket"))

"""Mutation-testing follow-up for `glide/providers/interrupt.py` (see docs/MUTATION_REPORT.md).

tests/test_cancel_wire.py proves the helpers over a real loopback socket. These are the offline pins for what that
file does not isolate: a control that was cancelled already closes at once and raises the cancel, `abort_response`
shuts the socket down before closing, and `Call.abort` shuts only the connection that carries its own request.
"""

from __future__ import annotations

import socket

import pytest

from glide.computer.control import RunControl, controlled
from glide.providers import interrupt
from glide.providers.errors import CANCELLED, ProviderError


class Socket:
    def __init__(self) -> None:
        self.shutdowns: list[int] = []

    def shutdown(self, how: int) -> None:
        self.shutdowns.append(how)


class Stream:
    def __init__(self, sock: Socket | None) -> None:
        self.sock = sock

    def get_extra_info(self, name: str):
        return self.sock if name == "socket" else None


def test_a_control_that_is_already_cancelled_closes_the_connection_at_once_and_raises_the_cancel():
    control = RunControl("t")
    control.cancel("stopped by the user")
    closed = []
    with controlled(control), pytest.raises(ProviderError) as caught, interrupt.closing(lambda: closed.append("socket"), "p"):
        pytest.fail("the block must not run under a cancelled control")
    assert caught.value.kind == CANCELLED and caught.value.provider == "p"
    assert "stopped by the user" in str(caught.value)
    assert closed == ["socket"]


def test_a_failing_close_does_not_hide_the_cancel():
    control = RunControl("t")
    control.cancel("stopped by the user")

    def broken():
        raise OSError("already closed")

    with controlled(control), pytest.raises(ProviderError) as caught, interrupt.closing(broken, "p"):
        pass
    assert caught.value.kind == CANCELLED


def test_abort_response_shuts_the_socket_down_before_closing_the_response():
    events = []

    class Sock(Socket):
        def shutdown(self, how: int) -> None:
            events.append(("shutdown", how))

    class Response:
        def __init__(self) -> None:
            self.extensions = {"network_stream": Stream(Sock())}

        def close(self) -> None:
            events.append(("close", None))

    interrupt.abort_response(Response())
    assert events == [("shutdown", socket.SHUT_RDWR), ("close", None)]


def test_abort_response_without_a_network_stream_still_closes_the_response():
    closed = []

    class Response:
        def __init__(self) -> None:
            self.extensions: dict = {}

        def close(self) -> None:
            closed.append(True)

    interrupt.abort_response(Response())
    assert closed == [True]


class Queued:
    def __init__(self, token: object, sock: Socket | None) -> None:
        self.request = type("Request", (), {"extensions": {"glide_call": token}})()
        stream = Stream(sock)
        self.connection = (
            None if sock is None else type("Conn", (), {"_connection": type("Inner", (), {"_network_stream": stream})()})()
        )


def client_with(*queued: Queued):
    pool = type("Pool", (), {"_requests": list(queued)})()
    return type("Client", (), {"_transport": type("Transport", (), {"_pool": pool})()})()


def test_a_call_shuts_down_only_the_connection_that_carries_its_own_request():
    mine, first, last = Socket(), Socket(), Socket()
    call = interrupt.Call(client_with())
    token = call.extensions["glide_call"]
    call._client = client_with(Queued(object(), first), Queued(token, mine), Queued(object(), last))
    call.abort()
    assert mine.shutdowns == [socket.SHUT_RDWR]
    assert first.shutdowns == [] and last.shutdowns == []


def test_a_call_whose_request_has_no_connection_yet_shuts_nothing_down():
    other = Socket()
    call = interrupt.Call(client_with())
    token = call.extensions["glide_call"]
    call._client = client_with(Queued(token, None), Queued(object(), other))
    call.abort()
    assert other.shutdowns == []


def test_a_call_that_finds_nothing_it_can_read_does_nothing():
    interrupt.Call(object()).abort()  # a client without the private attributes: nothing to shut, and no error

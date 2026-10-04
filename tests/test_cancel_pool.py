"""A request still waiting for a pool slot when the user stops is never sent afterwards (PR9-4175574737).

Over a real loopback socket (see test_cancel_wire.py for the server): the pool allows one connection, the first
request holds it, the second queues behind it and is cancelled while it waits, and then the slot is freed.
"""

from __future__ import annotations

import time

import httpx
import pytest
from test_cancel_wire import WAIT, Wire, run_under

from glide.computer.control import RunControl
from glide.providers import http
from glide.providers.errors import CANCELLED, ProviderError


@pytest.fixture
def serve():
    servers = []

    def start(*script: str) -> Wire:
        servers.append(Wire(*script))
        return servers[-1]

    yield start
    for server in servers:
        server.stop()


def queued(client: httpx.Client) -> int:
    return len(client._transport._pool._requests)  # httpcore's own list; test_cancel_wire's tripwire guards that it exists


def test_a_queued_request_that_is_cancelled_before_its_connection_arrives_sends_nothing(serve):
    server = serve("headers", "answer")
    client = httpx.Client(limits=httpx.Limits(max_connections=1), timeout=httpx.Timeout(WAIT))
    try:
        first = http.open_response(client, "t", "POST", timeout=WAIT, url=server.url + "/v1", json={"n": 1})
        assert server.entered.wait(WAIT)
        control = RunControl()
        done, box = run_under(
            control, lambda: http.open_response(client, "t", "POST", timeout=WAIT, url=server.url + "/v1", json={"n": 2})
        )
        end = time.monotonic() + WAIT
        while queued(client) < 2 and time.monotonic() < end:  # the second request is in the pool, waiting for a slot
            time.sleep(0.005)
        assert queued(client) == 2 and not done.is_set()
        control.cancel("stopped by the user")
        first.close()  # the slot frees: the queued request is handed a connection
        assert done.wait(WAIT)
        error = box["error"]
        assert isinstance(error, ProviderError) and error.kind == CANCELLED
        assert server.requests == 1, "the cancelled request reached the provider"
    finally:
        client.close()


def test_a_request_that_is_not_cancelled_still_goes_through(serve):
    server = serve("answer")
    client = httpx.Client(timeout=httpx.Timeout(WAIT))
    try:
        response = http.open_response(client, "t", "POST", timeout=WAIT, url=server.url + "/v1", json={})
        assert response.status_code == 200 and server.requests == 1
        response.close()
    finally:
        client.close()

"""The CDP session client, offline: a request is read until its deadline and never re-sent, a stop ends a wait, and the
session only connects to this Chrome's own loopback port. A fake socket and clock stand in for the browser."""

import json

import pytest

from glide.computer.browser import cdp
from glide.computer.browser.cdp import CDPError, local_debugger_url
from glide.computer.control import RunControl, controlled
from glide.computer.models import Abort


def test_the_session_only_connects_to_this_chromes_loopback_port():
    url = "ws://127.0.0.1:9222/devtools/page/ABC"
    assert local_debugger_url(url, 9222) == url
    for other in (
        "ws://127.0.0.1:9333/devtools/page/ABC",
        "ws://10.0.0.5:9222/devtools/page/ABC",
        "ws://evil.test:9222/devtools/page/ABC",
        "wss://127.0.0.1:9222/devtools/page/ABC",
    ):
        with pytest.raises(CDPError):
            local_debugger_url(other, 9222)


class _Socket:
    """A websocket that answers one request after `silent` empty polls, as a slow page does."""

    def __init__(self, silent, clock):
        self.silent, self.clock, self.sent = silent, clock, []

    def send(self, data):
        self.sent.append(json.loads(data))

    def settimeout(self, seconds):
        pass

    def recv(self):
        if self.silent:
            self.silent -= 1
            self.clock.advance(1.0)
            raise cdp.websocket.WebSocketTimeoutException()
        return json.dumps({"id": self.sent[-1]["id"], "result": {"ok": True}})


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def advance(self, seconds):
        self.now += seconds

    def __call__(self):
        return self.now


def _session(monkeypatch, silent, timeout):
    clock = _Clock()
    monkeypatch.setattr(cdp.time, "monotonic", clock)
    session = cdp.Session.__new__(cdp.Session)
    session.ws_url, session.timeout, session.navigation_timeout, session._id = "ws://127.0.0.1:1/x", timeout, timeout, 0
    session.calls = 0
    session._ws = _Socket(silent, clock)
    return session


def test_a_request_the_page_answers_before_its_deadline_is_read_not_abandoned(monkeypatch):
    session = _session(monkeypatch, silent=3, timeout=5)
    assert session.call("Runtime.evaluate") == {"ok": True}
    assert len(session._ws.sent) == 1  # one request, never re-sent while waiting


def test_a_request_unanswered_at_the_deadline_fails_without_resending(monkeypatch):
    session = _session(monkeypatch, silent=99, timeout=3)
    with pytest.raises(cdp.CDPError, match="deadline exceeded"):
        session.call("Runtime.evaluate")
    assert len(session._ws.sent) == 1


def test_a_stop_during_a_wait_ends_the_request(monkeypatch):
    session = _session(monkeypatch, silent=99, timeout=60)
    control = RunControl("t")
    control.cancel("stopped")
    with controlled(control), pytest.raises(Abort, match="stopped"):
        session.call("Input.dispatchKeyEvent")
    assert session._ws.sent == []  # the gate is before the send

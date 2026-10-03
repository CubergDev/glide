"""Stop gates in the browser backend: no new input after a stop, a release is never suppressed, and
navigation takes only http(s) addresses."""

import json
from types import SimpleNamespace

import pytest
from test_browser_loop import FakeBrowser, choice, login_page
from test_browser_loop import FakeTypeSafe as BrowserClassifier
from typesafe_sdk import NoulAnswer

from glide.computer.browser import act, cdp
from glide.computer.browser.runner import run_goal
from glide.computer.control import RunControl, checkpoint, controlled
from glide.computer.models import Abort


def test_browser_releases_key_on_stop_and_never_sends_another_press():
    control = RunControl("task")
    calls = []

    def call(method, params):
        checkpoint()
        calls.append(params["type"])
        if params["type"] == "keyDown":
            control.cancel()

    session = SimpleNamespace(call=call)
    with controlled(control):
        act.press(session, "enter")
        with pytest.raises(Abort):
            act.press(session, "escape")
    assert calls == ["keyDown", "keyUp"]


@pytest.mark.parametrize(("confidence", "satisfied", "outcome"), [(0.1, 0, "low_confidence"), (0.9, 1, "done")])
def test_browser_checks_confidence_and_done_before_acting(confidence, satisfied, outcome):
    browser = FakeBrowser(login_page())
    client = SimpleNamespace(
        system_one=lambda **kw: SimpleNamespace(
            answers={"kind": choice("press_enter", confidence), "satisfied": NoulAnswer(noul=satisfied), "element": choice("0")}
        )
    )
    result = run_goal(browser, client, "submit", verbose=False)
    assert result.outcome.startswith(outcome) and not browser.inputs


def test_browser_stop_after_the_classifier_prevents_input():
    browser = FakeBrowser(login_page())
    classifier = BrowserClassifier(("press_enter", None))
    control = RunControl("task")

    def classify(**kwargs):
        result = classifier.system_one(**kwargs)
        control.cancel()
        return result

    with controlled(control), pytest.raises(Abort):
        run_goal(browser, SimpleNamespace(system_one=classify), "submit", verbose=False)
    assert not browser.inputs


@pytest.mark.parametrize(
    "target", ["new tab", "Google Chrome", "file:///etc/passwd", "javascript:alert(1)", "chrome://settings", "http://", ""]
)
def test_navigation_refuses_anything_but_an_http_address(target):
    sent = []
    session = SimpleNamespace(call=lambda method, params: sent.append(method))
    with pytest.raises(ValueError, match="not an http"):
        act.navigate(session, target)
    assert sent == []


def test_navigation_to_http_and_https_goes_through():
    sent = []
    session = SimpleNamespace(call=lambda method, params: sent.append(params["url"]))
    act.navigate(session, "https://example.test/a?b=1")
    act.navigate(session, "http://127.0.0.1:8000/")
    assert sent == ["https://example.test/a?b=1", "http://127.0.0.1:8000/"]


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

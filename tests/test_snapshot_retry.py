"""A read that fails because the document changed is a page problem, never a lost connection, and a write is never retried.

Recorded failure: a read raised during a document change and was reported as "CDP disconnected". Reads are retried a
bounded number of times (`READ_ATTEMPTS`); browser writes are dispatched once.
"""

from types import SimpleNamespace

import pytest
import websocket

from glide.computer.browser.cdp import CDPError
from glide.computer.control import RunControl, controlled
from glide.computer.execution import dom
from glide.computer.execution.contracts import Action, Element, Observation
from glide.computer.models import Abort, BrowserConnectionError, DesktopError

DETAILS = {
    "text": "Uncaught",
    "lineNumber": 22,
    "columnNumber": 6,
    "exception": {"className": "TypeError", "description": "TypeError: transient document state"},
}
SNAPSHOT = {"url": "https://source.test", "document_id": "new", "ready": True}
NAVIGATED = CDPError("Runtime.evaluate: {'code': -32000, 'message': 'Inspected target navigated or closed'}")


def backend_with_replies(monkeypatch, replies):
    backend = dom.BrowserBackend("http://127.0.0.1:9222", "chosen")
    calls, sleeps, events = [], [], []

    def call(method, params=None, **kwargs):
        calls.append(method)
        reply = replies[min(len(calls) - 1, len(replies) - 1)]
        if isinstance(reply, Exception):
            raise reply
        return reply

    backend.page = dom.AttachedPage(SimpleNamespace(call=call), "session")
    monkeypatch.setattr(backend, "connect", lambda: None)
    monkeypatch.setattr(backend, "attach", lambda: None)
    monkeypatch.setattr(backend, "targets", lambda: {"chosen": {"url": SNAPSHOT["url"]}})
    monkeypatch.setattr(dom.time, "sleep", sleeps.append)
    monkeypatch.setattr(dom, "event", lambda name, **details: events.append((name, details)))
    return backend, calls, sleeps, events


@pytest.mark.parametrize(
    "first",
    [
        {"exceptionDetails": DETAILS},
        CDPError("Runtime.evaluate: Execution context was destroyed"),
        NAVIGATED,
        CDPError("Runtime.evaluate: request deadline exceeded"),
    ],
)
def test_transient_document_evaluation_is_retried_read_only(monkeypatch, first):
    backend, calls, sleeps, _ = backend_with_replies(monkeypatch, [first, {"result": {"value": SNAPSHOT}}])
    observed = backend.inspect()
    assert observed.owner == "chosen:new" and observed.url == SNAPSHOT["url"]
    assert calls == ["Runtime.evaluate", "Runtime.evaluate"] and sleeps == [dom.READ_BACKOFF_S]


def test_several_errors_while_the_document_navigates_then_an_answer(monkeypatch):
    """The recorded case: the page is mid-navigation, reads fail twice in different ways, then the new document answers."""
    reply = {"result": {"value": SNAPSHOT}}
    backend, calls, sleeps, events = backend_with_replies(
        monkeypatch, [CDPError("Runtime.evaluate: Execution context was destroyed"), NAVIGATED, reply]
    )
    observed = backend.inspect()
    assert observed.owner == "chosen:new"
    assert calls == ["Runtime.evaluate"] * dom.READ_ATTEMPTS and len(sleeps) == dom.READ_ATTEMPTS - 1
    assert not any(name in {"browser_inspection_failed", "browser_inspection_rejected"} for name, _ in events)


def test_persistent_javascript_error_is_bounded_and_not_a_connection_failure(monkeypatch):
    backend, calls, sleeps, events = backend_with_replies(monkeypatch, [{"exceptionDetails": DETAILS}])
    with pytest.raises(DesktopError, match="Couldn't read the selected page") as caught:
        backend.inspect()
    assert not isinstance(caught.value, BrowserConnectionError)
    assert isinstance(caught.value.__cause__, dom.PageEvaluationError)
    assert caught.value.__cause__.details == DETAILS
    assert calls == ["Runtime.evaluate"] * dom.READ_ATTEMPTS and sleeps == [dom.READ_BACKOFF_S] * (dom.READ_ATTEMPTS - 1)
    errors = [fields["page_content"]["exception_details"] for name, fields in events if name == "browser_snapshot_failed"]
    assert errors == [DETAILS] * dom.READ_ATTEMPTS
    assert not any(name == "browser_inspection_failed" for name, _ in events)


def test_the_page_error_text_is_never_in_the_message(monkeypatch):
    details = {"text": "Uncaught Error: the card number 4111 1111 1111 1111 was on the page"}
    backend, _, _, _ = backend_with_replies(monkeypatch, [{"exceptionDetails": details}])
    with pytest.raises(DesktopError) as caught:
        backend.inspect()
    assert "4111" not in str(caught.value) and "4111" not in str(caught.value.__cause__)
    assert caught.value.__cause__.details == details  # kept for a recorder that may keep content


@pytest.mark.parametrize(
    "lost",
    [
        CDPError("websocket closed"),
        websocket.WebSocketConnectionClosedException("Connection to remote host was lost."),
        BrokenPipeError(32, "Broken pipe"),
        ConnectionResetError(54, "reset"),
    ],
)
def test_a_lost_transport_is_a_connection_failure_and_is_not_retried_as_a_page_script(monkeypatch, lost):
    backend, calls, sleeps, _ = backend_with_replies(monkeypatch, [lost])
    with pytest.raises(BrowserConnectionError):
        backend.inspect()
    assert calls == ["Runtime.evaluate"] and not sleeps


def test_a_refused_request_is_not_a_lost_connection(monkeypatch):
    """The browser answered with a protocol error: it is reachable, so the message must not say it is not."""
    backend, calls, _, events = backend_with_replies(monkeypatch, [CDPError("Runtime.evaluate: {'code': -32601}")])
    with pytest.raises(DesktopError) as caught:
        backend.inspect()
    assert not isinstance(caught.value, BrowserConnectionError) and "Cannot connect" not in str(caught.value)
    assert calls == ["Runtime.evaluate"]
    assert [name for name, _ in events if name.startswith("browser_inspection")] == ["browser_inspection_rejected"]


def test_write_guard_evaluation_is_never_retried(monkeypatch):
    backend, calls, sleeps, _ = backend_with_replies(monkeypatch, [{"exceptionDetails": DETAILS}])
    observed = Observation(
        "browser",
        "chosen:old",
        "https://source.test",
        "chosen",
        elements={"button": Element("button", "Open", "button")},
        capabilities={"click"},
    )
    with pytest.raises(dom.PageEvaluationError):
        backend.execute(Action("click", observed.identity, "button"), observed)
    assert calls == ["Runtime.evaluate"] and not sleeps


@pytest.mark.parametrize("error", [NAVIGATED, CDPError("Runtime.evaluate: Execution context was destroyed")])
def test_a_document_change_during_a_write_guard_surfaces_and_dispatches_nothing(monkeypatch, error):
    backend, calls, sleeps, _ = backend_with_replies(monkeypatch, [error])
    observed = Observation(
        "browser",
        "chosen:old",
        "https://source.test",
        "chosen",
        elements={"field": Element("field", "Search", "input", "", True)},
        capabilities={"type"},
    )
    with pytest.raises(dom.PageEvaluationError):
        backend.execute(Action("type", observed.identity, "field", "unseen query"), observed)
    assert calls == ["Runtime.evaluate"] and not sleeps  # no focus, no clear, no text


def test_a_failed_input_write_is_dispatched_once_and_its_release_still_goes_out(monkeypatch):
    observed = Observation(
        "browser",
        "chosen:old",
        "https://source.test",
        "chosen",
        elements={"button": Element("button", "Open", "button")},
        capabilities={"click"},
    )
    calls = []

    def call(method, params=None, **kwargs):
        calls.append((method, (params or {}).get("type")))
        if method == "Runtime.evaluate":
            return {"result": {"value": {"x": 5, "y": 6}}}
        if (params or {}).get("type") == "mousePressed":
            raise CDPError("Input.dispatchMouseEvent: request deadline exceeded")
        return {}

    backend = dom.BrowserBackend("http://127.0.0.1:9222", "chosen")
    backend.page = dom.AttachedPage(SimpleNamespace(call=call), "session")
    monkeypatch.setattr(dom.time, "sleep", lambda _: pytest.fail("a write must not back off and retry"))
    with pytest.raises(CDPError, match="deadline"):
        backend.execute(Action("click", observed.identity, "button"), observed)
    presses = [c for c in calls if c[1] == "mousePressed"]
    assert presses == [("Input.dispatchMouseEvent", "mousePressed")]
    assert calls[-1] == ("Input.dispatchMouseEvent", "mouseReleased")


def test_navigation_timeout_is_one_attempt_and_says_so(monkeypatch):
    backend, calls, sleeps, _ = backend_with_replies(monkeypatch, [CDPError("Page.navigate: request deadline exceeded")])
    with pytest.raises(DesktopError, match="Timed out while opening"):
        backend.navigate("https://source.test/next")
    assert calls == ["Page.navigate"] and not sleeps


def test_stop_during_snapshot_backoff_prevents_another_read(monkeypatch):
    backend, calls, _, _ = backend_with_replies(monkeypatch, [{"exceptionDetails": DETAILS}])
    control = RunControl("stop-snapshot")
    monkeypatch.setattr(dom.time, "sleep", lambda _: control.cancel())
    with controlled(control), pytest.raises(Abort):
        backend.inspect()
    assert calls == ["Runtime.evaluate"]


def test_navigation_during_read_only_settle_uses_a_fresh_snapshot(monkeypatch):
    backend, calls, sleeps, _ = backend_with_replies(
        monkeypatch, [{"exceptionDetails": DETAILS}, {"result": {"value": SNAPSHOT}}]
    )
    backend.after_input = SimpleNamespace(target="field")
    observed = backend.inspect()
    assert observed.owner == "chosen:new"
    assert calls == ["Runtime.evaluate", "Runtime.evaluate"] and not sleeps


@pytest.mark.parametrize("record_content", [False, True])
def test_recorded_events_carry_no_credentials_and_no_page_text_unless_content_is_opted_in(monkeypatch, record_content):
    from glide.computer import diagnostics

    monkeypatch.setenv("GLIDE_WRITER_API_KEY", "fixture-env-secret-value")
    details = {"text": "Uncaught at https://example.test/private/path?q=my+typed+query fixture-env-secret-value"}
    backend = dom.BrowserBackend("http://127.0.0.1:9222", "chosen")
    backend.page = dom.AttachedPage(SimpleNamespace(call=lambda *a, **k: {"exceptionDetails": details}), "session")
    monkeypatch.setattr(backend, "connect", lambda: None)
    monkeypatch.setattr(backend, "attach", lambda: None)
    monkeypatch.setattr(backend, "targets", lambda: {"chosen": {"url": SNAPSHOT["url"]}})
    monkeypatch.setattr(dom.time, "sleep", lambda _: None)
    recorder = diagnostics.Diagnostics(record_content=record_content)
    with recorder.activate(), pytest.raises(DesktopError) as caught:
        backend.inspect()
    recorded = repr(list(recorder.events))
    assert "fixture-env-secret-value" not in recorded + str(caught.value)
    assert "my+typed+query" not in str(caught.value)
    assert ("my+typed+query" in recorded) == record_content  # with content off nothing of the page error is kept
    assert "private/path" not in recorded or record_content

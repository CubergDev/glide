"""Page evidence is read through the selected tab and tied to the observed document, on every browser provider.

What the fixed reader script keeps and drops, and how a record is validated, belong to `execution.reading` and its
tests (exec-research). Here the reader is a stand-in with the same two names, so these tests need only the backends.
"""

import sys
import types
from types import SimpleNamespace

import pytest

from glide.computer.control import RunControl, controlled
from glide.computer.execution.contracts import Observation
from glide.computer.execution.dom import BrowserBackend
from glide.computer.execution.obscura import ObscuraBackend
from glide.computer.execution.playwright_cli import PlaywrightBackend
from glide.computer.models import Abort, DesktopError

SCRIPT = "fixed page script"


@pytest.fixture(autouse=True)
def reader(monkeypatch):
    module = types.ModuleType("glide.computer.execution.reading")
    module.PAGE_SCRIPT = SCRIPT
    module.page_record = lambda data: {**data, "observed_at": "now"}
    monkeypatch.setitem(sys.modules, "glide.computer.execution.reading", module)


def payload(**kw):
    return {
        "url": "https://example.test/review",
        "title": "Review",
        "text": "The meal was excellent.",
        "links": [],
        "truncated": False,
        "document_id": "123",
        **kw,
    }


@pytest.mark.parametrize("backend_type", [BrowserBackend, ObscuraBackend, PlaywrightBackend])
def test_all_browser_providers_read_the_selected_document(monkeypatch, backend_type):
    backend = object.__new__(backend_type)
    backend.target = "chosen-tab"
    observed = Observation("browser", "chosen-tab:123", "https://example.test/review", "chosen-tab")
    monkeypatch.setattr(backend, "inspect", lambda: observed)
    expressions = []

    def evaluate(expression):
        expressions.append(expression)
        return payload()

    backend.page = SimpleNamespace(evaluate=evaluate)
    page = backend.read_page()
    assert page["text"] == "The meal was excellent." and page["url"] == observed.url
    assert expressions == [SCRIPT] and page["observed_at"]


@pytest.mark.parametrize("changed", [{"url": "https://different.test"}, {"document_id": "124"}])
def test_navigation_during_capture_cannot_misattribute_evidence(monkeypatch, changed):
    backend = BrowserBackend("http://127.0.0.1:9222", "chosen")
    monkeypatch.setattr(backend, "inspect", lambda: Observation("browser", "chosen:123", "https://example.test/review"))
    backend.page = SimpleNamespace(evaluate=lambda _: payload(**changed))
    with pytest.raises(DesktopError, match="changed while reading"):
        backend.read_page()


def test_a_page_that_is_not_ready_is_not_read(monkeypatch):
    backend = BrowserBackend("http://127.0.0.1:9222", "chosen")
    monkeypatch.setattr(backend, "inspect", lambda: Observation("browser", "chosen:123", ready=False))
    backend.page = SimpleNamespace(evaluate=lambda _: pytest.fail("Read a page that was not ready"))
    with pytest.raises(DesktopError, match="not ready to read"):
        backend.read_page()


def test_cancelled_read_never_reaches_the_browser(monkeypatch):
    backend = BrowserBackend("http://127.0.0.1:9222", "chosen")
    monkeypatch.setattr(backend, "inspect", lambda: pytest.fail("Read after cancellation"))
    control = RunControl("read")
    control.cancel()
    with controlled(control), pytest.raises(Abort):
        backend.read_page()

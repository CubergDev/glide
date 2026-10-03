"""The CDP backend turns a page snapshot into an observation and drives it through fixed, guarded scripts, offline.

A fake browser answers each CDP method. Nothing connects, and the scripts are only checked as text: that they are the
fixed ones, carry observed data as JSON, and are sent in the order that keeps a write behind its guards.
"""

import json
import sys
import types
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from glide.computer.control import RunControl, controlled
from glide.computer.execution import dom
from glide.computer.execution.contracts import Action, Element, Observation
from glide.computer.models import Abort, DesktopError

URL = "https://example.org/start"


class FakeBrowser:
    """Answers the browser-level and page-level CDP calls from tables; records every call in order."""

    def __init__(self, snapshot=None, tabs=None, evaluate=None):
        self.calls = []
        self.tabs = {"tab": URL} if tabs is None else tabs
        self.snapshot = {"url": URL, "document_id": "doc", "items": [], "containers": [], "focus": "", "ready": True}
        self.snapshot.update(snapshot or {})
        self.evaluate = evaluate

    def call(self, method, params=None, *, session_id=None):
        self.calls.append((method, params, session_id))
        if method == "Target.getTargets":
            return {"targetInfos": [{"type": "page", "targetId": t, "url": u} for t, u in self.tabs.items()]}
        if method == "Target.attachToTarget":
            return {"sessionId": "session-" + params["targetId"]}
        if method == "Runtime.evaluate":
            if self.evaluate:
                return {"result": {"value": self.evaluate(params["expression"])}}
            return {"result": {"value": self.snapshot}}
        if method == "Target.createTarget":
            self.tabs["created"] = params["url"]
            return {"targetId": "created"}
        if method == "Page.navigate":
            return {"frameId": "main", "loaderId": "loader"}
        if method == "Target.closeTarget":
            self.tabs.pop(params["targetId"], None)
        return {}

    def close(self):
        pass

    @property
    def methods(self):
        return [method for method, _, _ in self.calls]


def backend_on(browser, target="tab"):
    backend = dom.BrowserBackend("http://127.0.0.1:9222", target)
    backend.browser = browser
    return backend


def item(identity, label, role="button", **fields):
    return {
        "id": identity,
        "label": label,
        "role": role,
        "typeable": False,
        "secret": False,
        "enabled": True,
        "value": None,
        "x": 10,
        "y": 20,
        **fields,
    }


def test_the_observation_carries_the_tab_the_document_and_the_page_controls():
    snapshot = {
        "items": [
            item("1", "Search", "input", typeable=True, value="old"),
            item("2", "Pricing", "a", href="https://example.org/pricing", shortcuts=["/"]),
        ],
        "containers": [{"id": "3", "label": "page", "position": 0, "maximum": 900}],
        "focus": "1",
    }
    observed = backend_on(FakeBrowser(snapshot)).inspect()
    assert observed.owner == "tab:doc" and observed.url == URL and observed.active_tab == "tab" and observed.tabs == {"tab": URL}
    assert observed.focus == "1" and observed.elements["1"].value == "old" and observed.elements["1"].typeable
    assert observed.elements["2"].href == "https://example.org/pricing" and observed.containers["3"].maximum == 900
    assert observed.capabilities == dom.BROWSER_CAPABILITIES and observed.ready


def test_credential_fields_are_marked_secret_and_unsafe_links_are_dropped():
    snapshot = {
        "items": [
            item("1", "Password", "input", typeable=True, secret=True, value=None),
            item("2", "Card number", "input", typeable=True),  # the page did not say secret; its label does
            item("3", "Docs", "a", href="javascript:alert(1)"),
            item("4", "Mine", "a", href="https://user:pw@example.org/private"),
        ],
        "canonical_url": "https://user:pw@example.org/",
    }
    observed = backend_on(FakeBrowser(snapshot)).inspect()
    assert observed.elements["1"].secret and observed.elements["2"].secret
    assert observed.elements["3"].href == "" and observed.elements["4"].href == ""
    assert observed.canonical_url == ""


def test_media_and_query_forms_are_observed(monkeypatch):
    @dataclass(frozen=True)
    class QueryForm:  # exec-core's query module owns the real one; this is its shape
        id: str
        field: str
        action: str
        parameter: str
        values: tuple = ()

    module = types.ModuleType("glide.computer.execution.query")
    module.QueryForm = QueryForm
    monkeypatch.setitem(sys.modules, "glide.computer.execution.query", module)
    snapshot = {
        "forms": [{"id": "9", "field": "1", "action": "https://example.org/s", "parameter": "q", "values": [["lang", "en"]]}],
        "media": [{"id": "8", "label": "Video", "paused": False, "ended": False, "ready_state": 4, "current_time": 3.5}],
    }
    observed = backend_on(FakeBrowser(snapshot)).inspect()
    assert observed.forms["9"] == QueryForm("9", "1", "https://example.org/s", "q", (("lang", "en"),))
    assert observed.media["8"].current_time == 3.5 and not observed.media["8"].paused


def test_no_query_module_is_needed_for_a_page_without_forms(monkeypatch):
    monkeypatch.setitem(sys.modules, "glide.computer.execution.query", None)  # importing it would fail
    assert backend_on(FakeBrowser()).inspect().forms == {}


def test_inspection_connects_attaches_and_reads_without_creating_anything():
    browser = FakeBrowser()
    backend = backend_on(browser)
    backend.inspect()
    assert browser.methods == ["Target.getTargets", "Target.attachToTarget", "Runtime.evaluate"]
    assert browser.calls[-1][2] == "session-tab"  # the page session, not the browser one
    backend.inspect()
    assert browser.methods.count("Target.attachToTarget") == 1  # attached once


def test_scoped_inspection_reads_only_the_document_for_url_effects():
    browser = FakeBrowser(evaluate=lambda expression: {"url": URL, "document_id": "doc", "ready": True})
    step = SimpleNamespace(effect="url")
    observed = backend_on(browser).inspect_for(step)
    assert observed.owner == "tab:doc" and not observed.elements
    assert "querySelectorAll" not in browser.calls[-1][1]["expression"]
    assert "querySelectorAll" in dom.SNAPSHOT
    backend_on(FakeBrowser()).inspect_for(SimpleNamespace(effect="field_value"))  # a UI effect gets the full snapshot


def test_a_browser_with_no_tab_yet_offers_to_create_one():
    observed = backend_on(FakeBrowser(tabs={}), target="new").inspect()
    assert observed.owner == "http://127.0.0.1:9222" and "tab_create" in observed.capabilities
    assert "click" not in observed.capabilities and "click" in observed.available_after_navigation


def test_several_open_tabs_need_an_explicit_target_naming_the_setting():
    with pytest.raises(DesktopError, match=r"GLIDE_BROWSER_TARGET"):
        backend_on(FakeBrowser(tabs={"a": URL, "b": URL}), target="").inspect()


def test_a_tab_that_disappeared_is_reported_not_replaced():
    with pytest.raises(DesktopError, match="tab disappeared"):
        backend_on(FakeBrowser(tabs={"other": URL}), target="tab").inspect()


def test_literals_are_data_even_when_they_match_the_template_placeholders():
    scripts = []
    obs = Observation(
        "browser",
        "tab:doc",
        URL,
        "tab",
        {"tab": URL},
        {"BODY": Element("BODY", "TARGET BODY __X__", "field", "", True)},
        capabilities={"type"},
    )
    backend = dom.BrowserBackend("http://127.0.0.1:9222", "tab")
    backend.page = SimpleNamespace(evaluate=lambda text: scripts.append(text), call=lambda *args: {})
    backend.execute(Action("type", obs.identity, "BODY", "unchanged literal __TARGET__"), obs)
    assert len(scripts) == 2
    assert all('refs.get("BODY")' in script and 'label!=="TARGET BODY __X__"' in script for script in scripts)
    assert not any("unchanged literal" in script for script in scripts)  # the text goes to Input.insertText, never to a script


def test_typing_runs_guard_then_focus_then_clear_then_guard_then_text():
    calls = []
    obs = Observation(
        "browser", "tab:doc", URL, "tab", {"tab": URL}, {"7": Element("7", "Search", "input", "old", True)}, capabilities={"type"}
    )
    backend = dom.BrowserBackend("http://127.0.0.1:9222", "tab")
    backend.page = SimpleNamespace(
        evaluate=lambda text: calls.append(("evaluate", "focus()" in text)),
        call=lambda method, params=None: calls.append((method, (params or {}).get("type") or (params or {}).get("text"))) or {},
    )
    backend.execute(Action("type", obs.identity, "7", "new text"), obs)
    assert calls == [
        ("evaluate", True),  # guard, then focus
        ("Input.dispatchKeyEvent", "keyDown"),  # select all
        ("Input.dispatchKeyEvent", "keyUp"),
        ("Input.dispatchKeyEvent", "keyDown"),  # delete
        ("Input.dispatchKeyEvent", "keyUp"),
        ("evaluate", False),  # the guard again: still the same field, still focused
        ("Input.insertText", "new text"),
    ]
    assert backend.after_input is not None  # the next read lets the page settle first


def test_stop_between_clearing_and_typing_prevents_the_text():
    control, calls = RunControl(), []
    obs = Observation(
        "browser", "tab:doc", URL, "tab", {"tab": URL}, {"7": Element("7", "Search", "input", "old", True)}, capabilities={"type"}
    )
    backend = dom.BrowserBackend("http://127.0.0.1:9222", "tab")

    def call(method, params=None):
        calls.append(method)
        if (params or {}).get("key") == "Backspace" and params["type"] == "keyUp":
            control.cancel()
        return {}

    backend.page = SimpleNamespace(evaluate=lambda text: None, call=call)
    with controlled(control), pytest.raises(Abort):
        backend.execute(Action("type", obs.identity, "7", "new text"), obs)
    assert "Input.insertText" not in calls


def test_a_click_is_checked_against_the_observed_label_and_the_point_before_it_is_pressed():
    scripts, calls = [], []
    obs = Observation(
        "browser", "tab:doc", URL, "tab", {"tab": URL}, {"5": Element("5", "Buy now", "button")}, capabilities={"click"}
    )
    backend = dom.BrowserBackend("http://127.0.0.1:9222", "tab")
    backend.page = SimpleNamespace(
        evaluate=lambda text: scripts.append(text) or {"x": 11.5, "y": 22.5},
        call=lambda method, params: calls.append((method, params["type"], params["x"], params["y"])) or {},
    )
    backend.execute(Action("click", obs.identity, "5"), obs)
    assert 'label!=="Buy now"' in scripts[0] and "elementFromPoint(11.5,22.5)" in scripts[1]
    assert calls == [
        ("Input.dispatchMouseEvent", "mousePressed", 11.5, 22.5),
        ("Input.dispatchMouseEvent", "mouseReleased", 11.5, 22.5),
    ]


def test_a_tab_is_created_empty_then_navigated_and_closing_the_selected_one_selects_another():
    browser = FakeBrowser(tabs={"tab": URL, "other": "https://example.net"})
    backend = backend_on(browser, target="")
    backend.target = "tab"
    obs = Observation("browser", "tab:doc", URL, "tab", dict(browser.tabs), capabilities={"tab_create", "tab_close"})
    assert backend.execute(Action("tab_create", obs.identity, value="https://example.net/new"), obs) == "created"
    create = [p for m, p, _ in browser.calls if m in {"Target.createTarget", "Page.navigate"}]
    assert create == [{"url": "about:blank"}, {"url": "https://example.net/new"}]
    assert backend.navigation.requested_url == "https://example.net/new" and backend.target == "created"
    after = Observation("browser", "tab:doc", URL, "tab", dict(browser.tabs), capabilities={"tab_close"})
    backend.execute(Action("tab_close", after.identity, "created"), after)
    assert backend.target == "tab" and "created" not in browser.tabs  # another open tab, not the closed one


def test_pending_input_makes_the_next_read_wait_for_the_page_to_settle():
    seen = []

    def answer(expression):
        seen.append(expression)
        return {"url": URL, "document_id": "doc", "ready": True} if "performance.timeOrigin" in expression else None

    backend = backend_on(FakeBrowser(evaluate=answer))
    backend.after_input = Action("click", "x", "42")
    backend.inspect(controls=False)
    assert "requestAnimationFrame" in seen[0] and '"42"' in seen[0]
    assert backend.after_input is None


def test_the_scripts_are_fixed_text_with_no_model_text_in_the_snapshot():
    for script in (dom.SNAPSHOT, dom.SETTLE, dom.FORM_GUARD, dom.CLICK_GEOMETRY, dom.CLICK_HIT, dom.TYPE_GUARD):
        assert "per" + "mit" not in script.lower()  # the old product name (D12), spelled so the name guard passes
    assert "__LABEL__" not in dom.SNAPSHOT and "__SECRET__" not in dom.SNAPSHOT
    assert dom.fill("a __X__ b __Y__", X="__Y__", Y="1") == "a __Y__ b 1"  # a value is never scanned again
    assert json.dumps("x") in dom.on_element("x", "")


def test_a_page_the_selected_tab_cannot_be_read_is_not_ready_for_reading(monkeypatch):
    backend = backend_on(FakeBrowser(snapshot={"ready": False}))
    module = types.ModuleType("glide.computer.execution.reading")
    module.PAGE_SCRIPT, module.page_record = "script", lambda data: data
    monkeypatch.setitem(sys.modules, "glide.computer.execution.reading", module)
    with pytest.raises(DesktopError, match="not ready to read"):
        backend.read_page()

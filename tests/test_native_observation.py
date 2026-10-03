"""Scoped browser observations avoid UI walks, while UI input retains full validation."""

import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest

from glide.computer import macos
from glide.computer.control import RunControl, controlled
from glide.computer.execution.contracts import (
    Action,
    Element,
    InvalidAction,
    Milestone,
    Observation,
    effect,
    primitive_effect,
    rebind,
    validate,
)
from glide.computer.execution.native import NativeBackend
from glide.computer.models import Abort, AxNode, DesktopError, Field
from glide.computer.platform_adapter import using

MAC_FRONTMOST = macos.frontmost_app_and_pid


def ui_adapter_for(snapshot):
    return SimpleNamespace(
        frontmost_app_and_pid=lambda: ("Editor", 7),
        frontmost_window_bounds=lambda _: (0, 0, 600, 400),
        actionable_elements=lambda *_, **__: (snapshot["nodes"], [], False),
        focused_field=lambda: snapshot["field"],
        ax_value=lambda _: snapshot["value"],
        execution_tabs=lambda _: {"tabs": {}, "active": "", "unsupported": True},
        browser_url=lambda _: None,
        execution_scrolls=lambda _: [],
        execution_labels=lambda _: [],
        screenshot=lambda: pytest.fail("Accessibility observation reached pixels"),
        check_abort=lambda: None,
        ax_set_value=lambda *_: pytest.fail("Stale target was written"),
    )


def field_snapshot():
    ref = object()
    node = AxNode("AXTextField", "Search", 10, 20, 100, 30, False, ref=ref)
    field = Field(node.role, node.label, "", "", node.x, node.y, node.w, node.h, ref=ref)
    other = AxNode("AXButton", "Unrelated", 200, 20, 100, 30, True, ref=object())
    return {"nodes": [node, other], "field": field, "value": ""}


def test_native_field_identity_survives_autocomplete_label_geometry_and_enumeration_changes():
    snapshot = field_snapshot()
    with using(ui_adapter_for(snapshot)):
        backend = NativeBackend("Brave Browser")
        before = backend.inspect()
        action = Action("type", before.identity, before.focus, "unseen query")
        node, other = snapshot["nodes"]
        snapshot["nodes"] = [other, replace(node, label="Search suggestions", x=20, y=30, w=120)]
        snapshot["field"] = replace(snapshot["field"], label="Search suggestions", x=20, y=30, w=120, value=action.value)
        snapshot["value"] = action.value
        after = backend.inspect()
    assert before.focus == after.focus and before.focus in after.elements
    assert before.elements[before.focus].point != after.elements[after.focus].point
    assert primitive_effect(action, before, after)


def test_native_focused_only_fallback_uses_same_field_identity_as_full_tree():
    snapshot = field_snapshot()
    with using(ui_adapter_for(snapshot)):
        backend = NativeBackend("Brave Browser")
        before = backend.inspect()
        snapshot["nodes"] = []
        snapshot["field"] = replace(snapshot["field"], label="Updated field label", x=30, y=40)
        fallback = backend.inspect()
    assert before.focus == fallback.focus
    assert before.elements[before.focus].ref is fallback.elements[fallback.focus].ref


@pytest.mark.parametrize("change", ["label", "geometry", "value"])
def test_stable_native_field_id_does_not_allow_changed_target_before_write(change):
    snapshot = field_snapshot()
    with using(ui_adapter_for(snapshot)):
        backend = NativeBackend("Brave Browser")
        before = backend.inspect()
        action = Action("type", before.identity, before.focus, "unseen query")
        if change == "label":
            snapshot["nodes"][0] = replace(snapshot["nodes"][0], label="Changed field")
        elif change == "geometry":
            snapshot["nodes"][0] = replace(snapshot["nodes"][0], x=50)
        else:
            snapshot["value"] = "user edit"
        fresh = backend.inspect()
        assert before.focus == fresh.focus
        with pytest.raises(InvalidAction, match="selected element changed"):
            bound = rebind(action, before, fresh)
            backend.execute(bound, fresh)


def test_irrelevant_native_element_reordering_preserves_preflight_target():
    snapshot = field_snapshot()
    with using(ui_adapter_for(snapshot)):
        backend = NativeBackend("Brave Browser")
        before = backend.inspect()
        snapshot["nodes"].reverse()
        fresh = backend.inspect()
    bound = rebind(Action("type", before.identity, before.focus, "unseen query"), before, fresh)
    assert bound.target == before.focus and bound.observation == fresh.identity


@pytest.mark.parametrize("unsupported", [False, True])
def test_full_native_observation_never_queries_absent_background_browser_url(unsupported):
    snapshot = field_snapshot()
    adapter = ui_adapter_for(snapshot)
    adapter.execution_tabs = lambda _: {"tabs": {}, "active": "", "ready": True, "unsupported": unsupported}
    adapter.browser_url = lambda _: pytest.fail("Full observation might launch absent browser")
    with using(adapter):
        obs = NativeBackend("Brave Browser").inspect()
    assert obs.app == "Editor" and not obs.browser_front and not obs.url and not obs.tabs


def text_field_observation(*, focused=False):
    return Observation(
        "Editor",
        "Editor:7",
        elements={"field": Element("field", "Search", "AXTextField", "", typeable=True, ref=object())},
        focus="field" if focused else "other",
        capabilities={"type"},
    )


@pytest.mark.parametrize("focused", [False, True])
@pytest.mark.parametrize("write_supported", [False, True])
def test_native_type_focuses_selected_field_before_value_or_keyboard_input(focused, write_supported):
    obs, calls = text_field_observation(focused=focused), []
    ref = obs.elements["field"].ref
    adapter = SimpleNamespace(
        check_abort=lambda: None,
        ax_focus=lambda selected: calls.append(("focus", selected)) or True,
        ax_set_value=lambda selected, value: calls.append(("value", selected, value)) or write_supported,
        clear_field=lambda: calls.append(("clear",)),
        type_text=lambda value: calls.append(("type", value)),
    )
    with using(adapter):
        NativeBackend("Brave Browser").execute(Action("type", obs.identity, "field", "unseen query"), obs)
    expected = [] if focused else [("focus", ref)]
    expected.append(("value", ref, "unseen query"))
    if not write_supported:
        expected.extend([("clear",), ("type", "unseen query")])
    assert calls == expected


def test_native_type_refuses_write_if_observed_field_cannot_be_focused():
    obs, calls = text_field_observation(), []
    adapter = SimpleNamespace(
        check_abort=lambda: None,
        ax_focus=lambda selected: calls.append(("focus", selected)) or False,
        ax_set_value=lambda *_: pytest.fail("Value write despite refused focus"),
        clear_field=lambda: pytest.fail("Keyboard input despite refused focus"),
        type_text=lambda _: pytest.fail("Typing despite refused focus"),
    )
    with using(adapter), pytest.raises(DesktopError, match="Text field cannot be focused"):
        NativeBackend("Brave Browser").execute(Action("type", obs.identity, "field", "unseen query"), obs)
    assert calls == [("focus", obs.elements["field"].ref)]


def test_stop_after_native_type_focus_prevents_any_value_or_keyboard_write():
    obs, control = text_field_observation(), RunControl()
    adapter = SimpleNamespace(
        check_abort=lambda: None,
        ax_focus=lambda _: control.cancel() or True,
        ax_set_value=lambda *_: pytest.fail("Value write after cancellation"),
        clear_field=lambda: pytest.fail("Keyboard input after cancellation"),
        type_text=lambda _: pytest.fail("Typing after cancellation"),
    )
    with using(adapter), controlled(control), pytest.raises(Abort):
        NativeBackend("Brave Browser").execute(Action("type", obs.identity, "field", "unseen query"), obs)


def adapter_for(state, *, app="Brave Browser"):
    calls = []

    def forbid(*_args, **_kwargs):
        pytest.fail("Scoped browser observation walked or captured UI")

    def tabs(browser):
        calls.append(("tabs", browser))
        return state

    adapter = SimpleNamespace(
        frontmost_app_and_pid=lambda: (app, 7),
        execution_tabs=tabs,
        browser_url=forbid,
        frontmost_window_bounds=forbid,
        actionable_elements=forbid,
        focused_field=forbid,
        execution_labels=forbid,
        execution_scrolls=forbid,
        screenshot=forbid,
    )
    return adapter, calls


@pytest.mark.parametrize("kind", [None, "type", "key", "click", "tab_create"])
def test_query_milestone_keeps_full_field_observation_except_direct_navigation(monkeypatch, kind):
    backend, marker = NativeBackend("Brave Browser"), object()
    monkeypatch.setattr(backend, "inspect", lambda: marker)
    step = Milestone(
        "step", "Submit query", "url_query", target="https://example.org/search?q=unseen+query", value="unseen query"
    )
    action = None if kind is None else Action(kind, "old")
    assert backend.inspect_for(step, action) is marker


def test_direct_query_navigation_uses_lightweight_url_verification():
    url = "https://example.org/search?q=unseen+query"
    adapter, calls = adapter_for({"tabs": {"first": url}, "active": "first", "ready": True})
    step = Milestone("step", "Submit query", "url_query", target=url, value="unseen query")
    with using(adapter):
        obs = NativeBackend("Brave Browser").inspect_for(step, Action("navigate", "before", "first", url))
    assert obs.url == url and effect(step, None, obs, obs)
    assert calls == [("tabs", "Brave Browser")] and not obs.elements


@pytest.mark.parametrize("effect_name", ["url", "tab_created", "tab_active", "tab_closed"])
def test_browser_milestones_observe_real_tabs_without_ui_walks(effect_name):
    state = {"tabs": {"first": "https://example.org", "second": "https://docs.example.net"}, "active": "second", "ready": False}
    adapter, calls = adapter_for(state)
    step = Milestone("step", "Browser task", effect_name, target="first", value="https://docs.example.net")
    with using(adapter):
        obs = NativeBackend("Brave Browser").inspect_for(step)
    assert obs.tabs == state["tabs"] and obs.url == "https://docs.example.net" and obs.active_tab == "second"
    assert obs.browser_front and obs.owner == "Brave Browser:7:second:https://docs.example.net"
    assert not obs.ready and not obs.elements and not obs.containers and not obs.focus
    assert obs.capabilities == {"inspect", "navigate", "tab_create", "tab_switch", "tab_close"}
    assert calls == [("tabs", "Brave Browser")]


@pytest.mark.parametrize("kind", ["navigate", "tab_create", "tab_switch", "tab_close"])
def test_browser_action_preflight_still_binds_stable_tabs(kind):
    adapter, _ = adapter_for({"tabs": {"first": "https://example.org"}, "active": "first", "ready": False})
    step = Milestone("step", "Navigate", "url", value="https://example.net")
    with using(adapter):
        backend = NativeBackend("Brave Browser")
        before = backend.inspect_for(step)
        action = Action(
            kind,
            before.identity,
            target="" if kind == "tab_create" else "first",
            value="https://example.net" if kind in {"navigate", "tab_create"} else "",
        )
        fresh = backend.inspect_for(step, action)
        bound = rebind(action, before, fresh)
    validate(bound, fresh)


def test_scoped_readback_counts_observed_created_identity_and_url():
    adapter, _ = adapter_for({"tabs": {"first": "https://example.org"}, "active": "first", "ready": True})
    step = Milestone("step", "Create two tabs", "tab_created", value="https://docs.example.net", quantity=2)
    with using(adapter):
        backend = NativeBackend("Brave Browser")
        before = backend.inspect_for(step)
        action = Action("tab_create", before.identity, value=step.value)
        adapter.execution_tabs = lambda _: {
            "tabs": {**before.tabs, "new": step.value},
            "active": "new",
            "ready": False,
        }
        after = backend.inspect_for(step, action)
    assert effect(step, action, before, after, "new")
    assert not effect(step, action, before, after, "unobserved")


@pytest.mark.parametrize("kind", ["click", "type", "scroll", "key", "inspect"])
def test_ui_actions_under_browser_milestone_require_full_observation(monkeypatch, kind):
    backend, marker = NativeBackend("Brave Browser"), object()
    calls = []
    monkeypatch.setattr(backend, "inspect", lambda: calls.append("full") or marker)
    step = Milestone("step", "Navigate", "url", value="https://example.org")
    assert backend.inspect_for(step, Action(kind, "old")) is marker
    assert calls == ["full"]


@pytest.mark.parametrize("effect_name", ["field_value", "scroll", "element_present", "element_absent", "focus", "app"])
def test_ui_milestones_require_full_observation(monkeypatch, effect_name):
    backend, marker = NativeBackend("Brave Browser"), object()
    monkeypatch.setattr(backend, "inspect", lambda: marker)
    step = Milestone("step", "UI task", effect_name, target="observed control", value="down" if effect_name == "scroll" else "")
    assert backend.inspect_for(step) is marker


def test_browser_snapshot_does_not_assume_front_app_or_page_ready():
    adapter, _ = adapter_for({"tabs": {"first": "https://example.org"}, "active": "first"}, app="Editor")
    step = Milestone("step", "Navigate", "url", value="https://example.org")
    with using(adapter):
        obs = NativeBackend("Brave Browser").inspect_for(step)
    assert obs.app == "Editor" and obs.owner == "Editor:7" and not obs.browser_front and not obs.ready
    assert not effect(step, None, obs, obs)


def test_supported_absent_browser_does_not_issue_url_query_that_could_launch_it():
    adapter, _ = adapter_for({"tabs": {}, "active": "", "ready": True}, app="Editor")
    with using(adapter):
        obs = NativeBackend("Brave Browser").inspect_for(Milestone("step", "Navigate", "url", value="https://example.org"))
    assert not obs.url and not obs.active_tab and not obs.tabs


def test_unsupported_background_browser_does_not_query_url():
    adapter, _ = adapter_for({"tabs": {}, "active": "", "unsupported": True}, app="Editor")
    with using(adapter):
        obs = NativeBackend("Brave Browser").inspect_for(Milestone("step", "Navigate", "url", value="https://example.org"))
    assert obs.capabilities == {"inspect", "navigate"} and not obs.url and not obs.ready


def test_url_only_front_browser_does_not_claim_tab_operations_or_document_readiness():
    adapter, _ = adapter_for({"tabs": {}, "active": "", "unsupported": True})
    adapter.browser_url = lambda _: "https://example.org"
    step = Milestone("step", "Navigate", "url", value="https://example.org")
    with using(adapter):
        obs = NativeBackend("Brave Browser").inspect_for(step)
    assert obs.capabilities == {"inspect", "navigate"} and not obs.ready and not obs.tabs
    assert effect(step, None, obs, obs)


def test_failed_tab_observation_does_not_claim_verified_tab_capabilities():
    adapter, _ = adapter_for({})

    def unavailable(_):
        raise DesktopError("Browser scripting unavailable")

    adapter.execution_tabs = unavailable
    adapter.browser_url = lambda _: None
    with using(adapter):
        obs = NativeBackend("Brave Browser").inspect_for(Milestone("step", "Navigate", "url", value="https://example.org"))
    assert not obs.ready and not obs.url and not obs.tabs
    assert obs.capabilities == {"inspect", "navigate"}


def test_cancelled_scoped_inspection_cannot_continue_into_another_machine_call():
    control = RunControl()
    adapter, _ = adapter_for({"tabs": {}, "active": "", "unsupported": True})
    adapter.frontmost_app_and_pid = lambda: control.cancel() or ("Brave Browser", 7)
    adapter.execution_tabs = lambda _: pytest.fail("Browser query after cancellation")
    with using(adapter), controlled(control), pytest.raises(Abort):
        NativeBackend("Brave Browser").inspect_for(Milestone("step", "Navigate", "url", value="https://example.org"))


def test_cancelled_tab_read_does_not_fall_back_to_another_machine_call():
    control = RunControl()
    adapter, _ = adapter_for({})
    adapter.execution_tabs = lambda _: control.cancel() or {"unsupported": True}
    with using(adapter), controlled(control), pytest.raises(Abort):
        NativeBackend("Brave Browser").inspect_for(Milestone("step", "Navigate", "url", value="https://example.org"))


def test_macos_frontmost_lookup_uses_native_workspace_without_applescript(monkeypatch):
    process = SimpleNamespace(localizedName=lambda: "Brave Browser", processIdentifier=lambda: 123)
    workspace = SimpleNamespace(frontmostApplication=lambda: process)
    monkeypatch.setitem(sys.modules, "AppKit", SimpleNamespace(NSWorkspace=SimpleNamespace(sharedWorkspace=lambda: workspace)))
    assert MAC_FRONTMOST() == ("Brave Browser", 123)


def test_missing_native_foreground_is_reported_instead_of_inventing_app(monkeypatch):
    workspace = SimpleNamespace(frontmostApplication=lambda: None)
    monkeypatch.setitem(sys.modules, "AppKit", SimpleNamespace(NSWorkspace=SimpleNamespace(sharedWorkspace=lambda: workspace)))
    with pytest.raises(DesktopError, match="Frontmost application is unavailable"):
        MAC_FRONTMOST()


@pytest.mark.parametrize("name", ["frontmost_app", "frontmost_app_and_pid", "frontmost_pid"])
def test_offline_guard_covers_native_foreground_entrypoints(name):
    with pytest.raises(RuntimeError, match="a test reached the real machine"):
        getattr(macos, name)()

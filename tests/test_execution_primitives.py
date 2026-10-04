"""Boundary validation, targeting and the bounded waits of the contracts, without a backend.

The tests of the adapters (CDP, native, the macOS and Windows input calls) belong to the backends port.
"""

from dataclasses import replace

import pytest
from execution_world import Computer

from glide.computer.control import RunControl, controlled
from glide.computer.execution.contracts import Action, Element, InvalidAction, Milestone, effect, rebind, validate
from glide.computer.execution.progress import wait_effect
from glide.computer.models import Abort


@pytest.mark.parametrize(
    "kind,target,value,mods",
    [
        ("click", "missing", "", ()),
        ("type", "save", "text", ()),
        ("navigate", "initial", "javascript:alert(1)", ()),
        ("navigate", "initial", "https://name:password@example.org", ()),
        ("tab_close", "missing", "", ()),
        ("scroll", "missing", "down", ()),
        ("scroll", "feed", "around", ()),
        ("key", "", "made-up", ()),
        ("key", "", "a", ("shell",)),
        ("click", "save", "", ("command",)),
    ],
)
def test_invalid_primitive_cannot_reach_adapter(kind, target, value, mods):
    obs = Computer().inspect()
    with pytest.raises(InvalidAction):
        validate(Action(kind, obs.identity, target, value, mods), obs)


def test_stale_disabled_secret_and_unknown_target_are_refused():
    computer = Computer()
    obs = computer.inspect()
    action = Action("type", obs.identity, "name", "hello")
    computer.state.elements["name"] = replace(computer.state.elements["name"], secret=True)
    with pytest.raises(InvalidAction):
        computer.execute(action, obs)
    assert not computer.actions
    obs = computer.inspect()
    with pytest.raises(InvalidAction, match="Credential"):
        validate(replace(action, observation=obs.identity), obs)


def test_scrolling_wrong_container_never_fulfils_feed_milestone():
    computer = Computer()
    before = computer.inspect()
    action = Action("scroll", before.identity, "sidebar", "down")
    computer.execute(action, before)
    step = Milestone("feed", "Scroll feed", "scroll", target="Feed", value="down")
    assert not effect(step, action, before, computer.inspect())


def test_duplicate_field_labels_need_unambiguous_verification():
    obs = Computer().inspect()
    obs.elements["name"] = replace(obs.elements["name"], value="new")
    obs.elements["second"] = replace(obs.elements["name"], id="second")
    assert not effect(Milestone("field", "Edit", "field_value", target="Name", value="new"), None, obs, obs)


def test_readiness_returns_immediately_for_verified_effect_and_waits_only_when_needed(monkeypatch):
    computer = Computer()
    before = computer.inspect()
    action = Action("navigate", before.identity, before.active_tab, "https://example.net")
    computer.execute(action, before)
    monkeypatch.setattr("glide.computer.execution.progress.time.sleep", lambda _: pytest.fail("Unnecessary fixed wait"))
    after, verified = wait_effect(
        computer, Milestone("nav", "Navigate", "url", value=action.value), action, before, "", timeout=1.5
    )
    assert verified and after.url == action.value


def test_stop_during_readiness_prevents_another_poll(monkeypatch):
    computer, control = Computer(), RunControl()
    before = computer.inspect()
    computer.on_inspect = lambda _: control.cancel()
    with controlled(control), pytest.raises(Abort):
        wait_effect(computer, Milestone("nav", "Navigate", "url", value="https://example.net"), None, before, "", timeout=1)
    assert computer.reads == 2


def test_closing_an_unrelated_tab_does_not_complete_requested_close():
    computer = Computer()
    computer.state.tabs["other"] = "https://example.net"
    before = computer.inspect()
    action = Action("tab_close", before.identity, "other")
    computer.execute(action, before)
    assert not effect(Milestone("close", "Close initial", "tab_closed", target="initial"), action, before, computer.inspect())


def test_same_label_in_different_native_window_cannot_replace_target():
    before = Computer().inspect()
    before.elements["name"] = replace(before.elements["name"], ref=object())
    fresh = Computer().inspect()
    fresh.elements["name"] = replace(fresh.elements["name"], ref=object())
    with pytest.raises(InvalidAction, match="selected element changed"):
        rebind(Action("type", before.identity, "name", "new"), before, fresh)


def test_large_values_are_bounded_in_model_observations_without_false_prefix_matches():
    value = "a" * 5000
    obs = Computer().inspect()
    obs.elements["name"] = Element("name", "Name", "field", value, True)
    public = obs.elements["name"].public()
    assert public["value"] is None and public["value_length"] == 5000 and public["value_digest"]
    assert not effect(Milestone("field", "Replace text", "field_value", target="Name", value=value[:2048]), None, obs, obs)
    secret = replace(obs.elements["name"], secret=True).public()
    assert secret["value"] is None and secret["value_length"] is None and secret["value_digest"] is None

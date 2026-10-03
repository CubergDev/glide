"""Rejected model plans are corrected before dispatch, without replaying operations."""

import json
from dataclasses import replace

import pytest
from execution_world import Computer, Jev, Reasoner, drive, response

from glide.computer.control import RunControl
from glide.computer.execution import planning
from glide.computer.execution.contracts import Milestone, Observation, UnsupportedCapability
from glide.computer.generation import GenerationError
from glide.computer.writer import compose_plan

# Exact planner response from live task 63fc668f-9182-4f86-861a-1da093d9ab9b.
RECORDED_PLAN = {
    "question": "",
    "steps": [
        {
            "id": "m1",
            "goal": "Open one new browser tab for the Google search.",
            "effect": "tab_created",
            "target": "browser tab",
            "value": "new tab",
            "quantity": 1,
        },
        {
            "id": "m2",
            "goal": "Search Google for the official Hong Kong Space Museum and Hong Kong Science Museum visitor information pages.",
            "effect": "query_submitted",
            "target": "Google search form",
            "value": "official Hong Kong Space Museum visitor information and Hong Kong Science Museum visitor information",
            "quantity": 1,
        },
    ],
    "unsupported": [],
}


def test_repeated_invalid_plan_stops_after_one_correction(monkeypatch, tmp_path):
    computer, writer = Computer(), Reasoner([RECORDED_PLAN])
    state = drive(monkeypatch, tmp_path, computer, writer, Jev("plan"))
    assert state.outcome == "blocked" and "Invalid milestone URL" in state.failure
    assert len(writer.requests) == 2 and not computer.actions


def test_url_effect_schema_excludes_labels_and_uses_grounded_destinations():
    writer = Reasoner([RECORDED_PLAN])
    destination = "https://configured.example.test/search"
    compose_plan(writer, {"task_context": {"observed_urls": ["", destination, "new tab"]}})
    schema = writer.requests[0].schema["properties"]["steps"]["items"]
    for effect in ("url", "tab_created"):
        variants = [s for s in schema["anyOf"] if effect in s["properties"]["effect"]["enum"]]
        assert len(variants) == 1
        assert variants[0]["properties"]["value"]["enum"] == [destination]
    query = next(s for s in schema["anyOf"] if "query_submitted" in s["properties"]["effect"]["enum"])
    assert "enum" not in query["properties"]["value"]  # Search text stays free text, not a URL choice.


def test_cancellation_during_correction_prevents_dispatch(monkeypatch, tmp_path):
    computer, control = Computer(), RunControl("cancel-correction")
    corrected = response(Milestone("tab", "Open tab", "tab_created", value="https://example.net"))
    writer = Reasoner([RECORDED_PLAN, corrected])
    generate = writer.generate

    def cancel_correction(request, token=None):
        if len(writer.requests) == 1:
            control.cancel()
        return generate(request, token)

    writer.generate = cancel_correction
    state = drive(monkeypatch, tmp_path, computer, writer, control=control)
    assert state.outcome == "aborted" and len(writer.requests) == 2 and not computer.actions


def test_provider_failure_is_not_retried_as_plan_correction(monkeypatch, tmp_path):
    computer, writer = Computer(), Reasoner([GenerationError("Provider failure")])
    state = drive(monkeypatch, tmp_path, computer, writer)
    assert state.outcome == "blocked" and len(writer.requests) == 1 and not computer.actions


def test_unsupported_capability_is_not_retried_or_dropped(monkeypatch, tmp_path):
    computer, writer = Computer(), Reasoner([response(unsupported=["read unavailable file"])])
    state = drive(monkeypatch, tmp_path, computer, writer)
    assert state.outcome == "unsupported" and len(writer.requests) == 1 and not computer.actions


def test_empty_target_can_plan_supported_playback_after_navigation():
    observed = Observation(
        "browser",
        "selected-session",
        capabilities={"inspect", "navigate", "tab_create"},
        available_after_navigation={"query_form", "click", "type", "key", "media_state"},
    )
    opening = Milestone("open", "Open the requested site", "url", value="https://media.example.test")
    search = Milestone("search", "Find the requested recording", "query_submitted", target="search", value="a recording")
    playback = Milestone("play", "Play the first result", "media_playing")
    writer = Reasoner([response(opening, search, playback)])
    plan, question = planning.plan(writer, "Open the media site, search a recording, and play the first result", observed)
    assert plan == [opening, search, playback] and not question
    packet = json.loads(writer.requests[0].text)
    assert packet["observation"]["capabilities"] == ["inspect", "navigate", "tab_create"]
    assert "media_state" in packet["observation"]["available_after_navigation"]
    assert "media_playing proves actual HTML audio/video playback" in writer.requests[0].instructions
    assert "available_after_navigation" in writer.requests[0].instructions


def test_future_capability_requires_navigation_before_the_effect():
    observed = Observation(
        "browser",
        "selected-session",
        capabilities={"inspect", "navigate"},
        available_after_navigation={"media_state"},
    )
    playback = Milestone("play", "Play the requested recording", "media_playing")
    opening = Milestone("open", "Open the media site", "url", value="https://media.example.test")
    with pytest.raises(UnsupportedCapability, match="media_state"):
        planning.validate_plan(response(playback, opening), observed)


@pytest.mark.parametrize("effect_name", ["tab_active", "tab_closed"])
def test_descriptive_tab_target_is_corrected_before_dispatch(monkeypatch, tmp_path, effect_name):
    """Same malformed binding as live museum task 6b14d7cb before it burned recovery calls."""
    computer = Computer()
    url = "https://hk.space.museum/en/web/spm/visit-and-facilities.html"
    computer.state.tabs["museum"] = url
    malformed = Milestone("museum", "Activate or close the existing museum tab", effect_name, target="Existing tab at " + url)
    corrected = replace(malformed, target="", value=url)
    writer = Reasoner([response(malformed), response(corrected)])
    state = drive(monkeypatch, tmp_path, computer, writer, Jev("plan"))
    assert state.answer.achieved, state.failure
    assert [action.kind for action in computer.actions] == ["tab_switch" if effect_name == "tab_active" else "tab_close"]
    packet = json.loads(writer.requests[1].text)
    assert "exact observed tab ID" in packet["plan_correction"]["validation_error"]
    report = json.loads((tmp_path / "run.json").read_text())
    assert report["recoveries"] == 0  # Invalid contracts never reach action selection/recovery.
    variants = writer.requests[0].schema["properties"]["steps"]["items"]["anyOf"]
    matching = [variant for variant in variants if effect_name in variant["properties"]["effect"]["enum"]]
    assert len(matching) == 2
    assert {target for variant in matching for target in variant["properties"]["target"]["enum"]} == {"", "initial", "museum"}


def test_future_tab_url_can_be_bound_after_creation_but_not_before():
    observed = Computer().inspect()
    url = "https://new.example.test"
    create = Milestone("create", "Open new tab", "tab_created", value=url)
    close = Milestone("close", "Close the new tab", "tab_closed", value=url)
    plan, _ = planning.validate_plan(response(create, close), observed)
    assert plan == [create, close]
    with pytest.raises(ValueError, match="exact observed tab ID"):
        planning.validate_plan(response(close, create), observed)


def test_unobserved_future_disclosure_needs_observation_before_planning_expansion():
    observed = Computer().inspect()
    opening = Milestone("open", "Open future page", "url", value="https://future.example.test")
    expand = Milestone("expand", "Reveal its future section", "disclosure_expanded", target="Unobserved section")
    writer = Reasoner([response(opening, expand)])
    with pytest.raises(ValueError, match="observed controller"):
        planning.plan(writer, "Open the page and expand its section", observed)
    assert len(writer.requests) == 2
    for request in writer.requests:
        assert all(
            "disclosure_expanded" not in variant["properties"]["effect"]["enum"]
            for variant in request.schema["properties"]["steps"]["items"]["anyOf"]
        )


def test_correction_cannot_drop_previously_registered_requirements():
    original = Milestone("tabs", "Create three tabs", "tab_created", value="https://example.net", quantity=3)
    writer = Reasoner([RECORDED_PLAN, response(Milestone("other", "Go elsewhere", "url", value="https://other.test"))])
    with pytest.raises(ValueError, match="changed an original requirement"):
        planning.plan(writer, "Create three tabs", Computer().inspect(), steps=[original])
    assert len(writer.requests) == 2
    assert json.loads(writer.requests[-1].text)["original_milestones"][0]["quantity"] == 3

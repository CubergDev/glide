"""The engine's own contract with its caller: events, a dry run, private files, the words a person hears."""

import json
import os
import stat

import pytest
from execution_world import Computer, Jev, Reasoner, drive, response

from glide.computer.actions import Context
from glide.computer.control import RunControl
from glide.computer.execution import engine
from glide.computer.execution.contracts import Element, Milestone
from glide.computer.runner import RunConfig, RunState, run, spoken_failure


def collect(**kwargs):
    events = []
    return events, {"control": RunControl("task", events.append), **kwargs}


def test_a_dry_run_proposes_one_action_and_takes_none(monkeypatch, tmp_path):
    events, kwargs = collect()
    computer = Computer()
    monkeypatch.setattr(engine, "make_backend", lambda _: computer)
    step = Milestone("name", "Set the name", "field_value", target="Name", value="Quarterly report")
    cfg = RunConfig("Set the name", tmp_path, act=False, engine="structured", readiness_timeout=0)  # drive() always acts
    writer, jev = Reasoner([response(step)]), Jev("plan")
    state = run(
        cfg,
        lambda client, history: Context(cfg.goal, "Brave Browser", None, client, writer, history),
        classifier_factory=lambda: jev,
        **kwargs,
    )
    assert state.outcome == "dry run" and not computer.actions and not state.history and not state.failure
    assert state.would_do == "type 'Quarterly report' in 'Name'"
    assert [e.kind for e in events][-1] == "dry_run" and events[-1].text == state.would_do
    assert "Quarterly report" not in (tmp_path / "run.json").read_text()  # words about the page stay in memory (D3)


@pytest.mark.skipif(os.name != "posix", reason="POSIX permissions")
def test_every_file_the_engine_writes_is_private(monkeypatch, tmp_path):
    computer = Computer()
    step = Milestone("open", "Open the site", "url", value="https://example.net")
    drive(monkeypatch, tmp_path / "run", computer, Reasoner([response(step)]), Jev("plan"), record_content=True)
    folder = tmp_path / "run"
    modes = {p.name: stat.S_IMODE(p.stat().st_mode) for p in folder.iterdir()}
    assert {"run.json", "progress.sqlite3", "events.jsonl", "diagnostic.json"} <= set(modes)
    assert set(modes.values()) == {0o600} and stat.S_IMODE(folder.stat().st_mode) == 0o700


def test_the_blocked_event_is_a_sentence_a_person_can_hear(monkeypatch, tmp_path):
    events, kwargs = collect()
    state = drive(
        monkeypatch, tmp_path, Computer(), Reasoner([response(unsupported=["read unavailable file"])]), Jev("plan"), **kwargs
    )
    assert "Available capabilities" in state.failure  # the durable record keeps the list
    assert events[-1].kind == "blocked" and events[-1].outcome == "unsupported"
    assert "Available capabilities" not in events[-1].text and "task report" in events[-1].text


def test_effect_counts_are_not_spoken(monkeypatch, tmp_path):
    events, kwargs = collect()
    step = Milestone("save", "Save and verify", "element_present", target="Saved")
    computer = Computer()
    computer.on_execute = lambda c, a: ""  # the click does nothing
    drive(monkeypatch, tmp_path, computer, Reasoner([response(step)]), Jev("plan"), **kwargs)
    assert events[-1].kind == "blocked" and "Verified" not in events[-1].text and "remain" not in events[-1].text


def test_spoken_failure_leaves_other_failures_whole():
    plain = RunState(
        failure="Targets kept changing before dispatch; no further action was issued. Verified 1 effect(s); 2 remain."
    )
    assert spoken_failure(plain) == "Targets kept changing before dispatch; no further action was issued."
    assert spoken_failure(RunState(failure="Dry run: nothing.")) == "Dry run: nothing."


def test_the_events_of_a_finished_run_are_accepted_then_completed_once(monkeypatch, tmp_path):
    events, kwargs = collect()
    drive(monkeypatch, tmp_path, Computer(), Reasoner([]), Jev("navigate"), goal="Open https://different.test", **kwargs)
    kinds = [e.kind for e in events]
    assert kinds.count("accepted") == 1 and kinds[0] == "accepted" and kinds[-1] == "completed"
    assert "action_checked" in kinds and events[-1].text == "Done. Your task is complete."


def test_a_cached_repeat_is_not_asked_of_jev_again(monkeypatch, tmp_path):
    jev = Jev("tab_create", bindings={"quantity": "5"})
    computer = Computer()
    state = drive(monkeypatch, tmp_path, computer, Reasoner([]), jev, goal="Open 5 tabs of https://example.net")
    assert state.answer.achieved and len(computer.actions) == 5
    assert len(jev.requests) == 2  # the scope, then one binding; the other four tabs reuse it, each re-validated


def test_the_phase_report_names_every_stage_with_its_count(monkeypatch, tmp_path):
    computer = Computer()
    computer.state.elements["saved"] = Element("saved", "Saved", "status")
    step = Milestone("save", "Verify", "element_present", target="Saved")
    drive(monkeypatch, tmp_path, computer, Reasoner([response(step)]), Jev("plan"))
    phases = json.loads((tmp_path / "run.json").read_text())["phases"]
    assert {"task_scope", "initial_observation", "routing", "planning", "observation", "readiness"} <= set(phases)
    assert all(p["calls"] >= 1 and p["seconds"] >= 0 for p in phases.values())

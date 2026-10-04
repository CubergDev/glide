"""Safety findings of the consolidation audit, each shown failing before its fix: an unverified write is never
repeated, a failure's words are fixed unless content is recorded, a plan is grounded to the goal, and so on.
Everything is offline: a fake backend, scripted models and Jev."""

import json

from execution_world import Computer, Jev, Reasoner, drive, response

from glide.computer.control import RunControl
from glide.computer.execution.contracts import Element, Milestone


def changes_an_unrelated_control(machine, action):
    """A write whose effect is not the milestone's: it only redraws a counter control that the plan never mentioned."""
    count = sum(1 for e in machine.state.elements if e.startswith("counter")) + 1
    machine.state.elements[f"counter{count}"] = Element(f"counter{count}", f"Item {count}", "button")
    return ""


# -- finding 1: an action whose effect was not verified is never dispatched again -------------------------------------


def test_a_write_that_only_changed_an_unrelated_control_is_not_repeated(monkeypatch, tmp_path):
    computer, events = Computer(), []
    computer.on_execute = changes_an_unrelated_control
    step = Milestone("never", "Reach a state the click cannot produce", "element_present", target="Confirmed")
    state = drive(
        monkeypatch,
        tmp_path,
        computer,
        Reasoner([response(step)]),
        Jev("plan"),
        steps=20,
        handoffs=5,
        control=RunControl("unverified", emit=events.append),
    )
    assert [a.kind for a in computer.actions] == ["click"]  # dispatched once, then the run stopped
    assert state.outcome == "blocked" and state.uncertain is True
    assert "may or may not have happened" in state.failure
    assert state.readback.endswith("completion unknown")
    report = json.loads((tmp_path / "run.json").read_text())
    assert report["uncertain"] is True and report["steps_taken"] == 1

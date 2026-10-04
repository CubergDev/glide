"""Regressions for review findings against the structured engine (ids in each test name's docstring)."""

import json
import sqlite3

from execution_world import Computer, Reasoner, drive, response

from glide.computer.control import RunControl
from glide.computer.execution.contracts import Action, Milestone
from glide.computer.execution.progress import Ledger


def test_pending_write_blocks_a_restart_under_a_new_task_id(tmp_path):
    """PR10-4175413682: the CLI mints a task id per invocation, so the journal must not be keyed on it."""
    step = Milestone("m", "Open", "url", value="https://example.org")
    first = Ledger(tmp_path / "progress.sqlite3", "task-one")
    first.register([step], 0)
    first.begin(step, Action("navigate", "x", value="https://example.org"))
    first.close()
    second = Ledger(tmp_path / "progress.sqlite3", "task-two")
    assert second.unresolved()
    second.close()


def test_restart_with_another_control_dispatches_nothing(monkeypatch, tmp_path):
    """PR10-4175413682, through the engine: a second invocation with a fresh RunControl is blocked before any action."""
    computer = Computer()

    def unknown(c, action):
        raise OSError("connection lost after write")

    computer.on_execute = unknown
    planner = Reasoner([response(Milestone("nav", "Reach", "url", value="https://example.net"))])
    first = drive(monkeypatch, tmp_path, computer, planner, control=RunControl("first-invocation"))
    assert first.uncertain and len(computer.actions) == 1
    second = drive(monkeypatch, tmp_path, computer, planner, control=RunControl("second-invocation"))
    assert second.outcome == "blocked" and len(computer.actions) == 1
    assert "unresolved operation" in second.failure


def test_planner_milestone_ids_stay_out_of_stored_run_data(monkeypatch, tmp_path):
    """PR10-4175413687: a planner id can echo the goal, so neither progress.sqlite3 nor run.json holds it."""
    sentinel = "email_alice_medical_results"
    state = drive(
        monkeypatch,
        tmp_path,
        Computer(),
        Reasoner([response(Milestone(sentinel, "Open the page", "url", value="https://example.net"))]),
    )
    assert state.outcome == "done" and state.progress[0]["id"] == sentinel  # the planner still sees its own ids
    assert sentinel not in (tmp_path / "run.json").read_text()
    assert json.loads((tmp_path / "run.json").read_text())["progress"][0]["id"] == "m1"
    with sqlite3.connect(tmp_path / "progress.sqlite3") as db:
        assert sentinel not in str(list(db.iterdump()))

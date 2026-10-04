"""Regressions exercise the real runner, planner parsing, policy and effect ledger offline."""

import json
import sqlite3
from dataclasses import replace

import pytest
from execution_world import SCENARIOS, Computer, Jev, Reasoner, drive, response

from glide.computer.control import RunControl
from glide.computer.execution.contracts import Element, Milestone
from glide.computer.execution.progress import Ledger
from glide.computer.generation import GenerationResult


@pytest.mark.parametrize("goal", SCENARIOS)
def test_diverse_goals_reach_verified_effects(monkeypatch, tmp_path, goal):
    computer, jev = Computer(), Jev()
    steps = SCENARIOS[goal]
    reasoner = Reasoner([response(*steps)])
    state = drive(monkeypatch, tmp_path, computer, reasoner, jev, goal=goal)
    assert state.answer and state.answer.achieved and state.outcome == "done"
    assert all(p["verified"] == p["requested"] and p["remaining"] == 0 for p in state.progress)
    assert len(reasoner.requests) == 1 and computer.closed
    assert state.answer.text == "Done. Your task is complete."
    if "tabs" in goal:
        expected = steps[0].quantity
        assert len(computer.state.tabs) == expected + 1
        assert len([a for a in computer.actions if a.kind == "tab_create"]) == expected
        # Task scope, observed workflow and concrete selection, regardless of quantity.
        assert len(jev.requests) == 3
    if "X" in goal:
        assert computer.state.url == "https://x.com" and computer.state.containers["feed"].position == 100
        assert computer.state.containers["sidebar"].position == 0


def test_simple_literal_navigation_has_no_reasoner_or_confirmation_call(monkeypatch, tmp_path):
    computer, reasoner, jev = Computer(), Reasoner([RuntimeError("must not call")]), Jev("navigate")
    state = drive(monkeypatch, tmp_path, computer, reasoner, jev, goal="visit https://example.net")
    assert state.answer.achieved and computer.state.url == "https://example.net"
    assert not reasoner.requests and len(jev.requests) == 2


def test_stop_during_repeated_actions_reconciles_effect_and_leaves_remaining(monkeypatch, tmp_path):
    computer, control = Computer(), RunControl("offline")
    execute = computer.execute

    def stop_after_three(c, action):
        c.on_execute = None
        result = execute(action, c.state)
        c.on_execute = stop_after_three
        if len(c.actions) >= 6:  # test wrapper + simulated primitive log once each
            control.cancel()
        return result

    computer.on_execute = stop_after_three
    state = drive(monkeypatch, tmp_path, computer, Reasoner([response(*SCENARIOS["open ten Google tabs"])]), control=control)
    assert state.outcome == "aborted" and not state.answer
    assert len(computer.state.tabs) == 4
    assert state.progress[0] == {"id": "tabs", "requested": 10, "verified": 3, "remaining": 7}
    assert not state.uncertain and state.readback == "effect verified"


def test_same_receipt_cannot_count_multiple_created_tabs(monkeypatch, tmp_path):
    computer = Computer()

    def duplicate(c, action):
        c.state.tabs["one"] = action.value
        return "one"

    computer.on_execute = duplicate
    state = drive(monkeypatch, tmp_path, computer, Reasoner([response(*SCENARIOS["open ten Google tabs"])]))
    assert state.outcome == "blocked" and state.uncertain
    assert state.progress[0]["verified"] == 1 and state.progress[0]["remaining"] == 9
    assert len(computer.actions) == 2


def test_progress_is_durable_and_privacy_preserving(monkeypatch, tmp_path):
    secret = "user-private-value"
    state = drive(
        monkeypatch,
        tmp_path,
        Computer(),
        Reasoner([response(Milestone("field", "Private text", "field_value", target="Name", value=secret))]),
    )
    ledger = Ledger(tmp_path / "progress.sqlite3", "offline")
    assert ledger.summary() == state.progress and not ledger.unresolved()
    ledger.close()
    assert secret not in (tmp_path / "run.json").read_text()
    with sqlite3.connect(tmp_path / "progress.sqlite3") as db:
        assert secret not in str(list(db.iterdump()))


def test_unknown_write_never_replays_and_restarted_task_is_blocked(monkeypatch, tmp_path):
    computer = Computer()

    def unknown(c, action):
        raise OSError("connection lost after write")

    computer.on_execute = unknown
    planner = Reasoner([response(*SCENARIOS["open ten Google tabs"])])
    first = drive(monkeypatch, tmp_path, computer, planner)
    assert first.uncertain and first.progress[0]["verified"] == 0 and len(computer.actions) == 1
    second = drive(monkeypatch, tmp_path, computer, planner)
    assert second.outcome == "blocked" and len(computer.actions) == 1
    assert "unresolved operation" in second.failure


def test_contradictory_navigation_replans_without_losing_requirements(monkeypatch, tmp_path):
    computer = Computer()
    original = SCENARIOS["navigate to X and scroll"]
    count = []

    def contradictory(c, action):
        if action.kind == "navigate":
            count.append(True)
            if len(count) < 3:
                c.state.url = "https://example.org/wrong"
                return ""
        c.on_execute = None
        result = c.execute(action, c.state)
        c.on_execute = contradictory
        return result

    computer.on_execute = contradictory
    reasoner = Reasoner([response(*original), response(*original)])
    state = drive(monkeypatch, tmp_path, computer, reasoner)
    assert state.answer.achieved and state.plan_revisions == 1 and len(reasoner.requests) == 2
    packet = json.loads(reasoner.requests[-1].text)
    assert packet["contradiction"] and packet["verified_progress"][0]["remaining"] == 1
    assert len(packet["original_milestones"]) == 2


def test_replanning_cannot_reduce_requested_quantity(monkeypatch, tmp_path):
    original = Milestone("scroll", "Scroll ten times", "scroll", target="Feed", value="down", quantity=10)
    computer = Computer()
    computer.state.containers["feed"] = replace(computer.state.containers["feed"], position=5000)
    reasoner = Reasoner([response(original), response(replace(original, quantity=1))])
    state = drive(monkeypatch, tmp_path, computer, reasoner)
    assert state.outcome == "blocked" and state.progress[0]["remaining"] == 10
    assert "changed an original requirement" in state.failure


def test_genuine_no_progress_loop_is_bounded(monkeypatch, tmp_path):
    computer = Computer()
    computer.state.containers["feed"] = replace(computer.state.containers["feed"], position=5000)
    step = Milestone("scroll", "Scroll", "scroll", target="Feed", value="down", quantity=3)
    reasoner = Reasoner([response(step)])
    state = drive(monkeypatch, tmp_path, computer, reasoner)
    assert state.outcome == "blocked" and not state.answer
    assert len(reasoner.requests) == 3 and len(computer.actions) == 6
    assert state.progress[0]["verified"] == 0


def test_partial_or_invalid_plans_never_execute(monkeypatch, tmp_path):
    for i, reply in enumerate(
        [
            GenerationResult('{"steps":', "fake", completed=False),
            {"question": "", "steps": []},
            response(
                Milestone("x", "Goal", "url", value="https://example.net"),
                Milestone("x", "Goal", "url", value="https://example.net"),
            ),
        ]
    ):
        computer = Computer()
        state = drive(monkeypatch, tmp_path / str(i), computer, Reasoner([reply]))
        assert state.outcome == "blocked" and not computer.actions


def test_stop_during_planning_prevents_first_action(monkeypatch, tmp_path):
    control = RunControl("offline")
    reasoner = Reasoner([response(*SCENARIOS["open ten Google tabs"])])
    generate = reasoner.generate

    def stop(request, cancel):
        result = generate(request, cancel)
        control.cancel()
        return result

    reasoner.generate = stop
    computer = Computer()
    state = drive(monkeypatch, tmp_path, computer, reasoner, control=control)
    assert state.outcome == "aborted" and not computer.actions and not state.uncertain


def test_stale_target_is_rejected_before_input(monkeypatch, tmp_path):
    computer = Computer()

    def changing(c):
        el = c.state.elements["name"]
        c.state.elements["name"] = replace(el, value=str(c.reads))

    computer.on_inspect = changing
    state = drive(
        monkeypatch,
        tmp_path,
        computer,
        Reasoner([response(Milestone("value", "Edit Name", "field_value", target="Name", value="new"))]),
    )
    assert state.outcome == "blocked" and not computer.actions and "Targets kept changing" in state.failure


def test_tab_close_and_switch_are_generic(monkeypatch, tmp_path):
    computer = Computer()
    computer.state.tabs["other"] = "https://example.net"
    steps = [
        Milestone("switch", "Switch tab", "tab_active", target="other"),
        Milestone("close", "Close initial tab", "tab_closed", target="initial"),
    ]
    state = drive(monkeypatch, tmp_path, computer, Reasoner([response(*steps)]))
    assert state.answer.achieved and computer.state.active_tab == "other" and "initial" not in computer.state.tabs


def test_quantity_remaining_survives_history_window(monkeypatch, tmp_path):
    computer = Computer()
    step = Milestone("tabs", "Create sixteen tabs", "tab_created", value="https://example.net", quantity=16)
    state = drive(monkeypatch, tmp_path, computer, Reasoner([response(step)]))
    assert state.answer.achieved and state.progress[0]["verified"] == 16 and len(state.history) == 16


@pytest.mark.parametrize(
    "goal,kind,expected",
    [("open 12 https://example.net tabs", "tab_create", 12), ("scroll the Feed down four times", "scroll", 4)],
)
def test_resolved_quantities_use_generic_primitives_without_reasoner(monkeypatch, tmp_path, goal, kind, expected):
    computer, reasoner, jev = Computer(), Reasoner([RuntimeError("not needed")]), Jev(kind, bindings={"quantity": str(expected)})
    state = drive(monkeypatch, tmp_path, computer, reasoner, jev, goal=goal)
    assert state.answer.achieved and len(computer.actions) == expected
    assert not reasoner.requests and len(jev.requests) == 2
    assert state.progress[0]["verified"] == expected


def test_clarification_precedes_any_action_and_carries_reply(monkeypatch, tmp_path):
    computer = Computer()
    reasoner = Reasoner(
        [
            response(question="Which destination?"),
            response(Milestone("nav", "Reach destination", "url", value="https://example.net")),
        ]
    )
    asked = []

    def ask(question):
        assert not computer.actions
        asked.append(question)
        return "https://example.net"

    state = drive(monkeypatch, tmp_path, computer, reasoner, ask=ask)
    assert state.answer.achieved and asked == ["Which destination?"]
    assert json.loads(reasoner.requests[1].text)["user_reply"] == "https://example.net"


def test_recovery_question_cannot_discard_original_quantities(monkeypatch, tmp_path):
    computer = Computer()
    computer.state.containers["feed"] = replace(computer.state.containers["feed"], position=5000)
    original = Milestone("scroll", "Scroll ten times", "scroll", target="Feed", value="down", quantity=10)
    reasoner = Reasoner(
        [response(original), response(question="Should I change direction?"), response(replace(original, quantity=1))]
    )
    state = drive(monkeypatch, tmp_path, computer, reasoner, ask=lambda _: "No")
    assert state.outcome == "blocked" and state.progress[0]["remaining"] == 10
    assert json.loads(reasoner.requests[-1].text)["original_milestones"][0]["quantity"] == 10


def test_unrelated_live_status_does_not_invalidate_selected_navigation(monkeypatch, tmp_path):
    computer = Computer()

    def clock(c):
        c.state.elements["clock"] = Element("clock", "Clock", "status", str(c.reads))

    computer.on_inspect = clock
    state = drive(
        monkeypatch, tmp_path, computer, Reasoner([RuntimeError("not needed")]), Jev("navigate"), goal="visit https://example.net"
    )
    assert state.answer.achieved and len(computer.actions) == 1


def test_multistep_menu_operation_is_verified_without_finishing_goal(monkeypatch, tmp_path):
    computer = Computer()
    computer.state.elements.pop("save")
    computer.state.elements["menu"] = Element("menu", "File", "button")

    def open_menu(c, action):
        if action.target == "menu":
            c.state.elements["save"] = Element("save", "Save", "button")
            return ""
        c.on_execute = None
        return c.execute(action, c.state)

    computer.on_execute = open_menu

    def select(state, criteria):
        wanted = "save" if any(e["id"] == "save" for e in state["observation"]["elements"]) else "menu"
        return next(
            k
            for k, v in criteria.items()
            if k != "replan" and json.loads(v)["kind"] == "click" and json.loads(v)["target"] == wanted
        )

    step = Milestone("save", "Save document", "element_present", target="Saved")
    result = drive(monkeypatch, tmp_path, computer, Reasoner([response(step)]), Jev(selection=select))
    assert result.answer.achieved and result.progress[0]["verified"] == 1
    assert result.history == ["click", "click"]
    with sqlite3.connect(tmp_path / "progress.sqlite3") as db:
        assert db.execute("SELECT status FROM operations ORDER BY rowid").fetchall() == [("verified",), ("verified",)]


def test_focus_cycle_cannot_claim_completion(monkeypatch, tmp_path):
    step = Milestone("save", "Save document", "element_present", target="Saved")
    computer = Computer()

    def select(_, criteria):
        return "key"

    reasoner = Reasoner([response(step)])
    result = drive(monkeypatch, tmp_path, computer, reasoner, Jev(selection=select))
    assert result.outcome == "blocked" and not result.answer and result.progress[0]["verified"] == 0
    # Changed from 12 identical key presses: a press that moved focus but did not show the effect is not sent again.
    assert len(computer.actions) == 1 and result.uncertain


@pytest.mark.parametrize(
    "goal,kind,expected",
    [("open four https://docs.python.org/3/ tabs", "tab_create", 4), ('type "twenty 123" in Name', "type", 1)],
)
def test_literal_argument_numbers_are_not_action_quantities(monkeypatch, tmp_path, goal, kind, expected):
    computer, reasoner = Computer(), Reasoner([RuntimeError("must not plan")])
    result = drive(monkeypatch, tmp_path, computer, reasoner, Jev(kind, bindings={"quantity": str(expected)}), goal=goal)
    assert result.answer.achieved and len(computer.actions) == expected and not reasoner.requests


def test_unbounded_quantity_requires_planning_not_one_silent_action(monkeypatch, tmp_path):
    computer = Computer()
    reasoner = Reasoner([response(question="Please choose at most 100 new tabs per task.")])
    result = drive(monkeypatch, tmp_path, computer, reasoner, goal="open 1000 https://example.net tabs")
    assert result.outcome == "blocked" and not computer.actions and len(reasoner.requests) == 1


def test_journal_failure_blocks_before_models_or_actions(monkeypatch, tmp_path):
    def unavailable(*_):
        raise PermissionError("journal read-only")

    monkeypatch.setattr("glide.computer.execution.engine.Ledger", unavailable)
    computer, reasoner, jev = Computer(), Reasoner([]), Jev()
    result = drive(monkeypatch, tmp_path, computer, reasoner, jev)
    assert result.outcome == "blocked" and "journal" in result.failure
    assert not computer.actions and not reasoner.requests and not jev.requests


def test_ambient_changes_do_not_verify_an_unknown_click(monkeypatch, tmp_path):
    computer = Computer()

    def clock(c, _):
        c.state.elements["clock"] = Element("clock", "Clock", "status", str(c.reads))
        return ""

    computer.on_execute = clock
    step = Milestone("save", "Save document", "element_present", target="Saved")
    result = drive(monkeypatch, tmp_path, computer, Reasoner([response(step)]))
    assert result.outcome == "blocked" and result.uncertain and not result.answer
    assert len(computer.actions) == 1 and result.progress[0]["verified"] == 0


@pytest.mark.parametrize("phase", ["routing", "action_selection"])
def test_stop_during_jev_never_dispatches(monkeypatch, tmp_path, phase):
    computer, jev, control = Computer(), Jev(), RunControl("offline")
    call = jev.system_one

    def stop(state, questions):
        result = call(state, questions)
        if ("milestone" in state) == (phase == "action_selection"):
            control.cancel()
        return result

    jev.system_one = stop
    result = drive(monkeypatch, tmp_path, computer, Reasoner([response(*SCENARIOS["open ten Google tabs"])]), jev, control)
    assert result.outcome == "aborted" and not result.answer and not computer.actions


def test_stop_during_clarification_cannot_resume_task(monkeypatch, tmp_path):
    computer, control = Computer(), RunControl("offline")
    reasoner = Reasoner([response(question="Which destination?")])

    def ask(_):
        control.cancel()
        return "https://example.net"

    result = drive(monkeypatch, tmp_path, computer, reasoner, control=control, ask=ask)
    assert result.outcome == "aborted" and not computer.actions and len(reasoner.requests) == 1


def test_abort_corner_reason_is_reported_and_preserved_without_content(monkeypatch, tmp_path):
    from glide.computer.models import Abort

    computer = Computer()

    def abort(_):
        raise Abort("mouse in top-left corner")

    computer.on_inspect = abort
    events = []
    state = drive(monkeypatch, tmp_path, computer, Reasoner([]), control=RunControl("diagnostic", events.append))
    assert state.outcome == "aborted" and state.cancel_reason == "mouse in top-left corner" and not computer.actions
    assert json.loads((tmp_path / "run.json").read_text())["cancel_reason"] == "mouse in top-left corner"
    assert "Move the pointer" in events[-1].text

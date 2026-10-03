"""Recorded failures of the execution engine (HANDOFF section 7), replayed offline with fakes and a fake clock.

Each test names the failure it keeps from coming back. What a fake cannot prove (that a live site behaves so) is
not claimed: these show that the engine's own handling is right for the recorded shape.
"""

import json

import pytest
from execution_world import SCENARIOS, Computer, FakeClock, Jev, Reasoner, drive, response
from test_planning_repair import RECORDED_PLAN

from glide.computer import config
from glide.computer.control import RunControl
from glide.computer.execution.contracts import Element, Media, Milestone

# -- readiness deadline too short (1.5 s, later raised): a page ready at 3 s was given up on ---------------------


def not_ready_until(computer, clock, seconds):
    def inspect(machine):
        machine.state.ready = clock.now >= seconds

    computer.on_inspect = inspect


def save_plan():
    return Reasoner([response(Milestone("save", "Save and verify", "element_present", target="Saved"))])


def test_a_page_ready_at_three_seconds_is_read_without_reopening_under_the_default_deadline(monkeypatch, tmp_path):
    clock, computer = FakeClock(monkeypatch), Computer()
    not_ready_until(computer, clock, 3.0)
    state = drive(monkeypatch, tmp_path, computer, save_plan(), readiness_timeout="default")
    assert config.DEFAULT_READINESS_TIMEOUT > 3.0
    assert state.answer and state.answer.achieved, state.failure
    assert [a.kind for a in computer.actions] == ["click"]  # no navigate, no tab_create: the same page, read again
    assert 3.0 <= clock.now < 3.5 and computer.reads < 90  # the waiting is polling, and it ends as the page is ready


@pytest.mark.parametrize("deadline", [1.5, 2.0, 7.0])
def test_a_page_that_is_never_ready_blocks_exactly_at_the_configured_deadline(monkeypatch, tmp_path, deadline):
    clock, computer = FakeClock(monkeypatch), Computer()
    not_ready_until(computer, clock, deadline + 1)
    state = drive(monkeypatch, tmp_path, computer, save_plan(), readiness_timeout=deadline)
    assert state.outcome == "blocked" and "did not become ready" in state.failure and not computer.actions
    assert clock.now == pytest.approx(deadline) and max(clock.sleeps) <= 0.05
    assert json.loads((tmp_path / "run.json").read_text())["failure_stage"] == "readiness"


def test_the_last_read_is_made_at_the_deadline_so_a_page_ready_just_before_it_is_used(monkeypatch, tmp_path):
    clock, computer = FakeClock(monkeypatch), Computer()
    not_ready_until(computer, clock, 1.97)
    state = drive(monkeypatch, tmp_path, computer, save_plan(), readiness_timeout=2.0)
    assert state.answer and state.answer.achieved and [a.kind for a in computer.actions] == ["click"]


def test_tab_and_url_effects_do_not_wait_for_the_source_document(monkeypatch, tmp_path):
    clock, computer = FakeClock(monkeypatch), Computer()
    not_ready_until(computer, clock, 100)
    step = Milestone("open", "Open the site", "tab_created", value="https://example.net")
    state = drive(monkeypatch, tmp_path, computer, Reasoner([response(step)]), readiness_timeout=2.0)
    assert state.answer and state.answer.achieved and clock.now < 1.0


# -- invalid first planner step ("new tab" as a URL), recorded from a live task ----------------------------------


def test_the_recorded_invalid_first_plan_gets_one_bounded_correction_and_then_stops(monkeypatch, tmp_path):
    computer, writer = Computer(), Reasoner([RECORDED_PLAN])
    state = drive(monkeypatch, tmp_path, computer, writer, Jev("plan"))
    assert state.outcome == "blocked" and "Invalid milestone URL" in state.failure
    assert [r.role for r in writer.requests] == ["planner", "planner"]  # the plan, and its one correction
    assert not computer.actions and not state.history
    correction = json.loads(writer.requests[1].text)["plan_correction"]
    assert correction["rejected_plan"] == RECORDED_PLAN and "Invalid milestone URL" in correction["validation_error"]
    assert json.loads((tmp_path / "run.json").read_text())["recoveries"] == 0


def test_cancelling_during_the_correction_dispatches_nothing(monkeypatch, tmp_path):
    computer, control = Computer(), RunControl("cancel-correction")
    corrected = response(Milestone("tab", "Open tab", "tab_created", value="https://example.net"))
    writer = Reasoner([RECORDED_PLAN, corrected])
    generate = writer.generate

    def cancel_while_correcting(request, token=None):
        if len(writer.requests) == 1:
            control.cancel()
        return generate(request, token)

    writer.generate = cancel_while_correcting
    state = drive(monkeypatch, tmp_path, computer, writer, control=control)
    assert state.outcome == "aborted" and len(writer.requests) == 2 and not computer.actions
    assert state.cancel_reason == "task cancelled" and not state.uncertain  # nothing was in flight


# -- YouTube playback reported failed although it had started ---------------------------------------------------


def test_playback_that_starts_after_six_seconds_under_a_changed_canonical_url_is_one_click(monkeypatch, tmp_path):
    clock = FakeClock(monkeypatch)
    listed = "https://video.example.test/watch?v=abc123&pp=tracking-token"
    canonical = "https://video.example.test/watch?v=abc123"  # the page drops the tracking parameter on load
    computer = Computer()
    computer.state.capabilities.add("media_state")
    computer.state.elements["result"] = Element("result", "First recording", "link", href=listed)

    def clicked(machine, action):
        assert action.kind == "click" and action.target == "result"
        machine.state.url = canonical
        machine.state.canonical_url = canonical
        machine.state.tabs[machine.state.active_tab] = canonical
        machine.state.owner = "watch-document"
        machine.state.media = {"video": Media("video", "First recording", True, False, 1, 0)}
        return ""

    def observe(machine):
        if machine.actions and clock.now >= 6.0:
            machine.state.media = {"video": Media("video", "First recording", False, False, 4, 0.4)}

    def select(state, options):
        return next(k for k, text in options.items() if k not in {"replan", "key"} and json.loads(text)["target"] == "result")

    computer.on_execute, computer.on_inspect = clicked, observe
    writer = Reasoner([response(Milestone("play", "Open and play the first recording", "media_playing"))])
    state = drive(
        monkeypatch,
        tmp_path,
        computer,
        writer,
        Jev("plan", selection=select),
        goal="Open and play the first recording",
        readiness_timeout=1.5,
    )
    assert [a.kind for a in computer.actions] == ["click"]  # one click: never a second try while the video loads
    assert state.answer and state.answer.achieved, state.failure
    assert 6.0 <= clock.now < 7.0  # it waited for the video, and stopped waiting as it played
    report = json.loads((tmp_path / "run.json").read_text())
    assert report["recoveries"] == 0 and report["phases"]["media_verification"]["calls"] == 1


# -- a request taken as the answer to a pending clarification ----------------------------------------------------


NEW_REQUEST = "Actually, open https://another-request.test instead"


def clarifying_jev():
    """A classifier that asks until it has a clarification, and then says the task is pure reasoning."""
    jev = Jev("clarify")
    classify = jev.system_one

    def system_one(state, questions):
        jev.route = "reason" if state.get("clarifications") else "clarify"
        return classify(state, questions)

    jev.system_one = system_one
    return jev


def question_events(events):
    return [e for e in events if e.kind == "question"]


def test_a_new_request_that_arrives_while_a_question_is_pending_is_not_appended_to_it(monkeypatch, tmp_path):
    from execution_world import FakeSupervisor

    events = []
    control = RunControl("first-task", events.append)

    def ask(question):
        # The user speaks a new request instead of answering. The session cancels this task, and the text still
        # reaches the pending reply slot (the late-reply race).
        control.cancel("superseded by a new request")
        return NEW_REQUEST

    jev, writer, computer, supervisor = clarifying_jev(), Reasoner([]), Computer(), FakeSupervisor()
    state = drive(
        monkeypatch, tmp_path, computer, writer, jev, control=control, goal="Help me with this", ask=ask, supervisor=supervisor
    )
    assert state.outcome == "aborted" and state.cancel_reason == "interrupted"
    assert len(question_events(events)) == 1 and not computer.actions and computer.reads == 0
    assert supervisor.goal is None and not writer.requests
    assert all(NEW_REQUEST not in json.dumps(request) for request in jev.requests)  # never reached a model
    assert len(jev.requests) == 1  # the scope was not decided again with it


def test_an_answered_question_is_kept_once_and_used_for_the_scope(monkeypatch, tmp_path):
    from execution_world import FakeSupervisor

    from glide.computer.writer import Answer

    events = []
    supervisor = FakeSupervisor([("answer", Answer("16", True))])
    jev = clarifying_jev()
    state = drive(
        monkeypatch,
        tmp_path,
        Computer(),
        Reasoner([]),
        jev,
        control=RunControl("task", events.append),
        goal="Help me with this",
        ask=lambda question: "  Subtract 24 from 40  ",
        supervisor=supervisor,
    )
    assert state.outcome == "done" and len(question_events(events)) == 1
    asked = [r for r in jev.requests if r.get("clarifications")]
    assert len(asked) == 1 and len(asked[0]["clarifications"]) == 1
    assert asked[0]["clarifications"][0].endswith("User: Subtract 24 from 40")
    assert question_events(events)[0].question_id  # a reply is bound to its question


def test_a_declined_question_ends_the_task_without_a_guess(monkeypatch, tmp_path):
    events = []
    jev, computer = clarifying_jev(), Computer()
    state = drive(
        monkeypatch,
        tmp_path,
        computer,
        Reasoner([]),
        jev,
        control=RunControl("task", events.append),
        goal="Help me with this",
        ask=lambda question: "   ",
    )
    assert state.outcome == "blocked" and "Clarification was declined" in state.failure
    assert len(question_events(events)) == 1 and not computer.actions and len(jev.requests) == 1


def test_a_run_asks_at_most_three_questions(monkeypatch, tmp_path):
    events = []
    jev = Jev("clarify")  # never satisfied
    state = drive(
        monkeypatch,
        tmp_path,
        Computer(),
        Reasoner([]),
        jev,
        control=RunControl("task", events.append),
        goal="Help me with this",
        ask=lambda question: "still unclear",
    )
    assert state.outcome == "blocked" and len(question_events(events)) == 3


def test_nobody_to_ask_is_a_blocked_run_that_states_the_question(monkeypatch, tmp_path):
    state = drive(monkeypatch, tmp_path, Computer(), Reasoner([]), Jev("clarify"), goal="Help me with this")
    assert state.outcome == "blocked" and state.failure.startswith("What outcome should this task reach?")


def test_a_scenario_with_several_effects_reports_each_verified_effect(monkeypatch, tmp_path):
    state = drive(monkeypatch, tmp_path, Computer(), Reasoner([response(*SCENARIOS["navigate to X and scroll"])]), Jev("plan"))
    assert state.answer.achieved and [p["verified"] for p in state.progress] == [1, 1]

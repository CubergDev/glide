"""The run under a RunControl: stops, unknown writes, unavailable desktops and what a run leaves behind (D2, D3).

Recorded failures this covers: a dead browser endpoint ends the run with an actionable message and zero actions,
and the task result still names the goal (so the user can retry) while the stored run folder does not; a stop during a correction (a re-decision) dispatches nothing; an interrupted write is read
back once and never replayed; stopping mid-typing releases the key and types no further character.
"""

import json
import threading
from types import SimpleNamespace

import pytest
from world import FakeTypeSafe, FakeWriter, Page, World, scripted

from glide.assistant.tasks import STOPPED_BY_USER, ComputerTask, abort_on
from glide.computer import macos, runner, windows
from glide.computer.actions import Context
from glide.computer.config import DEFAULT_READINESS_TIMEOUT
from glide.computer.control import RunControl, checkpoint, controlled, dispatch
from glide.computer.models import Abort, BrowserConnectionError, DesktopPermissionError
from glide.computer.platform_adapter import desktop, dispatching
from glide.computer.runner import RunConfig, RunState


def one_page() -> World:
    return World([Page(name="search", items=["Search"], field="Search", url="https://example.test")])


def run_with(monkeypatch, tmp_path, policy, control, *, record_content=False, writer=None, goal="Find invoice 007381"):
    world = one_page()
    world.install(monkeypatch)
    classifier = FakeTypeSafe(policy)
    monkeypatch.setattr(runner, "TypeSafeClient", lambda: classifier)
    cfg = RunConfig(goal, tmp_path, act=True, delay=0, record_content=record_content)
    state = runner.run(
        cfg,
        lambda client, history: Context(goal, "Google Chrome", None, client, writer or FakeWriter(), history),
        control=control,
    )
    return world, state


def test_stop_during_classification_never_dispatches(monkeypatch, tmp_path):
    control = RunControl("task")

    def policy(*args):
        control.cancel("stopped by the user")
        return "type_text", None

    world, state = run_with(monkeypatch, tmp_path, policy, control)
    assert state.outcome == "aborted (stopped by the user)" and not state.uncertain and not world.typed


def test_a_stop_during_a_re_decision_after_a_writer_focus_dispatches_nothing_more(monkeypatch, tmp_path):
    control = RunControl("task")
    calls = []

    def policy(state, questions):
        calls.append(state)
        if len(calls) == 1:
            return ("done", None)
        control.cancel("stopped by the user")  # the classifier was sent back with a focus and is being asked again
        return ("type_text", None)

    writer = FakeWriter(reviews=[{"focus": "look at the search box", "achieved": False}])
    world, state = run_with(monkeypatch, tmp_path, policy, control, writer=writer)
    assert len(calls) == 2  # the focus really did send the classifier back
    assert state.handoffs and state.handoffs[0].focus == "look at the search box"
    assert state.outcome == "aborted (stopped by the user)" and not state.uncertain
    assert world.log == [] and not world.typed  # nothing was dispatched, before or after the focus


def test_interrupted_dispatch_gets_one_readback_and_is_never_replayed(monkeypatch, tmp_path):
    control = RunControl("task")
    performed = []
    world = one_page()
    world.install(monkeypatch)

    def interrupted(*args):
        def mutation():
            performed.append("typed")
            world.typed["Search"] = "007"
            control.cancel()
            checkpoint()

        return dispatch(mutation)

    monkeypatch.setattr(runner, "perform", interrupted)
    classifier = FakeTypeSafe(scripted(("type_text", None), ("type_text", None)))
    monkeypatch.setattr(runner, "TypeSafeClient", lambda: classifier)
    state = runner.run(
        RunConfig("Find", tmp_path, act=True, delay=0),
        lambda client, history: Context("Find", "Google Chrome", None, client, FakeWriter(), history),
        control=control,
    )
    assert state.uncertain and state.readback == "captured; completion unknown"
    assert performed == ["typed"] and world.ticks == 2  # one action; one capture before it and one read back after
    assert state.answer is None and state.outcome.startswith("aborted")


@pytest.mark.parametrize("in_flight", [False, True])
def test_capture_permission_loss_blocks_without_retry_or_claiming_success(monkeypatch, tmp_path, in_flight):
    attempts, events = [], []

    def unavailable(*a, **kw):
        attempts.append("capture")
        raise DesktopPermissionError("Screen Recording access is missing for Terminal.")

    world = one_page()
    world.install(monkeypatch)
    monkeypatch.setattr(runner, "capture", unavailable)  # after install, which patches the capture too
    classifier = FakeTypeSafe(scripted(("done", None)))
    control = RunControl("task", events.append)
    control.in_flight = in_flight
    state = runner.run(
        RunConfig("Open browser", tmp_path, act=True, delay=0),
        lambda client, history: Context("Open browser", "Google Chrome", None, classifier, FakeWriter(), history),
        classifier_factory=lambda: classifier,
        control=control,
    )
    assert attempts == ["capture"] and not world.typed and not state.handoffs
    assert state.outcome == "desktop unavailable" and state.answer is None
    assert state.uncertain == in_flight
    assert events[-1].kind == "blocked" and "Terminal" in events[-1].text


def test_a_dead_browser_endpoint_stops_with_an_actionable_message_zero_actions_and_the_goal_kept(monkeypatch, tmp_path):
    events, writer = [], FakeWriter()
    goal = "Search Google for HKU"
    refused = BrowserConnectionError("cdp", "http://127.0.0.1:9222", ConnectionRefusedError(61, "refused"))

    def capture(*a, **kw):
        raise refused

    world = one_page()
    world.install(monkeypatch)
    monkeypatch.setattr(runner, "capture", capture)
    performed = []
    monkeypatch.setattr(runner, "perform", lambda *a: performed.append(a))
    state = runner.run(
        RunConfig(goal, tmp_path, act=True, delay=0),
        lambda client, history: Context(goal, "Google Chrome", None, client, writer, history),
        classifier_factory=lambda: FakeTypeSafe(scripted(("done", None))),
        control=RunControl("task", events.append),
    )
    assert performed == [] and not world.log and not writer.requests and state.history == []
    assert state.outcome == "desktop unavailable"
    assert "Start or reconnect the selected debugging browser" in state.failure
    assert events[-1].kind == "blocked" and events[-1].text == state.failure
    stored = json.loads((tmp_path / "run.json").read_text())
    assert stored["failure"] and "HKU" not in json.dumps(stored)  # storage keeps the failure, not the goal (D3)

    # the user-facing task keeps the goal in memory, says what to do, and does not call this a crash
    monkeypatch.setattr(desktop, "accessibility_trusted", lambda: True)
    config = SimpleNamespace(
        writer=lambda: writer, classifier=lambda: FakeTypeSafe(scripted(("done", None))), record_content=False
    )
    task = ComputerTask(goal, act=True, config=config, folder=tmp_path / "task")
    result = task._execute()
    assert result.goal == goal and result.outcome == "desktop unavailable" and goal in result.summary()
    assert "debugging browser" in result.failure
    assert "debugging browser" in result.spoken("en") and "went wrong" not in result.spoken("en")


def test_a_stop_before_the_first_step_dispatches_nothing_and_says_why(monkeypatch, tmp_path):
    control = RunControl("task")
    control.cancel(STOPPED_BY_USER)
    events = []
    control.emit = events.append
    world, state = run_with(monkeypatch, tmp_path, scripted(("click_item", "Search")), control)
    assert state.outcome == f"aborted ({STOPPED_BY_USER})" and not world.log
    assert [e.kind for e in events][-1] == "cancelled"


# -- what a run leaves behind (D3) -----------------------------------------------------------------


def test_default_recording_is_off_and_the_folder_holds_only_counts_and_a_scrubbed_failure(monkeypatch, tmp_path):
    from glide.providers.errors import ProviderError

    def failing(*args):
        raise ProviderError(
            "openai answered 401: bad key for https://api.example.test/v1/x?key=sk-abcdef0123456789",
            kind="auth",
            provider="openai",
        )

    _, state = run_with(monkeypatch, tmp_path, failing, None)
    assert state.outcome == "provider failure"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["run.json"]
    stored = (tmp_path / "run.json").read_text()
    summary = json.loads(stored)
    assert summary["outcome"] == "provider failure" and summary["failure"]
    for private in ("Find invoice", "007381", "/v1/x", "sk-abcdef", "https://example.test"):
        assert private not in stored
    assert {"steps_taken", "calls", "usage", "timing", "task_id"} <= set(summary)
    assert "goal" not in summary and "history" not in summary and "answer" not in summary


def test_opting_in_writes_the_run_log_and_step_files(monkeypatch, tmp_path):
    run_with(monkeypatch, tmp_path, scripted(("done", None)), None, record_content=True)
    names = {p.name for p in tmp_path.iterdir()}
    assert {"run.log", "run.json"} <= names and any(n.startswith("step-") for n in names)
    assert json.loads((tmp_path / "run.json").read_text())["goal"] == "Find invoice 007381"


def test_a_dry_run_keeps_its_first_move_in_memory_only(monkeypatch, tmp_path):
    world = World([Page(name="home", items=["Tickets", "Help"], url="https://example.test")])
    world.install(monkeypatch)
    classifier = FakeTypeSafe(scripted(("click_item", "Tickets")))
    monkeypatch.setattr(runner, "TypeSafeClient", lambda: classifier)
    events = []
    state = runner.run(
        RunConfig("Open tickets", tmp_path, act=False, delay=0),
        lambda client, history: Context("Open tickets", "Google Chrome", None, client, FakeWriter(), history),
        control=RunControl("task", events.append),
    )
    assert state.outcome == "dry run" and state.would_do == "click 'Tickets'" and not world.log
    assert events[-1].kind == "dry_run" and events[-1].text == "click 'Tickets'"
    assert "Tickets" not in (tmp_path / "run.json").read_text()


def test_runner_prints_nothing_of_its_own(monkeypatch, tmp_path, capsys):
    run_with(monkeypatch, tmp_path, scripted(("done", None)), None)
    assert capsys.readouterr().out == ""


# -- engines and configuration ------------------------------------------------------------------------


def test_the_default_engine_is_the_legacy_loop_and_an_unknown_engine_is_refused(tmp_path):
    cfg = RunConfig("goal", tmp_path)
    assert cfg.engine == "legacy" and cfg.record_content is False and cfg.readiness_timeout == DEFAULT_READINESS_TIMEOUT
    with pytest.raises(ValueError, match="Unknown execution engine"):
        runner.run(RunConfig("goal", tmp_path, engine="nope"), lambda *a: None)


def test_a_replay_runs_the_legacy_loop_even_when_structured_is_asked_for(monkeypatch, tmp_path):
    seen = []

    def fake_run(cfg, ctx_factory, classifier_factory):
        seen.append(cfg.engine)
        return RunState(outcome="done")

    monkeypatch.setattr(runner, "_run", fake_run)
    runner.run(RunConfig("goal", tmp_path, engine="structured", image=tmp_path / "x.png"), lambda *a: None)
    runner.run(RunConfig("goal", tmp_path, engine="legacy"), lambda *a: None)
    assert seen == ["legacy", "legacy"]  # run() never asked for the structured engine on a replay


# -- how the stop hook and the control compose --------------------------------------------------------


def test_abort_on_and_the_control_give_the_same_reason_whichever_fires_first(monkeypatch):
    monkeypatch.setattr(desktop, "check_abort", lambda: None)
    stop, control = threading.Event(), RunControl("t")
    with abort_on(stop, control):
        desktop.check_abort()
        control.cancel(STOPPED_BY_USER)
        with pytest.raises(Abort, match=STOPPED_BY_USER):
            desktop.check_abort()
    stop2, control2 = threading.Event(), RunControl("t2")
    with abort_on(stop2, control2):
        stop2.set()
        with pytest.raises(Abort, match=STOPPED_BY_USER):
            desktop.check_abort()


def test_computer_task_stop_cancels_the_control_and_the_event(tmp_path):
    task = ComputerTask("goal", act=False, config=SimpleNamespace(), folder=tmp_path)
    task.stop()
    assert task.stop_requested and task.control.cancelled.is_set() and task.control.reason == STOPPED_BY_USER
    with pytest.raises(Abort, match=STOPPED_BY_USER):
        task.control.check()


# -- platform input stops cleanly -----------------------------------------------------------------------


@pytest.mark.parametrize("adapter", [macos, windows], ids=["mac", "windows-simulated"])
def test_stop_during_typing_releases_key_but_types_no_next_character(monkeypatch, adapter):
    control = RunControl("task")
    sent = []
    if adapter is macos:
        monkeypatch.setattr(macos, "_unicode_key", lambda ch, down: (ch, down))

        def post(event):
            sent.append(event)
            if event[1]:
                control.cancel()

        monkeypatch.setattr(macos, "_post", post)
        expected = [("a", True), ("a", False)]
    else:
        monkeypatch.setattr(windows, "_key", lambda vk=0, scan=0, flags=0: (scan, flags))

        def send(event):
            sent.append(event)
            if not event[1] & windows.KEYEVENTF_KEYUP:
                control.cancel()

        monkeypatch.setattr(windows, "_send", send)
        expected = [(ord("a"), windows.KEYEVENTF_UNICODE), (ord("a"), windows.KEYEVENTF_UNICODE | windows.KEYEVENTF_KEYUP)]
    with controlled(control), dispatching(), pytest.raises(Abort):
        adapter.type_text("abc")
    assert sent == expected


@pytest.mark.parametrize("adapter", [macos, windows], ids=["mac", "windows"])
def test_a_cancelled_wait_exits_without_sleeping(monkeypatch, adapter):
    control = RunControl("task")
    control.cancel()
    monkeypatch.setattr(adapter.time, "sleep", lambda _: pytest.fail("slept after cancellation"))
    with controlled(control), pytest.raises(Abort):
        adapter.sleep_watching(20)


# -- the command line --------------------------------------------------------------------------------------


@pytest.mark.parametrize(("flags", "recorded"), [([], False), (["--record-content"], True)])
def test_the_command_line_records_content_only_when_asked(monkeypatch, tmp_path, flags, recorded):
    from glide.computer import cli

    seen = {}
    monkeypatch.setattr(cli, "make_writer", lambda config: None)
    monkeypatch.setattr(cli.config, "writer_vision", lambda: True)

    def fake_run(cfg, ctx_factory, **kwargs):
        seen.update(cfg=cfg, control=kwargs.get("control"))
        return SimpleNamespace(outcome="done", failure="")

    monkeypatch.setattr(cli, "run", fake_run)
    cli.main(["a goal", "--out", str(tmp_path), *flags], SimpleNamespace(classifier=lambda: None))
    assert seen["cfg"].record_content is recorded and isinstance(seen["control"], RunControl)

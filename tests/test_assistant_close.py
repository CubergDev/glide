"""Closing the assistant waits, a bounded time, for a stopped task to unwind (finding 3).

Every exit path of the command line ends in `Assistant.close()`; the task thread is a daemon, so returning while it was
still unwinding let the process end in the middle of the write the stop was waiting out. The screen-driving loop is
replaced by one that unwinds when the test lets it; the bound is measured on the assistant's injected clock.
"""

from __future__ import annotations

import threading

from test_assistant_core import build
from test_assistant_fakes import WAIT, FakeClassifier, route_json
from test_cancel_assistant import Model, in_thread, llm_of

from glide.assistant.core import Assistant
from glide.computer import runner
from glide.computer.platform_adapter import desktop
from glide.computer.runner import RunState

ROUTE = route_json("computer", reply="On it.", goal="Send the invoice")


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def late_unwinding(monkeypatch):
    """A run that sees the stop at once but needs `release` before it reports how it ended, as a write in flight does."""
    run = type("Run", (), {})()
    run.started, run.release = threading.Event(), threading.Event()

    def fake_run(cfg, ctx_factory, classifier_factory=None, control=None):
        run.started.set()
        control.cancelled.wait(WAIT)
        run.release.wait(WAIT)
        return RunState(outcome="aborted (stopped by the user)", uncertain=True, readback="captured; completion unknown")

    monkeypatch.setattr(runner, "run", fake_run)
    monkeypatch.setattr(desktop, "accessibility_trusted", lambda: True)  # an acting task asks for the permission first
    return run


def rig_with_task(tmp_path, run, **kw):
    rig = build(tmp_path, llm=llm_of(Model(chats=[ROUTE])), writer=object(), classifier=FakeClassifier(None))
    rig.assistant = Assistant(rig.config, io=rig.assistant.io, runs_dir=tmp_path / "runs", **kw)
    task = rig.assistant.handle_text("send the invoice", act=True, wait=False).task
    assert run.started.wait(WAIT)
    return rig, task


def test_close_waits_for_the_stopped_task_to_report_how_it_unwound(tmp_path, monkeypatch):
    run = late_unwinding(monkeypatch)
    rig, task = rig_with_task(tmp_path, run)
    seen = {}

    def close():
        seen["closed"] = rig.assistant.close()
        seen["result_at_return"] = task.result

    done, _ = in_thread(close)
    assert not done.wait(0.2), "close returned while the task was still unwinding"
    assert not rig.player.closed  # nothing was shut under it

    run.release.set()  # the write ends and the run reports it

    assert done.wait(WAIT)
    assert seen["closed"] is True and seen["result_at_return"] is not None and seen["result_at_return"].uncertain
    assert rig.player.closed and rig.warned == []


def test_close_says_so_when_the_task_does_not_unwind_in_time(tmp_path, monkeypatch):
    run, clock = late_unwinding(monkeypatch), Clock()
    rig, task = rig_with_task(tmp_path, run, clock=clock, close_wait_s=2.0)
    seen = {}
    done, _ = in_thread(lambda: seen.setdefault("closed", rig.assistant.close()))
    assert task.control.cancelled.wait(WAIT)  # close has taken its deadline and stopped the task

    clock.now = 2.5  # the time it may wait is up, and the task has not unwound

    assert done.wait(WAIT) and seen["closed"] is False
    assert any("still stopping" in line and "check the screen" in line for line in rig.warned)
    assert rig.player.closed  # the assistant still closes: the process is going
    run.release.set()
    assert task.wait(WAIT)


def test_close_with_no_task_neither_waits_nor_warns(tmp_path):
    rig = build(tmp_path)
    assert rig.assistant.close() is True
    assert rig.warned == [] and rig.player.closed


def test_close_with_a_finished_task_neither_waits_nor_warns(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "run", lambda *a, **k: RunState(outcome="done"))
    monkeypatch.setattr(desktop, "accessibility_trusted", lambda: True)
    rig = build(tmp_path, llm=llm_of(Model(chats=[ROUTE])), writer=object(), classifier=FakeClassifier(None))
    rig.assistant.handle_text("send the invoice", act=True)
    assert rig.assistant.close() is True and rig.warned == []


def test_close_waits_for_a_task_that_started_between_its_first_look_and_the_stop(tmp_path):
    """audit2 finding 1: a task started concurrently with close() was cancelled by stop() but not waited for."""
    rig = build(tmp_path)
    clock = Clock()
    assistant = Assistant(rig.config, io=rig.assistant.io, runs_dir=tmp_path / "runs", clock=clock, close_wait_s=2.0)

    class Never:
        running = True

        def wait(self, timeout=None):
            clock.now += 1.0  # each poll is a second of the assistant's clock
            return False

    class Runner:
        """No task is visible until `stop()` has run: the one that a request thread started just before it."""

        def __init__(self) -> None:
            self.stopped = False

        @property
        def current(self):
            return Never() if self.stopped else None

        running = False

        def stop(self):
            self.stopped = True
            return True

    assistant._tasks = Runner()
    assert assistant.close() is False
    assert any("still stopping" in line for line in rig.warned)


def test_no_task_starts_after_close_has_begun(tmp_path, monkeypatch):
    started = []
    monkeypatch.setattr(runner, "run", lambda *a, **k: started.append(1) or RunState(outcome="done"))
    monkeypatch.setattr(desktop, "accessibility_trusted", lambda: True)
    rig = build(tmp_path, llm=llm_of(Model(chats=[ROUTE])), writer=object(), classifier=FakeClassifier(None))
    assert rig.assistant.close() is True
    reply = rig.assistant.handle_text("send the invoice", act=True)
    assert reply.task is None and started == []

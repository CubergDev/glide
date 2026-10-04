"""A task that was stopped, or failed, with a write of unknown outcome is reported as exactly that (finding 1).

The runner knows (`RunState.uncertain`, `RunState.readback`, run.json); the assistant used to copy neither into the
`TaskResult`, so a stop was spoken as a clean stop. Here the screen-driving loop is replaced by one that ends the way
the real loop does after a write whose effect it could not confirm, and what the person is shown and told is read back.
"""

from __future__ import annotations

import threading

import pytest
from test_assistant_core import build, spoken
from test_assistant_fakes import WAIT, FakeClassifier, route_json
from test_cancel_assistant import Model, llm_of

from glide.assistant.phrases import PHRASES, say
from glide.assistant.tasks import TaskResult
from glide.computer import runner
from glide.computer.platform_adapter import desktop
from glide.computer.runner import RunState

HONEST = "I stopped, and the last action may or may not have happened: check the screen."
ROUTE = route_json("computer", reply="On it.", goal="Send the invoice")


def rig_for(tmp_path, monkeypatch, fake_run):
    monkeypatch.setattr(runner, "run", fake_run)
    monkeypatch.setattr(desktop, "accessibility_trusted", lambda: True)  # an acting task asks for the permission first
    return build(tmp_path, llm=llm_of(Model(chats=[ROUTE])), writer=object(), classifier=FakeClassifier(None))


def test_the_result_carries_the_runner_s_uncertainty(tmp_path, monkeypatch):
    def fake_run(cfg, ctx_factory, classifier_factory=None, control=None):
        return RunState(outcome="aborted (stopped by the user)", uncertain=True, readback="captured; completion unknown")

    rig = rig_for(tmp_path, monkeypatch, fake_run)
    task = rig.assistant.handle_text("send the invoice", act=True, wait=False).task
    assert task.wait(WAIT)
    assert task.result.uncertain is True and task.result.readback == "captured; completion unknown"
    assert task.result.stopped is True


def test_a_stop_that_left_a_write_unknown_is_told_honestly_and_never_replayed(tmp_path, monkeypatch):
    """F1: stop in the middle of an action; the run ends uncertain; the person is shown and told that, once, in a fixed
    sentence that does not claim the task is done or cleanly stopped, and nothing is sent again."""
    calls, started = [], threading.Event()

    def fake_run(cfg, ctx_factory, classifier_factory=None, control=None):
        calls.append(cfg.goal)
        started.set()
        control.cancelled.wait(WAIT)
        return RunState(outcome=f"aborted ({control.reason})", uncertain=True, readback="captured; completion unknown")

    rig = rig_for(tmp_path, monkeypatch, fake_run)
    task = rig.assistant.handle_text("send the invoice", act=True, wait=False).task
    assert started.wait(WAIT)

    assert rig.assistant.stop() is True

    assert task.wait(WAIT) and rig.assistant.wait_idle(WAIT)
    assert any(HONEST in shown for shown in rig.shown)
    assert HONEST in spoken(rig)  # the stop cut the voice, and this was said after the cut
    assert "Stopped." not in rig.shown  # not the plain stop
    assert calls == ["Send the invoice"]  # never replayed
    note = rig.assistant._history[-1]["content"]
    assert "may or may not have happened" in note and "done" not in note.lower()


def test_an_uncertain_result_never_speaks_the_writer_s_answer(tmp_path, monkeypatch):
    """The writer's words about the screen are not a claim that the last action happened."""
    from glide.computer.writer import Answer

    def fake_run(cfg, ctx_factory, classifier_factory=None, control=None):
        return RunState(
            outcome="desktop unavailable",
            failure="A desktop operation failed; completion could not be verified.",
            uncertain=True,
            readback="unavailable; completion unknown",
            answer=Answer("The invoice was sent.", True, spoken_text="The invoice was sent."),
        )

    rig = rig_for(tmp_path, monkeypatch, fake_run)
    task = rig.assistant.handle_text("send the invoice", act=True, wait=False).task
    assert task.wait(WAIT) and rig.assistant.wait_idle(WAIT)
    assert HONEST in spoken(rig) and not any("invoice was sent" in text for text in spoken(rig))
    assert not any("invoice was sent" in shown for shown in rig.shown if "task" in shown)
    assert task.result.uncertain and not task.result.stopped


def test_a_crash_after_a_write_began_is_uncertain_too(tmp_path, monkeypatch):
    def fake_run(cfg, ctx_factory, classifier_factory=None, control=None):
        control.in_flight = True  # a mutation was dispatched and its outcome was never observed
        raise RuntimeError("boom")

    rig = rig_for(tmp_path, monkeypatch, fake_run)
    task = rig.assistant.handle_text("send the invoice", act=True, wait=False).task
    assert task.wait(WAIT) and rig.assistant.wait_idle(WAIT)
    assert task.result.outcome == "crashed" and task.result.uncertain
    assert HONEST in spoken(rig)


def test_a_certain_result_is_not_marked(tmp_path, monkeypatch):
    def fake_run(cfg, ctx_factory, classifier_factory=None, control=None):
        return RunState(outcome="stalled")

    rig = rig_for(tmp_path, monkeypatch, fake_run)
    task = rig.assistant.handle_text("send the invoice", act=True, wait=False).task
    assert task.wait(WAIT) and rig.assistant.wait_idle(WAIT)
    assert task.result.uncertain is False and HONEST not in spoken(rig) and "stuck" in " ".join(spoken(rig))


@pytest.mark.parametrize("language", ["en", "yue", "zh", "zh-HK", None, "fr"])
def test_the_sentence_exists_in_every_language_and_overrides_the_outcome(language):
    result = TaskResult("g", True, "done", answer="All finished.", uncertain=True)
    assert result.spoken(language) == say("uncertain", language) and "All finished" not in result.spoken(language)
    assert TaskResult("g", True, "aborted (x)", stopped=True, uncertain=True).spoken(language) == say("uncertain", language)
    assert TaskResult("g", True, "aborted (x)", stopped=True).spoken(language) == ""


def test_the_english_sentence_is_the_fixed_one():
    assert PHRASES["uncertain"]["en"] == HONEST


@pytest.mark.parametrize(
    ("outcome", "key"),
    [
        ("provider failure", "provider"),
        ("generation unavailable", "provider"),
        ("desktop unavailable", "desktop"),
        ("crashed", "crashed"),
    ],
)
def test_a_run_that_fails_after_an_earlier_answer_says_the_failure_not_the_stale_answer(outcome, key):
    # PR4-4175615764: the answer of an earlier stop of the same run survives in the run state when it later fails
    result = TaskResult("g", True, outcome, answer="old unachieved answer", failure="The provider said no.")
    assert result.spoken() == say(key) and "old unachieved" not in result.spoken()
    assert TaskResult("g", True, "stalled", answer="what the screen shows").spoken() == "what the screen shows"

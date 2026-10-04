"""`IO.approve`: the assistant asks before ANY computer task starts, and only an explicit True starts one.

The hook is what lets a front end (the app server) hold the person's yes. Without one, the caller's own `act` choice stands, as the
terminal commands have it; those tests are in test_assistant_core.py.
"""

from __future__ import annotations

import pytest
from test_assistant_fakes import WAIT, FakeConfig, FakeLLM, FakePlayer, FakeTTS, route_json

from glide.assistant.core import IO, Assistant
from glide.assistant.tasks import ComputerTask
from glide.computer import runner

SECRET = "sk-SECRET-1234567890"


def forbid_runs(monkeypatch):
    def refuse(*args, **kwargs):
        pytest.fail("a computer task was started without an approval")

    monkeypatch.setattr(ComputerTask, "_execute", refuse)
    monkeypatch.setattr(runner, "run", refuse)


def computer_llm(goal: str = "read the title"):
    return FakeLLM(route=route_json("computer", reply="On it.", goal=goal, language="en"))


def assistant_with(approve, tmp_path, monkeypatch):
    forbid_runs(monkeypatch)
    config = FakeConfig(llm=computer_llm(), secret=SECRET)
    shown, warned = [], []
    return Assistant(config, io=IO(show=shown.append, warn=warned.append, approve=approve), runs_dir=tmp_path / "runs"), warned


@pytest.mark.parametrize(
    "approver", [lambda goal, act: False, lambda goal, act: None, lambda goal, act: "yes", lambda goal, act: 1]
)
def test_only_true_starts_a_task(approver, tmp_path, monkeypatch):
    assistant, warned = assistant_with(approver, tmp_path, monkeypatch)
    reply = assistant.handle_text("read it", act=True)
    assert reply.task is None and reply.error and "not approved" in reply.error and warned == [reply.error]


def test_an_approver_that_raises_is_a_refusal(tmp_path, monkeypatch):
    def broken(goal, act):
        raise RuntimeError("boom")

    assistant, _ = assistant_with(broken, tmp_path, monkeypatch)
    reply = assistant.handle_text("read it")
    assert reply.task is None and "not approved" in reply.error


def test_the_approver_is_given_the_goal_and_whether_the_task_will_act(tmp_path, monkeypatch):
    seen = []
    assistant, _ = assistant_with(lambda goal, act: seen.append((goal, act)) or False, tmp_path, monkeypatch)
    assistant.handle_text("read it", act=True)
    assistant.handle_text("read it", act=False)
    assert seen == [("read the title", True), ("read the title", False)]


def test_a_refused_task_is_not_acknowledged_aloud(tmp_path, monkeypatch):
    player = FakePlayer()
    forbid_runs(monkeypatch)
    tts = FakeTTS()
    config = FakeConfig(llm=computer_llm(), tts=tts, secret=SECRET)
    assistant = Assistant(config, io=IO(player=player, approve=lambda goal, act: False), runs_dir=tmp_path / "runs")
    assistant.handle_text("read it")
    assert assistant.wait_idle(WAIT) and tts.calls == []  # no "On it." for something that will not happen

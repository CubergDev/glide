"""The engine's backend door is shut in tests unless a test opens it with a fake."""

import pytest
from execution_world import Computer, Jev, Reasoner, drive

from glide.computer.execution import engine


def test_the_real_backend_door_is_refused_by_default():
    with pytest.raises(RuntimeError, match="real browser or desktop backend"):
        engine.make_backend("Brave Browser")


def test_a_test_that_hands_the_engine_a_fake_gets_it(monkeypatch, tmp_path):
    computer = Computer()
    state = drive(monkeypatch, tmp_path, computer, Reasoner([]), Jev("navigate"), goal="Open https://different.test")
    assert state.outcome == "done" and [a.kind for a in computer.actions] == ["navigate"]


def test_an_engine_run_that_forgets_the_fake_fails_loudly_and_acts_on_nothing(tmp_path):
    from glide.computer.actions import Context
    from glide.computer.control import RunControl
    from glide.computer.runner import RunConfig, run

    cfg = RunConfig("Open https://different.test", tmp_path, act=True, engine="structured", readiness_timeout=0)
    writer = Reasoner([])
    state = run(
        cfg,
        lambda client, history: Context(cfg.goal, "Brave Browser", None, client, writer, history),
        classifier_factory=lambda: Jev("navigate"),
        control=RunControl("t"),
    )
    assert state.outcome == "blocked" and state.failure.startswith("Execution stopped (RuntimeError)")
    assert not state.history and not writer.requests

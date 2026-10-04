"""Keep the selected-link proof while media loads; never replay an unverified click."""

import json

import pytest
from execution_world import Computer, Jev, Reasoner, drive, response

from glide.computer.execution import progress
from glide.computer.execution.contracts import Element, Media, Milestone


@pytest.mark.parametrize("plays", [True, False])
def test_delayed_playback_keeps_original_action_proof_and_never_reclicks(monkeypatch, tmp_path, plays):
    clock = [0.0]
    monkeypatch.setattr(progress.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(progress.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    computer = Computer()
    computer.state.capabilities.add("media_state")
    destination = "https://media.example.test/watch/first"
    computer.state.elements["result"] = Element("result", "First recording", "link", href=destination)

    def clicked(machine, action):
        assert action.kind == "click" and action.target == "result"
        machine.state.url = destination
        machine.state.tabs[machine.state.active_tab] = destination
        machine.state.owner = "result-document"
        machine.state.media = {"video": Media("video", "First recording", True, False, 1, 0)}
        return ""

    def observe(machine):
        if plays and machine.actions and clock[0] >= 2.0:
            machine.state.media = {"video": Media("video", "First recording", False, False, 4, 0.3)}

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
        record_content=True,  # the event log (events.jsonl) is written only when content recording is on (D3)
    )
    assert [action.kind for action in computer.actions] == ["click"]
    assert len(writer.requests) == 1
    report = json.loads((tmp_path / "run.json").read_text())
    assert report["recoveries"] == 0
    if plays:
        assert state.answer.achieved and state.progress[0]["verified"] == 1
        assert state.answer.text == "Done. Your task is complete."
        assert 2 <= clock[0] < 3  # Early exit when playback starts, not a fixed ten-second sleep.
    else:
        assert state.answer is None and state.progress[0]["remaining"] == 1
        assert "Playback could not be verified" in state.failure
        assert report["failure_stage"] == "media_verification" and clock[0] == 10
        events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
        timeout = next(event["details"] for event in events if event["event"] == "media_verification_timed_out")
        assert timeout["media_state"][0]["paused"] and timeout["timeout_s"] == 10


def test_zero_wait_media_verification_does_not_sleep_or_replay(monkeypatch, tmp_path):
    def refuse_sleep(_):
        pytest.fail("Explicit zero wait was ignored")

    monkeypatch.setattr(progress.time, "sleep", refuse_sleep)
    computer = Computer()
    computer.state.capabilities.add("media_state")
    computer.state.media = {"video": Media("video", "Recording", True, False, 1, 0)}
    writer = Reasoner([response(Milestone("play", "Play recording", "media_playing", target="video"))])
    state = drive(monkeypatch, tmp_path, computer, writer, readiness_timeout=0)
    assert state.answer is None and "Playback could not be verified" in state.failure
    assert len(computer.actions) == 1 and len(writer.requests) == 1

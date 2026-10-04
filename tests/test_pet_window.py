"""The pet window on Qt's offscreen platform, over a fake core. It is never shown, never given a window handle."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6.QtWidgets")

from pet_fakes import FakeCore, qt_application, window_factory
from PySide6.QtCore import Qt
from test_assistant_fakes import FakeLLM
from test_pet_core import PetConfig, make_core

from glide.ui import pet
from glide.ui.core import MAX_TEXT, PetEvent


@pytest.fixture
def qt_app():
    return qt_application()


@pytest.fixture
def pet_window(qt_app):
    yield from window_factory(qt_app)


def test_the_pet_starts_passive_with_actions_and_recording_off_and_no_indicator(pet_window):
    core = FakeCore()
    window = pet_window(core)
    assert core.calls == []  # nothing started: no microphone, request or task
    assert not window.act.isChecked() and not window.record.isChecked() and not window.headset.isChecked()
    assert window.badges.isHidden() and window.act_banner.isHidden() and window.record_banner.isHidden()
    assert window.status.text() == "Idle · microphone off" and window.pet.mood == "idle"


def test_turning_recording_on_shows_the_indicator_in_the_window_and_in_settings_and_reaches_the_core(pet_window, tmp_path):
    core, config = make_core(tmp_path, llm=FakeLLM())
    window = pet_window(core, config)
    assert config.record_content is False and window.badges.isHidden()
    window.record.setChecked(True)
    assert config.record_content is True  # the core reads this when a task starts
    assert not window.badges.isHidden() and "REC" in window.badges.text() and not window.record_banner.isHidden()
    window.poll()  # the core's own `recording` event arrives: the state stays shown
    assert window.record.isChecked() and "REC" in window.badges.text()
    window.record.setChecked(False)
    assert config.record_content is False and window.badges.isHidden() and window.record_banner.isHidden()
    core.close()


def test_recording_started_from_the_command_line_is_shown_from_the_first_frame(pet_window, tmp_path):
    core, config = make_core(tmp_path, llm=FakeLLM(), record_content=True)
    window = pet_window(core, config)
    assert window.record.isChecked() and "REC" in window.badges.text() and not window.record_banner.isHidden()
    core.close()


def test_allowing_computer_actions_is_visible_and_reaches_the_core(pet_window):
    core = FakeCore()
    window = pet_window(core)
    window.act.setChecked(True)
    assert core.act is True and window.badges.text() == "ACT" and not window.act_banner.isHidden()
    window.act.setChecked(False)
    assert core.act is False and window.badges.isHidden()


def test_both_indicators_show_together(pet_window):
    core = FakeCore()
    window = pet_window(core)
    window.act.setChecked(True)
    core.queue.append(PetEvent("recording", {"on": True}))
    window.poll()
    assert window.badges.text() == "● REC  ACT"


def test_settings_that_a_running_session_depends_on_are_locked_until_it_ends(pet_window):
    core = FakeCore()
    window = pet_window(core)
    core.busy = True
    window.refresh()
    assert not window.act.isEnabled() and not window.headset.isEnabled() and not window.record.isEnabled()
    assert window.stop_action.isEnabled()
    core.busy = False
    window.refresh()
    assert window.act.isEnabled() and window.record.isEnabled() and not window.stop_action.isEnabled()


def test_what_the_person_does_is_forwarded_to_the_core_and_nothing_else(pet_window):
    core = FakeCore()
    window = pet_window(core)
    window.open_chat()
    window.input.setText("  hello  ")
    window.submit()
    window.start_voice()
    window.finish_voice()
    window.start_voice()
    window.cancel_voice()
    assert core.calls == [("text", "hello"), "start_voice", "pause_voice", "start_voice", "stop"]
    assert window.input.maxLength() == MAX_TEXT


def test_events_move_the_raccoon_and_the_activity_is_plain_text_that_is_never_interpreted(pet_window):
    core = FakeCore()
    window = pet_window(core)
    markup = "<b>bold</b><img src='file:///etc/passwd'>"
    for event in (
        PetEvent("transcript", {"role": "assistant", "text": markup, "partial": False}),
        PetEvent("switch", {"role": "llm.fast", "from_slot": "a", "to_slot": "b", "kind": "timeout", "reason": markup}),
        PetEvent("state", {"assistant": "thinking"}),
    ):
        core.queue.append(event)
    window.poll()
    assert window.pet.mood == "thinking" and window.status.text() == "Working on it"
    assert markup in window.log.toPlainText() and not window.log.acceptRichText()
    assert window.status.textFormat() == Qt.TextFormat.PlainText
    assert "fallback: llm.fast a -> b (timeout:" in window.log.toPlainText()  # every fallback is on screen


def test_the_core_closing_the_microphone_ends_the_voice_bar(pet_window):
    from glide.ui.raccoon.widget import COMPACT, VOICE

    core = FakeCore()
    window = pet_window(core)
    window.start_voice()
    assert window.mode == VOICE
    core.queue.append(PetEvent("mic", {"open": False, "detail": "unavailable"}))
    core.queue.append(PetEvent("notice", {"message": "Voice input could not start"}))
    window.poll()
    assert window.mode == COMPACT and "could not start" in window.status.text()


def test_stop_ends_the_voice_session_and_closing_the_window_closes_the_core(pet_window):
    from glide.ui.raccoon.widget import COMPACT

    core = FakeCore()
    window = pet_window(core)
    window.start_voice()
    window.stop()
    assert core.calls[-1] == "stop" and window.mode == COMPACT
    window.close()
    assert core.calls[-1] == "close"


def test_the_activity_view_keeps_no_more_than_the_bound(pet_window):
    core = FakeCore()
    window = pet_window(core)
    for i in range(300):
        core.queue.append(PetEvent("notice", {"message": f"line {i}"}))
    window.poll()
    assert len(window.log.toPlainText().splitlines()) == 80


def test_the_pet_leaves_no_file_behind(pet_window, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    core = FakeCore()
    window = pet_window(core)
    window.act.setChecked(True)
    window.record.setChecked(True)
    window.show_settings()
    window.close()
    assert [p for p in tmp_path.rglob("*") if p.is_file()] == []


# -- the entry point, and the refusals that keep tests off the real machine ----------------------------


def test_the_real_window_and_event_loop_are_refused_in_tests():
    from glide.ui import pet_ui

    with pytest.raises(RuntimeError, match="real machine"):
        pet_ui.show_pet(FakeCore(), PetConfig())
    with pytest.raises(RuntimeError, match="real machine"):
        pet.open_window(FakeCore(), PetConfig())


def test_the_real_voice_hardware_is_refused_in_tests():
    from glide.ui import core

    with pytest.raises(RuntimeError, match="real machine"):
        core._open_voice(PetConfig(), None, io=None, act=False, assistant_factory=None)


def test_qt_runs_offscreen_in_every_test():
    import os

    assert os.environ["QT_QPA_PLATFORM"] == "offscreen"


@pytest.mark.parametrize("args", [["--silence-ms", "100"], ["--silence-ms", "5000"], ["--workspace", "/nonexistent-folder-xyz"]])
def test_bad_arguments_are_refused_before_anything_is_built(args, monkeypatch):
    monkeypatch.setattr(pet, "open_window", lambda *a: pytest.fail("no window for bad arguments"))
    with pytest.raises(SystemExit) as error:
        pet.main(args)
    assert error.value.code == 2


def test_main_builds_a_core_that_records_nothing_unless_asked(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("GLIDE_CONFIG", raising=False)
    opened = []
    monkeypatch.setattr(pet, "open_window", lambda core, config: (opened.append((core, config)), 0)[1])
    assert pet.main(["--workspace", str(tmp_path), "--silence-ms", "900"]) == 0
    core = opened[0][0]
    assert core.record_content is False and core.act is False and core.silence_ms == 900
    assert core._runs_dir == tmp_path.resolve() / "runs"
    core.close()
    assert pet.main(["--workspace", str(tmp_path), "--record-content"]) == 0
    assert opened[1][0].record_content is True
    opened[1][0].close()
    assert [p for p in tmp_path.rglob("*") if p.is_file()] == []

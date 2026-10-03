"""The raccoon widget on Qt's offscreen platform: nothing is shown on screen and no window handle is requested."""

import pytest

pytest.importorskip("PySide6.QtWidgets")

from glide.ui.raccoon import sprites


@pytest.fixture
def window(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    from glide.ui.raccoon.widget import PetWindow

    w = PetWindow()
    yield w
    w.deleteLater()
    app.processEvents()


def test_every_mood_frame_draws_inside_its_canvas():
    for mood, poses in sprites.moods().items():
        assert poses, mood
        for pose in poses:
            im = sprites.draw(pose)
            assert im.size == (sprites.GRID_W, sprites.GRID_H)
            box = im.getchannel("A").getbbox()
            # nothing clipped at the edges (the outline would be missing there)
            assert box[0] > 0 and box[1] > 0 and box[2] < sprites.GRID_W and box[3] < sprites.GRID_H, (mood, box)


def test_chat_submit_emits_trimmed_text_and_returns_to_compact(window):
    from glide.ui.raccoon.widget import CHAT, COMPACT

    sent = []
    window.text_submitted.connect(sent.append)
    window.open_chat()
    assert window.mode == CHAT and window.pet.mood == "typing" and not window.send_button.isEnabled()
    window.submit()
    assert sent == [] and window.mode == CHAT
    window.input.setText("  open my calendar ")
    assert window.send_button.isEnabled()
    window.submit()
    assert sent == ["open my calendar"] and window.mode == COMPACT and window.input.text() == ""
    assert window.pet.mood == "idle"


def test_voice_finish_and_cancel(window):
    from glide.ui.raccoon.widget import COMPACT, VOICE

    events = []
    window.voice_started.connect(lambda: events.append("start"))
    window.voice_finished.connect(lambda s: events.append(("done", s >= 0)))
    window.voice_cancelled.connect(lambda: events.append("cancel"))
    window.start_voice()
    assert window.mode == VOICE and window.pet.mood == "listening" and window.wave.timer.isActive()
    window.push_level(3.0)
    assert window.wave.levels[-1] == 1.0
    window.finish_voice()
    assert window.mode == COMPACT and not window.wave.timer.isActive()
    window.start_voice()
    window.close_bar()  # Escape while recording cancels
    assert events == ["start", ("done", True), "start", "cancel"]


def test_rest_mood_and_unknown_mood(window):
    window.set_mood("sleeping", rest=True)
    window.open_chat()
    window.close_bar()
    assert window.pet.mood == "sleeping"
    with pytest.raises(ValueError):
        window.set_mood("dancing")


def test_raccoon_stays_visible_when_macos_app_becomes_inactive(window):
    from PySide6.QtCore import Qt

    assert window.testAttribute(Qt.WidgetAttribute.WA_MacAlwaysShowToolWindow)

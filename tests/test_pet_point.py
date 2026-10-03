"""Point and ask in the pet window: the controls, the answer card, the pin on screen. Offscreen Qt, a fake core."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("PySide6.QtWidgets")

from pet_fakes import FakeCore, qt_application, window_factory
from PySide6.QtCore import Qt

from glide.ui import pet, pet_ui, point_ui
from glide.ui.core import PetEvent

SHOW_MARKER = point_ui.show_marker  # read at collection, before the guard replaces it for every test


@pytest.fixture
def qt_app():
    return qt_application()


@pytest.fixture
def pet_window(qt_app):
    yield from window_factory(qt_app)


@pytest.fixture
def pins(monkeypatch):
    """The two calls that would draw on the real screen, recorded."""
    shown, dismissed = [], []
    monkeypatch.setattr(pet, "show_point_marker", lambda point: (shown.append(point), f"marker{len(shown)}")[1])
    monkeypatch.setattr(pet, "dismiss_point_marker", dismissed.append)
    return SimpleNamespace(shown=shown, dismissed=dismissed)


def point_window(pet_window, core=None):
    window = pet_window(core or FakeCore())
    window.activity.setCurrentIndex(pet_ui.POINT)
    return window


def event(kind, **data):
    return PetEvent("point", {"kind": kind, **data})


def test_choosing_point_and_ask_shows_its_controls_and_shares_nothing_by_default(pet_window):
    window = point_window(pet_window)
    assert window.share.isVisibleTo(window.settings) and not window.act.isVisibleTo(window.settings)
    assert not window.share.isChecked() and not window.include_image.isChecked()
    assert not window.include_image.isEnabled() and not window.voice_button.isEnabled()
    assert window.ask_button.text() == "Read this point" and window.input.maxLength() == 2048
    window.share.setChecked(True)
    assert window.ask_button.text() == "Ask this point" and window.voice_button.isEnabled()
    window.activity.setCurrentIndex(pet_ui.CONTROL)
    assert window.act.isVisibleTo(window.settings) and not window.share.isVisibleTo(window.settings)
    assert window.voice_button.isEnabled() and window.input.maxLength() == 8192


def test_the_ask_button_pins_with_the_default_question_unless_one_is_typed(pet_window):
    core = FakeCore()
    window = point_window(pet_window, core)
    window.ask_button.click()
    window.question.setText("What does this button do?")
    window.ask_button.click()
    assert core.calls == [
        ("point_start", pet_ui.DEFAULT_QUESTION, {"share": False, "with_image": False}),
        ("point_start", "What does this button do?", {"share": False, "with_image": False}),
    ]


def test_an_image_crop_is_asked_for_only_when_sharing_and_allowed(pet_window, monkeypatch):
    core = FakeCore()
    window = point_window(pet_window, core)
    window.include_image.setChecked(True)  # while not sharing: ignored
    window.ask_button.click()
    assert core.calls[-1][2] == {"share": False, "with_image": False}
    window.share.setChecked(True)
    monkeypatch.setattr(pet_ui, "_image_allowed", lambda: False)  # a text-only answer model
    window.refresh()
    assert not window.include_image.isEnabled()
    window.ask_button.click()
    assert core.calls[-1][2] == {"share": True, "with_image": False}
    monkeypatch.setattr(pet_ui, "_image_allowed", lambda: True)
    window.refresh()
    assert window.include_image.isEnabled()
    window.ask_button.click()
    assert core.calls[-1][2] == {"share": True, "with_image": True}


def test_a_malformed_vision_setting_is_not_a_yes(monkeypatch):
    monkeypatch.setattr(pet_ui, "writer_vision", lambda: (_ for _ in ()).throw(ValueError("bad setting")))
    assert pet_ui._image_allowed() is False


def test_the_bars_text_box_asks_the_pin_when_one_is_held_and_pins_otherwise(pet_window):
    core = FakeCore()
    window = point_window(pet_window, core)
    window.open_chat()
    window.input.setText("first")
    window.submit()
    core.point.holding = True
    window.open_chat()
    window.input.setText("and the second one?")
    window.submit()
    assert core.calls == [("point_start", "first", {"share": False, "with_image": False}), ("point_ask", "and the second one?")]
    window.activity.setCurrentIndex(pet_ui.CONTROL)
    window.open_chat()
    window.input.setText("open the calendar")
    window.submit()
    assert core.calls[-1] == ("text", "open the calendar")  # back in control mode, text goes to Glide


def test_voice_in_point_mode_needs_sharing_and_resumes_a_session_that_is_already_listening(pet_window):
    core = FakeCore()
    window = point_window(pet_window, core)
    window.start_voice()
    assert core.calls == [] and "Share selected context" in window.status.text()  # refused before anything starts
    window.share.setChecked(True)
    window.start_voice()
    assert core.calls == [("point_start", "", {"share": True, "with_image": False, "voice": True})]
    window.finish_voice()
    core.voice_active = True  # the microphone was paused, not closed: pressing voice again resumes it
    window.start_voice()
    assert core.calls[-2:] == ["pause_voice", "start_voice"]


def test_the_pin_is_drawn_once_replaced_when_it_moves_and_removed_when_the_pin_ends(pet_window, pins):
    core = FakeCore()
    window = point_window(pet_window, core)
    core.queue += [event("selected", point=[200, 150], text="Retry")]
    window.poll()
    window.refresh()
    assert pins.shown == [[200, 150]] and pins.dismissed == []
    core.queue += [event("selected", point=[900, 700], text="Cancel")]
    window.poll()
    assert pins.shown == [[200, 150], [900, 700]] and pins.dismissed == ["marker1"]
    core.queue += [event("closed")]
    window.poll()
    assert pins.dismissed == ["marker1", "marker2"] and window.marker is None


def test_a_retryable_error_keeps_the_pin_and_a_final_one_removes_it(pet_window, pins):
    core = FakeCore()
    window = point_window(pet_window, core)
    core.queue += [event("selected", point=[1, 2], text="x"), event("error", text="Please try again.", closed=False)]
    window.poll()
    assert pins.dismissed == [] and window.answer.text() == "Please try again."
    core.queue += [event("error", text="The pinned item expired. Select the point again.", closed=True)]
    window.poll()
    assert pins.dismissed == ["marker1"]


def test_closing_the_window_removes_the_pin(pet_window, pins):
    core = FakeCore()
    window = point_window(pet_window, core)
    core.queue += [event("selected", point=[1, 2], text="x")]
    window.poll()
    window.close()
    assert pins.dismissed == ["marker1"]


def test_a_pinned_text_and_an_answer_are_plain_text_in_a_bounded_card_that_says_how_old_it_is(pet_window, pins):
    core = FakeCore()
    window = point_window(pet_window, core)
    core.queue += [
        event("selected", point=[1, 2], text="<b>error 0007</b>"),
        event(
            "answer", text="<a href='https://example.com'>Meaning</a> " * 300, uncertain=True, age_s=61.0, model="m", seconds=2.0
        ),
    ]
    window.poll()
    for label in (window.target, window.answer, window.status, window.answer_note, window.point_provider):
        assert label.textFormat() == Qt.TextFormat.PlainText
    assert window.target.text() == "Pinned: <b>error 0007</b>" and "More context" in window.status.text()
    assert len(window.answer.text()) == 4096 and window.answer.text().startswith("<a href=")
    assert "61 s ago" in window.answer_note.text() and "may have changed" in window.answer_note.text()
    window.settings.layout().activate()
    assert window.answer_card.maximumHeight() == 200 and window.answer_card.widget() is window.answer
    assert window.settings.minimumSizeHint().height() < 850


def test_the_mode_cannot_be_switched_while_a_pin_is_being_taken(pet_window):
    core = FakeCore()
    window = point_window(pet_window, core)
    core.point.active = True
    window.refresh()
    assert not window.activity.isEnabled() and not window.share.isEnabled() and window.stop_action.isEnabled()
    core.point.active = False
    window.refresh()
    assert window.activity.isEnabled() and window.share.isEnabled() and not window.stop_action.isEnabled()


def test_the_pin_widget_is_transparent_to_input_and_never_takes_focus(qt_app):
    from glide.ui.point_ui import PointMarker

    marker = PointMarker()
    assert marker.windowFlags() & Qt.WindowType.WindowTransparentForInput
    assert marker.windowFlags() & Qt.WindowType.WindowDoesNotAcceptFocus
    assert marker.testAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
    marker.deleteLater()


@pytest.mark.parametrize(
    "platform,scale,point,expected", [("win32", 2, [400, 300], (184, 134)), ("darwin", 2, [200, 150], (184, 134))]
)
def test_the_pin_converts_coordinates_without_touching_the_pointer(qt_app, monkeypatch, platform, scale, point, expected):
    moves = []
    screen = SimpleNamespace(
        geometry=lambda: SimpleNamespace(x=lambda: 0, y=lambda: 0, width=lambda: 800, height=lambda: 600),
        devicePixelRatio=lambda: scale,
    )
    monkeypatch.setattr(point_ui, "QApplication", SimpleNamespace(primaryScreen=lambda: screen))
    monkeypatch.setattr(point_ui.sys, "platform", platform)
    monkeypatch.setattr(
        point_ui, "PointMarker", lambda: SimpleNamespace(move=lambda *xy: moves.append(xy), show=lambda: moves.append("shown"))
    )
    assert SHOW_MARKER(point) is not None and moves == [expected, "shown"]
    assert SHOW_MARKER([4000, 3000]) is None
    assert SHOW_MARKER([float("nan"), 0]) is None


def test_the_real_pin_and_its_removal_are_refused_in_tests():
    for call, args in ((pet.show_point_marker, ([1, 2],)), (pet.dismiss_point_marker, (None,))):
        with pytest.raises(RuntimeError, match="real machine"):
            call(*args)

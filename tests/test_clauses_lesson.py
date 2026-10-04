"""The lesson stepper: control words in, which step to present out. No model, no screen."""

from __future__ import annotations

import pytest

from glide.clauses import LessonStepper, parse_control

STEPS = ["one", "two", "three"]


@pytest.mark.parametrize(
    ("text", "want"),
    [
        ("next", "next"),
        ("okay next", "next"),
        ("Okay, next step please!", "next"),
        ("continue", "next"),
        ("go on", "next"),
        ("nex", "next"),
        ("back", "back"),
        ("go back", "back"),
        ("previous step", "back"),
        ("ok go back please", "back"),
        ("repeat", "repeat"),
        ("say that again", "repeat"),
        ("again please", "repeat"),
        ("one more time", "repeat"),
        ("dismiss", "dismiss"),
        ("I'm done", "dismiss"),
        ("end lesson", "dismiss"),
        ("got it", "dismiss"),
    ],
)
def test_control_words(text, want):
    assert parse_control(text) == want


@pytest.mark.parametrize("text", ["", "open notes", "next to the printer", "what is the back button", "stop the music", "text"])
def test_other_speech_is_not_a_control(text):
    assert parse_control(text) is None


def test_no_lesson_is_a_plain_result():
    s = LessonStepper()
    assert not s.active
    r = s.handle("next")
    assert r.kind == "no_lesson" and r.text == "" and r.index is None
    assert s.handle("open notes") is None  # not ours: the caller routes it


def test_walk_forward_back_repeat():
    s = LessonStepper()
    first = s.start(STEPS)
    assert (first.kind, first.index, first.text, first.total) == ("step", 0, "one", 3)
    assert s.handle("okay next").text == "two"
    assert s.handle("repeat").text == "two"
    assert s.handle("back").text == "one"
    assert s.handle("go back").index == 0  # clamps at the first step
    assert s.handle("next").text == "two" and s.handle("next").text == "three"


def test_past_the_last_ends_the_lesson():
    s = LessonStepper()
    s.start(["a", "b"])
    s.handle("next")
    r = s.handle("next")
    assert r.kind == "done" and r.text == "" and not s.active
    assert s.handle("next").kind == "no_lesson"


def test_dismiss_ends_the_lesson():
    s = LessonStepper()
    s.start(STEPS)
    assert s.handle("dismiss").kind == "dismissed" and not s.active


def test_start_replaces_a_lesson_and_current_reports_it():
    s = LessonStepper()
    assert s.current().kind == "no_lesson"
    s.start(STEPS)
    s.handle("next")
    s.start(["x", "y"])
    assert s.current().text == "x" and s.current().total == 2


@pytest.mark.parametrize("steps", [[], ["only"], list("abcdefg"), ["a", " "], ["a", 3]])
def test_a_lesson_has_two_to_six_real_steps(steps):
    with pytest.raises(ValueError):
        LessonStepper().start(steps)


def test_a_rejected_start_keeps_the_running_lesson():
    s = LessonStepper()
    s.start(STEPS)
    with pytest.raises(ValueError):
        s.start([])
    assert s.active and s.current().text == "one"


def test_the_caller_cannot_change_the_lesson_afterwards():
    steps = ["a", "b"]
    s = LessonStepper()
    s.start(steps)
    steps[0] = "changed"
    assert s.current().text == "a"

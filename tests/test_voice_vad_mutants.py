"""Mutation-testing follow-up for `TurnDetector` in `glide/speech/vad.py` (see docs/MUTATION_REPORT.md).

One utterance has one end, decided by silence: these tests pin the edges of that rule (the exact probabilities that start
a turn and count as silence, the exact silence setting, the utterance bound) and that nothing of one turn leaks into the
next (pre-roll, frame count, silence count), which is what a second, early or double commit would look like.
"""

from __future__ import annotations

import pytest
from guards_voice import no_real_audio  # noqa: F401

from glide.speech.vad import FRAME_BYTES, START_PROBABILITY, STOP_PROBABILITY, TurnDetector, UtteranceTooLong


def frame(tag: int) -> bytes:
    return bytes([tag]) * FRAME_BYTES


def test_the_silence_setting_is_accepted_from_200_to_2000_ms_inclusive():
    for ms in (200, 2000):
        assert TurnDetector(ms).silence_frames > 0
    for ms in (199, 2001):
        with pytest.raises(ValueError, match="200-2000"):
            TurnDetector(ms)


def test_a_turn_starts_at_exactly_the_start_probability_and_not_below_it():
    detector = TurnDetector()
    below = detector.feed(frame(1), START_PROBABILITY - 0.0001)
    assert below == (False, b"", False)
    started, pcm, _ = detector.feed(frame(2), START_PROBABILITY)
    assert started and pcm == frame(1) + frame(2)  # the pre-roll, then the frame that started it


def test_a_frame_exactly_at_the_stop_probability_is_not_silence():
    detector = TurnDetector(200)
    detector.feed(frame(1), 1.0)
    for _ in range(detector.silence_frames * 2):
        _, _, commit = detector.feed(frame(2), STOP_PROBABILITY)
        assert not commit
    for index in range(detector.silence_frames):
        _, _, commit = detector.feed(frame(3), STOP_PROBABILITY - 0.0001)
        assert commit == (index == detector.silence_frames - 1)


def test_a_turn_may_be_exactly_as_long_as_the_bound_and_not_one_frame_longer():
    detector = TurnDetector(max_utterance_s=0.032 * 5 + 0.001)
    assert detector.max_frames == 5
    detector.feed(frame(1), 1.0)
    for _ in range(4):
        detector.feed(frame(2), 1.0)
    with pytest.raises(UtteranceTooLong):
        detector.feed(frame(3), 1.0)
    assert not detector.active  # it ended: nothing of it is submitted


def test_a_turn_never_inherits_the_pre_roll_the_frames_or_the_silence_of_the_one_before():
    detector = TurnDetector(200)
    for tag in (10, 11, 12):
        detector.feed(frame(tag), 0.0)
    _, first, _ = detector.feed(frame(13), 1.0)
    assert first == frame(10) + frame(11) + frame(12) + frame(13)
    commit = False
    while not commit:  # the silence ends it
        _, _, commit = detector.feed(frame(14), 0.0)
    detector.feed(frame(20), 0.0)
    started, second, _ = detector.feed(frame(21), 1.0)
    assert started and second == frame(20) + frame(21)  # none of the first turn's frames come back
    assert detector.silent == 0 and detector.frames == 1  # and it is counted from its own start


def test_a_second_turn_does_not_end_on_its_first_frame_because_the_last_one_ended_in_silence():
    detector = TurnDetector(200)
    detector.feed(frame(1), 1.0)
    commit = False
    while not commit:
        _, _, commit = detector.feed(frame(2), 0.0)
    started, _, commit = detector.feed(frame(3), 1.0)
    assert started and not commit


def test_reset_forgets_the_turn_the_pre_roll_and_the_counts():
    detector = TurnDetector(200)
    detector.feed(frame(1), 0.0)
    detector.feed(frame(2), 1.0)
    detector.feed(frame(3), 0.0)
    detector.reset()
    assert (detector.active, detector.silent, detector.frames, len(detector.pre_roll)) == (False, 0, 0, 0)
    assert detector.finish() is False  # nothing is running: a late pause ends nothing and commits nothing
    detector.feed(frame(5), 0.0)
    detector.reset()  # frames heard before a turn are forgotten too
    started, pcm, _ = detector.feed(frame(4), 1.0)
    assert started and pcm == frame(4)  # no pre-roll survived the reset


def test_finish_ends_a_running_turn_once():
    detector = TurnDetector()
    detector.feed(frame(1), 1.0)
    assert detector.finish() is True
    assert detector.finish() is False

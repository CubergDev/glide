"""The executor's own progress counters are never voiced. A recorded failure: "Verified 0 effect(s); 2 remain" was
read aloud as if it were the answer to a research request."""

from __future__ import annotations

import pytest
from guards_voice import no_real_audio  # noqa: F401
from test_assistant_fakes import FakePlayer, FakeTTS, wait_until

from glide.assistant.speech import Speaker, is_internal_status


@pytest.mark.parametrize(
    "text",
    [
        "Verified 0 effect(s); 2 remain.",
        "verified 3 effects",
        "2 effects remaining",
        "0 effect(s) verified",
        "Checked the effect(s).",
    ],
)
def test_effect_counters_are_internal_status(text):
    assert is_internal_status(text)


@pytest.mark.parametrize(
    "text",
    [
        "There are 3 items left in your cart.",
        "I verified the page loaded.",
        "The side effects of this drug include nausea.",
        "Step 2 of the recipe takes 10 minutes.",
        "You have 2 messages remaining.",
        # PR6-4175247840: a counter inside a longer sentence is part of an answer
        "The study verified 2 effects of the treatment.",
        "There are 3 effects pending review.",
        "I verified 2 effects of the change, and 1 remains for you to check.",
    ],
)
def test_ordinary_sentences_with_numbers_are_still_spoken(text):
    assert not is_internal_status(text)


def test_the_speaker_does_not_voice_a_counter_and_reports_nothing_queued():
    tts, player = FakeTTS(chunks=1), FakePlayer()
    speaker = Speaker(tts, player)
    assert speaker.say("Verified 0 effect(s); 2 remain.") is False
    assert speaker.say("The museum opens at nine.") is True
    assert speaker.wait_idle(3)
    assert wait_until(lambda: tts.calls == [("The museum opens at nine.", None)])
    speaker.close()

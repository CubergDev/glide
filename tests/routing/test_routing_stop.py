"""The stop matcher moved to `glide.routing.stop` unchanged. The old module and the new one must agree on every case.

`STOP_CASES` and `NOT_STOP_CASES` are the cases of `tests/test_assistant_router.py` (copied, so this file survives that
file being moved when the router shim is applied), plus the configured extra phrases. Each case runs through
`glide.routing.stop` and `glide.assistant.router`, which must return the same answer; once `assistant/router.py` is a
shim that re-exports this module, the parity test is trivially true and can be deleted with the old module.
"""

# ruff: noqa: RUF001  the tests are about full-width punctuation and letters

from __future__ import annotations

import pytest

from glide.assistant import router as old
from glide.routing import stop as new

STOP_CASES = [
    "stop",
    "Stop!",
    "STOP.",
    "  stop  ",
    "stop it",
    "stop stop stop",
    "please stop",
    "stop please",
    "cancel",
    "cancel that",
    "never mind",
    "Never mind.",
    "nevermind",
    "enough",
    "be quiet",
    "stop talking",
    "Glide, stop",
    "hey glide stop please",
    "abort",
    "停",
    "停！",
    "停啦",
    "停止",
    "唔該停",
    "取消。",
    "收聲",
    "算了",
    "唔使喇",
    "别说了",
    "停停停",
    "ｓｔｏｐ",
]

NOT_STOP_CASES = [
    "stop the music",
    "cancel the meeting with Sam",
    "stop by the shop on the way",
    "don't stop believing",
    "what does stop mean",
    "open safari",
    "thank you",
    "好",
    "停車場喺邊度",
    "取消訂單",
    "",
    "   ",
    "...",
]


@pytest.mark.parametrize("text", STOP_CASES)
def test_a_stop_phrase_is_a_stop_in_both_places(text):
    assert new.is_stop(text) is True
    assert old.is_stop(text) is True


@pytest.mark.parametrize("text", NOT_STOP_CASES)
def test_anything_more_than_a_stop_phrase_is_not_a_stop_in_either_place(text):
    assert new.is_stop(text) is False
    assert old.is_stop(text) is False


@pytest.mark.parametrize("text", [*STOP_CASES, *NOT_STOP_CASES, "hold on please", "Hold ON, please!", "別吵"])
def test_the_two_matchers_never_disagree(text):
    extra_new = new.stop_phrases(["hold on please", "別吵"])
    extra_old = old.stop_phrases(["hold on please", "別吵"])
    assert extra_new == extra_old
    assert new.is_stop(text, extra_new) == old.is_stop(text, extra_old)
    assert new.normalize(text) == old.normalize(text)


def test_a_configured_phrase_must_be_the_whole_utterance():
    extra = new.stop_phrases(["hold on please"])
    assert new.is_stop("Hold on, please.", extra)
    assert not new.is_stop("hold on please open safari", extra)
    assert not new.is_stop("hold on please")  # not built in


def test_empty_configured_phrases_are_dropped():
    assert new.stop_phrases(["", "  ", "!!!"]) == frozenset()


def test_normalize_folds_case_width_and_punctuation():
    assert new.normalize("  Stop,   IT!! ") == "stop it"
    assert new.normalize("停！") == "停"
    assert new.normalize("ｓｔｏｐ") == "stop"

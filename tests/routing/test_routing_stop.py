"""The one stop matcher, `glide.routing.stop`. Everything that recognises "stop" imports it from there.

`STOP_CASES` and `NOT_STOP_CASES` are the cases of the old `tests/test_assistant_router.py`, kept when that file went with
the router it tested. `glide.assistant.router` is only a re-export shim for `glide/speech/` and must be the same objects.
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
def test_a_stop_phrase_is_a_stop(text):
    assert new.is_stop(text) is True


@pytest.mark.parametrize("text", NOT_STOP_CASES)
def test_anything_more_than_a_stop_phrase_is_not_a_stop(text):
    assert new.is_stop(text) is False


def test_the_old_module_is_only_a_re_export_of_the_one_list():
    assert (old.is_stop, old.normalize, old.stop_phrases) == (new.is_stop, new.normalize, new.stop_phrases)
    assert not hasattr(old, "STOP_CORE") and not hasattr(old, "fast_path")  # no second list can grow back there


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

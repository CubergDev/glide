"""The fixed sentences are found by language the way every other language-keyed table in the voice stack is."""

from __future__ import annotations

import pytest

from glide.assistant.phrases import say


@pytest.mark.parametrize(
    ("language", "text"),
    [
        (None, "Stopped."),
        ("en", "Stopped."),
        ("yue", "停咗。"),
        ("YUE", "停咗。"),
        ("zh", "已停止。"),
        ("zh-HK", "已停止。"),
        ("zh_TW", "已停止。"),
        ("fr", "Stopped."),
    ],
)
def test_a_sentence_is_found_by_the_language_then_the_part_before_the_dash_then_english(language, text):
    assert say("stopped", language) == text

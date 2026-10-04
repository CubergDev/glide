"""The sentence splitter, the speakable cleanup, and the speaker: speech starts after the first sentence."""

# ruff: noqa: RUF001  the tests are about full-width punctuation

from __future__ import annotations

import random
import threading

import pytest
from test_assistant_fakes import FakePlayer, FakeTTS, Log, wait_until

from glide.assistant.speech import (
    SentenceSplitter,
    Speaker,
    clean_for_speech,
    detect_language,
    has_content,
    split_sentences,
)

# -- the splitter -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hello there. How are you? I am fine!", ["Hello there.", "How are you?", "I am fine!"]),
        ("It costs 3.5 dollars. Cheap.", ["It costs 3.5 dollars.", "Cheap."]),
        ("Pi is 3.14159 and e is 2.718.", ["Pi is 3.14159 and e is 2.718."]),
        (
            "Dr. Smith met Mr. Jones at 5 p.m. on Monday. They talked.",
            ["Dr. Smith met Mr. Jones at 5 p.m. on Monday.", "They talked."],
        ),
        ("Use e.g. a pen, i.e. not a pencil. Then sign.", ["Use e.g. a pen, i.e. not a pencil.", "Then sign."]),
        ("J. K. Rowling wrote it. Yes.", ["J. K. Rowling wrote it.", "Yes."]),
        ("Visit example.com for details. Thanks.", ["Visit example.com for details.", "Thanks."]),
        ("So did I. Then we left.", ["So did I.", "Then we left."]),
        ("Wait... What?! Really.", ["Wait...", "What?!", "Really."]),
        ('He said "stop." Then he left.', ['He said "stop."', "Then he left."]),
        ("First line\nSecond line\n\nThird line", ["First line", "Second line", "Third line"]),
        ("1. Open the app. 2. Click save.", ["1. Open the app.", "2. Click save."]),
        ("No terminator at all", ["No terminator at all"]),
        ("", []),
        ("   ", []),
    ],
)
def test_english_sentences(text, expected):
    assert split_sentences(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("你好。我係Glide！你叫咩名呀？", ["你好。", "我係Glide！", "你叫咩名呀？"]),
        ("今日天氣好好。聽日會落雨；記得帶遮。", ["今日天氣好好。", "聽日會落雨；", "記得帶遮。"]),
        ("他说「你好。」然后走了。", ["他说「你好。」", "然后走了。"]),
        ("价格是3.5元。很便宜。", ["价格是3.5元。", "很便宜。"]),
        ("你好吗?我很好.谢谢", ["你好吗?", "我很好.", "谢谢"]),
        ("Hello。你好。Bye.", ["Hello。", "你好。", "Bye."]),
    ],
)
def test_chinese_sentences_end_at_the_mark_with_no_space_after_it(text, expected):
    assert split_sentences(text) == expected


def test_a_sentence_is_held_until_what_follows_its_period_is_known():
    splitter = SentenceSplitter()
    assert splitter.feed("It costs 3.") == []  # the next character may be a 5
    assert splitter.feed("5 dollars.") == []  # a period at the very end could still be "3.5 dollars.99"
    assert splitter.feed(" Next") == ["It costs 3.5 dollars."]
    assert splitter.flush() == ["Next"]


def test_a_period_before_a_lower_case_word_does_not_end_the_sentence():
    splitter = SentenceSplitter()
    assert splitter.feed("Call Dr. ") == []
    assert splitter.feed("Smith now. ") == []
    assert splitter.feed("Then wait") == ["Call Dr. Smith now."]


def test_an_ideographic_stop_ends_the_sentence_at_once():
    splitter = SentenceSplitter()
    assert splitter.feed("你好。") == ["你好。"]  # no waiting for the next character
    assert splitter.feed("我係") == []
    assert splitter.flush() == ["我係"]


def test_pieces_with_no_letter_or_digit_are_dropped():
    assert split_sentences('"') == []
    assert split_sentences("**") == []
    assert split_sentences("Yes.\n)\n**\nNo.") == ["Yes.", "No."]
    assert has_content("好") and has_content("3") and not has_content("...)")


@pytest.mark.parametrize(
    "text",
    [
        "Hello there. How are you? I am fine!",
        "Dr. Smith paid 3.5 dollars at 5 p.m. today. Then he left... Why?! Nobody knows.",
        "你好。我係Glide！你叫咩名呀？Hello. 1. First 2. Second",
        "Visit example.com or e.g. this. Fine.\nNext line\nAnd 3.14 more.",
    ],
)
def test_feeding_text_in_any_sized_pieces_gives_the_same_sentences(text):
    whole = split_sentences(text)
    rng = random.Random(7)
    for _ in range(40):
        splitter, out, i = SentenceSplitter(), [], 0
        while i < len(text):
            step = rng.randint(1, 6)
            out += splitter.feed(text[i : i + step])
            i += step
        out += splitter.flush()
        assert out == whole


def test_character_by_character_matches_too():
    text = "Hello there. Dr. Smith paid 3.5 dollars. 你好。Really?! Yes."
    splitter, out = SentenceSplitter(), []
    for ch in text:
        out += splitter.feed(ch)
    assert out + splitter.flush() == split_sentences(text)


def test_a_long_run_on_is_cut_at_a_comma_rather_than_waited_for():
    text = "word " * 12 + "and then, " + "more " * 20 + "until the end."
    splitter = SentenceSplitter(max_chars=100)
    first = splitter.feed(text[:150])
    assert first and first[0].endswith(",")
    assert 24 <= len(first[0]) < 100


def test_a_comma_inside_a_number_is_not_a_place_to_cut():
    splitter = SentenceSplitter(max_chars=40)
    assert splitter.feed("a" * 30 + " 1,000,000 and more words here") == []


# -- cleanup ----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "spoken"),
    [
        ("**Bold** and *italic* and `code`", "Bold and italic and code"),
        ("See [the docs](https://example.com/x) now", "See the docs now"),
        ("Go to https://example.com/page today", "Go to today"),
        ("## Title", "Title"),
        ("- an item", "an item"),
        ("1. Open the app.", "Open the app."),
        ("Great job 🎉 everyone", "Great job everyone"),
        ("a  lot   of   space", "a lot of space"),
    ],
)
def test_markdown_links_bullets_and_emoji_are_not_read_out(raw, spoken):
    assert clean_for_speech(raw) == spoken


@pytest.mark.parametrize(
    ("text", "code"),
    [("Hello there", "en"), ("你好嗎", "zh"), ("你喺邊度呀", "yue"), ("こんにちは", "ja"), ("안녕하세요", "ko"), ("", "en")],
)
def test_language_is_guessed_from_script(text, code):
    assert detect_language(text) == code


# -- the speaker ------------------------------------------------------------------------------------


def make_speaker(**tts_kw):
    log = Log()
    tts, player = FakeTTS(log=log, **tts_kw), FakePlayer()
    errors = []
    return Speaker(tts, player, on_error=errors.append), tts, player, log, errors


def test_sentences_are_spoken_in_order_and_reach_the_player_with_their_rate():
    speaker, tts, player, _, _ = make_speaker(chunks=2, rate=22050)
    speaker.say("First one.", language="en")
    speaker.say("Second one.", language="en")
    assert speaker.wait_idle(timeout=3)
    assert [t for t, _ in tts.calls] == ["First one.", "Second one."]
    assert player.played == [(b"\x01\x00\x02\x00", 22050)] * 4
    assert player.idle_waits == 1
    assert tts.closed == ["First one.", "Second one."]  # each sentence's stream was closed


def test_what_is_said_is_cleaned_first_and_nothing_empty_is_sent():
    speaker, tts, _, _, _ = make_speaker()
    assert speaker.say("**Hello**") is True
    assert speaker.say("**") is False
    assert speaker.say("   ") is False
    assert speaker.wait_idle(timeout=3)
    assert [t for t, _ in tts.calls] == ["Hello"]


def test_cancel_drops_what_is_queued_cuts_the_player_and_lets_later_speech_through():
    release = threading.Event()
    speaker, tts, player, _, _ = make_speaker(release=release)
    speaker.say("Being made.")
    speaker.say("Queued behind it.")
    assert wait_until(lambda: tts.calls)  # the first sentence is in the TTS, blocked
    speaker.cancel()
    assert player.cancels == 1
    release.set()
    assert wait_until(lambda: tts.closed)  # the abandoned request was closed when it came back
    assert [t for t, _ in tts.calls] == ["Being made."]  # the queued one never reached the TTS
    assert player.played == []  # and the abandoned one's audio was thrown away
    speaker.say("After the cut.")
    assert speaker.wait_idle(timeout=3)
    assert player.played  # speech after a cancel is heard in full


def test_speech_after_a_cancel_does_not_wait_for_the_abandoned_request():
    release = threading.Event()
    speaker, tts, player, _, _ = make_speaker(release=release)
    speaker.say("Stuck on the network.")
    assert wait_until(lambda: tts.calls)
    speaker.cancel()
    tts.release = None  # new requests answer at once; the old one is still blocked on `release`
    speaker.say("Fresh.")
    assert wait_until(lambda: player.played)  # heard while the abandoned request is still blocked
    release.set()


def test_only_if_is_checked_under_the_lock_cancel_takes():
    speaker, tts, _, _, _ = make_speaker()
    assert speaker.say("Allowed.", only_if=lambda: True)
    assert not speaker.say("Refused.", only_if=lambda: False)
    assert speaker.wait_idle(timeout=3)
    assert [t for t, _ in tts.calls] == ["Allowed."]


def test_a_failing_sentence_is_reported_and_the_next_is_still_spoken():
    speaker, tts, player, _, errors = make_speaker()
    tts.error = RuntimeError("voice down")
    speaker.say("Lost.")
    assert speaker.wait_idle(timeout=3)
    tts.error = None
    speaker.say("Heard.")
    assert speaker.wait_idle(timeout=3)
    assert [str(e) for e in errors] == ["voice down"]
    assert player.played


def test_wait_idle_times_out_while_a_sentence_is_still_being_made():
    release = threading.Event()
    speaker, tts, _, _, _ = make_speaker(release=release)
    speaker.say("Slow.")
    assert wait_until(lambda: tts.calls)
    assert speaker.wait_idle(timeout=0.05) is False
    release.set()
    assert speaker.wait_idle(timeout=3)


def test_first_audio_is_timed_from_mark():
    ticks = iter(range(100, 200))
    log = Log()
    speaker = Speaker(FakeTTS(log=log), FakePlayer(), clock=lambda: next(ticks))
    speaker.mark()
    assert speaker.first_audio_at is None
    speaker.say("Hi.")
    assert wait_until(lambda: speaker.first_audio_at is not None)


def test_nothing_is_said_after_close_and_the_tts_is_not_called():
    """audit2 finding 3: a task ending after close() spoke into a closed player and called the provider."""
    speaker, tts, _, _, _ = make_speaker()
    speaker.close()
    assert speaker.say("The task may have left a write half done.") is False
    assert tts.calls == [] and speaker._lane is None

"""The `[speech]` table, with the exact words of every refusal pinned.

A refusal is what a person reads when glide.toml is wrong, and the table is read in two places (the providers' view
in `glide.providers.config`, then the voice stack's `SpeechSettings.from_mapping`), so the words are part of the
contract: a rewrite of the validation must not change them without a test saying so.
"""

from __future__ import annotations

import pytest
from guards_voice import no_real_audio  # noqa: F401

from glide.providers.config import ConfigError, GlideConfig
from glide.speech.settings import SpeechSettings

KNOWN = (
    "barge_margin_db, barge_min_erle_db, barge_min_voiced_ms, echo_canceller, echo_tail_s, headset, idle_s, input_device, "
    "language, merge_window_s, output_device, output_rate, silence_ms, stop_phrases, vad, vad_model_path, vad_model_sha256, "
    "vad_model_url"
)
REFUSALS = [
    ({"headsett": True}, f"[speech] has unknown keys: headsett (known: {KNOWN})"),
    ({"headset": "yes"}, "[speech] headset must be true or false"),
    ({"headset": 1}, "[speech] headset must be true or false"),
    ({"silence_ms": 100}, "[speech] silence_ms must be 200-2000, not 100"),
    ({"silence_ms": 2001}, "[speech] silence_ms must be 200-2000, not 2001"),
    ({"silence_ms": 600.5}, "[speech] silence_ms must be a whole number"),
    ({"silence_ms": True}, "[speech] silence_ms must be a whole number"),
    ({"merge_window_s": 9}, "[speech] merge_window_s must be 0-5 seconds, not 9.0"),
    ({"merge_window_s": -0.1}, "[speech] merge_window_s must be 0-5 seconds, not -0.1"),
    ({"merge_window_s": "1"}, "[speech] merge_window_s must be a number"),
    ({"idle_s": -1}, "[speech] idle_s must not be negative, not -1.0"),
    ({"echo_tail_s": 3}, "[speech] echo_tail_s must be 0-2 seconds, not 3.0"),
    ({"output_rate": 12345}, "[speech] output_rate must be one of 16000, 22050, 24000, 44100, 48000, not 12345"),
    ({"output_rate": 24000.0}, "[speech] output_rate must be a whole number"),
    ({"vad": "magic"}, "[speech] vad must be one of auto, silero, energy, not 'magic'"),
    ({"vad": 1}, "[speech] vad must be text"),
    ({"vad": "silero"}, "[speech] vad = 'silero' needs vad_model_path and vad_model_sha256"),
    ({"vad": "silero", "vad_model_path": "p"}, "[speech] vad = 'silero' needs vad_model_path and vad_model_sha256"),
    ({"vad_model_url": "http://example.invalid/m"}, "[speech] vad_model_url must be an https:// address"),
    ({"vad_model_path": 3}, "[speech] vad_model_path must be text"),
    ({"echo_canceller": "magic"}, "[speech] echo_canceller must be one of auto, none, webrtc, nlms, not 'magic'"),
    ({"echo_canceller": 1}, "[speech] echo_canceller must be text"),
    ({"barge_min_voiced_ms": 50}, "[speech] barge_min_voiced_ms must be 96-1000, not 50"),
    ({"barge_min_voiced_ms": 2000}, "[speech] barge_min_voiced_ms must be 96-1000, not 2000"),
    ({"barge_margin_db": 1}, "[speech] barge_margin_db must be 3-30, not 1.0"),
    ({"barge_margin_db": 40}, "[speech] barge_margin_db must be 3-30, not 40.0"),
    ({"barge_min_erle_db": -1}, "[speech] barge_min_erle_db must be 0-30, not -1.0"),
    ({"barge_min_erle_db": 31}, "[speech] barge_min_erle_db must be 0-30, not 31.0"),
    ({"stop_phrases": "stop"}, "[speech] stop_phrases must be a list of text"),
    ({"stop_phrases": [1]}, "[speech] stop_phrases must be a list of text"),
    ({"stop_phrases": [""]}, "[speech] stop_phrases must be a list of non-empty phrases"),
    ({"stop_phrases": ["!!!"]}, "[speech] stop_phrases must be a list of non-empty phrases"),
]


@pytest.mark.parametrize(
    ("table", "message"), REFUSALS, ids=[next(iter(t)) + "=" + repr(next(iter(t.values()))) for t, _ in REFUSALS]
)
def test_every_refusal_says_exactly_what_it_said(table, message):
    with pytest.raises(ConfigError) as caught:
        SpeechSettings.from_mapping(table)
    assert str(caught.value) == message


def test_the_defaults_and_the_documented_bounds_are_the_ones_the_docs_state():
    s = SpeechSettings.from_mapping({})
    assert (s.silence_ms, s.echo_tail_s, s.output_rate, s.merge_window_s, s.idle_s) == (600, 0.3, 24000, 0.0, 0.0)
    assert (s.barge_min_voiced_ms, s.barge_margin_db, s.barge_min_erle_db) == (190, 8.0, 6.0)
    assert (s.vad, s.echo_canceller, s.headset, s.language, s.stop_phrases) == ("auto", "auto", False, None, ())
    edges = {
        "silence_ms": (200, 2000),
        "barge_min_voiced_ms": (96, 1000),
        "barge_margin_db": (3, 30),
        "barge_min_erle_db": (0, 30),
    }
    for key, (low, high) in edges.items():  # both ends are inside
        assert getattr(SpeechSettings.from_mapping({key: low}), key) == low
        assert getattr(SpeechSettings.from_mapping({key: high}), key) == high
    assert SpeechSettings.from_mapping({"merge_window_s": 5}).merge_window_s == 5.0
    assert SpeechSettings.from_mapping({"echo_tail_s": 2}).echo_tail_s == 2.0


def test_text_and_device_keys_take_what_the_stack_passes_on():
    s = SpeechSettings.from_mapping(
        {"language": "yue", "input_device": 3, "output_device": "Speakers", "vad_model_url": "https://m.example/x"}
    )
    assert (s.language, s.input_device, s.output_device, s.vad_model_url) == ("yue", 3, "Speakers", "https://m.example/x")


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("vad_model_url = 'http://x/y'", "[speech] vad_model_url must be an https URL"),
        ("silence_ms = 0", "[speech] silence_ms must be at least 1"),
        ("silence_ms = 100", "[speech] silence_ms must be 200-2000, not 100"),
        ("vad_model_sha256 = 'abc'", "[speech] vad_model_sha256 must be 64 hexadecimal digits"),
        ("headset = 'yes'", "[speech] headset must be true or false"),
        ("vad = 'silero'", "[speech] vad = 'silero' needs vad_model_path and vad_model_sha256"),
        ("idle_s = -1", "[speech] idle_s must not be negative, not -1.0"),
    ],
)
def test_through_the_config_file_the_person_reads_the_providers_words_first_then_the_voice_stacks(line, message):
    with pytest.raises(ConfigError) as caught:
        GlideConfig.from_toml("[speech]\n" + line, env={})
    assert str(caught.value) == message


def test_the_whole_table_reaches_the_voice_settings_through_the_config_file():
    cfg = GlideConfig.from_toml(
        '[speech]\nsilence_ms = 700\nheadset = true\nstop_phrases = ["hold on"]\nbarge_margin_db = 10', env={}
    )
    assert (cfg.voice.silence_ms, cfg.voice.headset, cfg.voice.stop_phrases, cfg.voice.barge_margin_db) == (
        700,
        True,
        ("hold on",),
        10.0,
    )
    assert (cfg.speech.silence_ms, cfg.speech.headset) == (700, True)

"""Settings validation, building the stack from settings, and the rule that no literal vendor value is in glide/speech."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest
from guards_voice import no_real_audio  # noqa: F401

sys.path.insert(0, str(Path(__file__).parent.parent))

from test_assistant_fakes import FakeConfig, FakeLLM, FakeTTS

import glide.speech
from glide.providers.config import ConfigError
from glide.speech.audio import FullDuplexDevice
from glide.speech.session import build_voice, make_vad
from glide.speech.settings import SpeechSettings
from glide.speech.vad import EnergyProbability, VadError

# -- settings ---------------------------------------------------------------------------------------


def test_an_empty_table_is_speaker_mode_with_loudness_detection():
    s = SpeechSettings.from_mapping(None)
    assert (s.headset, s.silence_ms, s.merge_window_s, s.idle_s, s.vad) == (False, 600, 0.0, 0.0, "auto")
    assert not s.silero_configured


def test_a_table_is_read_and_a_whole_number_is_accepted_where_seconds_are_expected():
    s = SpeechSettings.from_mapping({"headset": True, "silence_ms": 800, "merge_window_s": 1, "idle_s": 90})
    assert s.headset and s.silence_ms == 800 and s.merge_window_s == 1.0 and s.idle_s == 90.0


@pytest.mark.parametrize(
    "table",
    [
        {"headsett": True},
        {"headset": "yes"},
        {"silence_ms": 100},
        {"silence_ms": 600.5},
        {"merge_window_s": 9},
        {"idle_s": -1},
        {"vad": "magic"},
        {"vad": "silero"},
        {"vad_model_url": "http://example.invalid/m"},
        {"output_rate": 12345},
    ],
)
def test_a_bad_table_is_a_config_error_naming_the_key(table):
    with pytest.raises(ConfigError, match=r"\[speech\]"):
        SpeechSettings.from_mapping(table)


# -- building ---------------------------------------------------------------------------------------


class _Stream:
    def start(self): ...
    def stop(self): ...
    def close(self): ...


def fake_device(**kw):
    return FullDuplexDevice(input_factory=lambda cb: _Stream(), output_factory=lambda cb, rate: _Stream(), **kw)


def test_make_vad_is_loudness_unless_a_model_is_configured():
    assert isinstance(make_vad(SpeechSettings()), EnergyProbability)
    assert isinstance(make_vad(SpeechSettings(vad="energy")), EnergyProbability)


def test_a_silero_that_cannot_load_is_an_error_when_asked_for_and_a_visible_fallback_under_auto(tmp_path):
    missing = {"vad_model_path": str(tmp_path / "none.onnx"), "vad_model_sha256": "0" * 64}
    with pytest.raises(VadError):
        make_vad(SpeechSettings(vad="silero", **missing))
    warned = []
    assert isinstance(make_vad(SpeechSettings(vad="auto", **missing), warned.append), EnergyProbability)
    assert warned and "falls back" in warned[0]


def test_build_voice_makes_the_assistant_speak_through_the_device_and_wires_the_settings():
    device = fake_device()
    config = FakeConfig(llm=FakeLLM(), tts=FakeTTS())
    loop = build_voice(config, SpeechSettings(silence_ms=800, merge_window_s=1.0, idle_s=5.0), device=device, vad=lambda f: 0.0)
    assert loop.assistant.io.player is device
    assert loop._merge_frames == 32 and loop._idle_s == 5.0 and loop._detector.silence_frames == 25
    assert loop._act is False  # a dry run unless the caller says otherwise
    loop.assistant.close()


def test_build_voice_starts_a_device_it_made_and_closes_it_when_building_fails(monkeypatch):
    started = []
    monkeypatch.setattr(FullDuplexDevice, "start", lambda self: started.append(self))
    loop = build_voice(FakeConfig(), SpeechSettings(), vad=lambda f: 0.0)
    assert started == [loop.assistant.io.player]
    loop.assistant.close()

    closed = []
    monkeypatch.setattr(FullDuplexDevice, "close", lambda self: closed.append(self))

    def boom(*a, **k):
        raise VadError("no model")

    monkeypatch.setattr("glide.speech.session.make_vad", boom)
    with pytest.raises(VadError):
        build_voice(FakeConfig(), SpeechSettings())
    assert len(closed) == 1


# -- nothing hard-coded (D6) ------------------------------------------------------------------------

PACKAGE = Path(glide.speech.__file__).parent
FORBIDDEN = {
    "an address": re.compile(r"(?:https?|wss?)://[a-z0-9]", re.I),
    "a 64-digit hash": re.compile(r"\b[0-9a-f]{64}\b", re.I),
    "a vendor model or voice id": re.compile(r"\b(?:eleven|scribe|silero_vad|gpt|claude|deepseek)[-_.][\w.-]*\d", re.I),
}


@pytest.mark.parametrize("path", sorted(PACKAGE.glob("*.py")), ids=lambda p: p.name)
def test_the_speech_package_names_no_endpoint_hash_model_or_voice(path):
    text = path.read_text()
    for what, pattern in FORBIDDEN.items():
        assert not pattern.search(text), f"{path.name} contains {what}: {pattern.search(text).group(0)!r}"


# -- echo cancellation and barge-in wiring ----------------------------------------------------------


def test_build_voice_hands_the_barge_in_and_stop_settings_to_the_loop_and_the_assistant():
    settings = SpeechSettings(barge_min_voiced_ms=250, barge_margin_db=10, barge_min_erle_db=9, stop_phrases=("hold on please",))
    loop = build_voice(FakeConfig(llm=FakeLLM(), tts=FakeTTS()), settings, device=fake_device(), vad=lambda f: 0.0)
    thresholds = loop.status()["barge_in"]["thresholds"]
    assert thresholds["min_voiced_ms"] == 256 and thresholds["margin_db"] == 10 and thresholds["min_erle_db"] == 9
    assert loop.assistant.handle_text("hold on please").route == "stop"
    loop.assistant.close()


def test_build_voice_gives_a_speaker_device_a_canceller_and_a_headset_none(monkeypatch):
    import glide.speech.session as session

    built = []

    class Marker:
        hold = False
        stats = None

        def close(self): ...

    monkeypatch.setattr(session, "make_canceller", lambda name, warn: built.append(name) or Marker())
    monkeypatch.setattr(FullDuplexDevice, "start", lambda self: None)
    speakers = build_voice(FakeConfig(llm=FakeLLM(), tts=FakeTTS()), SpeechSettings(echo_canceller="nlms"), vad=lambda f: 0.0)
    assert built == ["nlms"] and speakers.assistant.io.player._canceller is not None
    speakers.assistant.close()
    built.clear()
    headset = build_voice(FakeConfig(llm=FakeLLM(), tts=FakeTTS()), SpeechSettings(headset=True), vad=lambda f: 0.0)
    assert built == [] and headset.assistant.io.player._canceller is None
    headset.assistant.close()


def test_a_named_canceller_that_cannot_be_built_is_an_error_not_a_quiet_half_duplex(monkeypatch):
    import glide.speech.echo as echo

    class Missing:
        def __init__(self):
            raise echo.EchoError("not installed")

    monkeypatch.setattr(echo, "CANCELLERS", {"webrtc": Missing, "nlms": Missing})
    monkeypatch.setattr(FullDuplexDevice, "start", lambda self: None)
    with pytest.raises(echo.EchoError):
        build_voice(FakeConfig(llm=FakeLLM(), tts=FakeTTS()), SpeechSettings(echo_canceller="webrtc"), vad=lambda f: 0.0)
    warned = []
    from glide.assistant.core import IO

    loop = build_voice(
        FakeConfig(llm=FakeLLM(), tts=FakeTTS()),
        SpeechSettings(echo_canceller="auto"),
        io=IO(warn=warned.append),
        vad=lambda f: 0.0,
    )
    assert len(warned) == 1 and "half duplex" in warned[0]
    loop.assistant.close()


def test_build_voice_asks_for_a_yes_per_task_when_acting_unless_the_settings_say_otherwise():
    config = FakeConfig(llm=FakeLLM(), tts=FakeTTS())
    on = build_voice(
        config,
        SpeechSettings(confirm_phrase="go ahead glide", confirm_timeout_s=5),
        act=True,
        device=fake_device(),
        vad=lambda f: 0.0,
    )
    assert on._approval is not None and on._approval.phrase == "go ahead glide" and on._approval.timeout_s == 5
    off = build_voice(config, SpeechSettings(confirm_tasks=False), act=True, device=fake_device(), vad=lambda f: 0.0)
    assert off._approval is None
    dry = build_voice(config, SpeechSettings(), act=False, device=fake_device(), vad=lambda f: 0.0)
    assert dry._approval is None  # nothing acts, so there is nothing to confirm
    for loop in (on, off, dry):
        loop.assistant.close()

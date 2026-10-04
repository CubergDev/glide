"""Integration-seam audit, round 2: settings that every front end must see, not only `glide voice`."""

from __future__ import annotations

from test_assistant_fakes import FakeConfig, FakeLLM, route_json

from glide.assistant.core import Assistant
from glide.speech.settings import SpeechSettings


class VoiceConfig(FakeConfig):
    voice = SpeechSettings(stop_phrases=("hold on please",))


def test_a_configured_stop_phrase_stops_in_every_front_end_not_only_glide_voice(tmp_path):
    # finding 5: `[speech] stop_phrases` reached only build_voice; ask, chat, listen and the app sent it to the model
    config = VoiceConfig(llm=FakeLLM(route=route_json("answer", reply="Sure.")))
    assistant = Assistant(config, runs_dir=tmp_path)  # built the way cmd_chat and cmd_listen build it
    try:
        assistant.handle_text("hold on please")
        assert config.calls.llm == 0
    finally:
        assistant.close()


def test_a_config_without_voice_settings_adds_no_phrase(tmp_path):
    config = FakeConfig(llm=FakeLLM(route=route_json("answer", reply="Sure.")))
    assistant = Assistant(config, runs_dir=tmp_path)
    try:
        assistant.handle_text("hold on please")
        assert config.calls.llm >= 1
    finally:
        assistant.close()


def test_listen_uses_the_speech_language_of_glide_toml(monkeypatch):
    # finding 16: `glide listen` took only --lang and ignored [speech] language, which `glide voice` honours
    import test_assistant_cli as rig_module

    languages = []

    class RecordingSTT(rig_module.FakeSTT):
        def stream(self, chunks, *, sample_rate=16000, language=None):
            languages.append(language)
            return super().stream(chunks, sample_rate=sample_rate, language=language)

    config = rig_module.answering("Four.")
    config.voice = SpeechSettings(language="yue")
    rig = rig_module.listen_rig(monkeypatch, RecordingSTT(final="two plus two"), config=config)
    rig_module.talk(rig)
    rig.keys.press("q")
    assert rig_module.finish(rig.thread, rig.result) == 0
    assert languages == ["yue"]


def test_the_missing_audio_package_hint_is_the_extra_every_other_command_names():
    import sys

    import pytest

    from glide.assistant.audio_io import AudioUnavailable, SoundDeviceOutput

    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(sys.modules, "sounddevice", None)
        with pytest.raises(AudioUnavailable) as caught:
            SoundDeviceOutput()
    assert "uv sync --extra speech" in str(caught.value) and "uv add" not in str(caught.value)

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

"""Hands-free voice: a full-duplex audio device, local turn detection, and the loop that feeds each turn to the assistant.

Importing this package never opens an audio device, loads a model or connects anywhere. The pieces that
need an optional package (`sounddevice`, `numpy` and `onnxruntime`) import it when one is built, and every
piece takes its hardware or model as an argument so tests give it a fake.

- `settings.py`: the `[speech]` table of glide.toml, validated.
- `vad.py`: who is speaking (Silero when installed, loudness otherwise) and where a turn begins and ends.
- `audio.py`: one microphone and one speaker as one device, with the echo guard and the headset mode.
- `turns.py`: `VoiceLoop`, which turns frames into turns, hands each to `Assistant.handle_audio`, and
  decides barge-in, self-correction and idleness.
- `session.py`: builds the real thing from settings.
- `elevenlabs.py`: an optional realtime text-to-speech adapter that speaks the `providers/tts` shape.

Speech to text is not here: the assistant asks `config.stt()`, so every transcriber stays swappable.
"""

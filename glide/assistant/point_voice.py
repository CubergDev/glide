"""Spoken questions about a pinned item: the voice loop's assistant, pointed at a `PointSession` instead of the router.

`VoiceLoop` (glide/speech/turns.py) hears a turn and hands its audio to `assistant.handle_audio`, which transcribes it
and calls `_handle(text, ...)`. `PointAssistant` is an `Assistant` whose `_handle` does not route at all: a stop phrase
cuts the answer and its readout, anything else is a question about the pin. Transcription, the stop fast-path, the
epoch that drops a request a newer one has overtaken, the speaker and its sentence queue are all the base class's, so
there is exactly one implementation of each. It never starts a computer task: the router is not called.

Four members of `Assistant` are used from here and nowhere else outside it: `_handle`, `_begin`, `_silence` and
`_speaker_or_none`. tests/test_point_voice.py runs this class over the real `Assistant`, so a change to one of them fails
there instead of in front of a person.

The session is bound after the loop is built (`bind`), because the session needs this assistant's voice and this
assistant needs the session. A question that arrives before `bind` is dropped, never queued.
"""

from __future__ import annotations

import threading

from .core import Assistant, Reply
from .router import fast_path
from .speech import detect_language, split_sentences


class PointAssistant(Assistant):
    def __init__(self, config, **options) -> None:
        super().__init__(config, **options)
        self._session = None
        self._bound = threading.Lock()

    def bind(self, session) -> None:
        with self._bound:
            self._session = session

    def _handle(self, text: str, act: bool, wait: bool, hint_language: str | None, epoch: int) -> Reply:
        text = " ".join(text.split())
        with self._bound:
            session = self._session
        if not text or session is None:
            return Reply("none")
        if fast_path(text) is not None:  # "stop": the answer and its readout end, the pin and the microphone stay
            self._silence()
            session.stop()
            return Reply("stop")
        if self._begin(epoch) is None:  # a barge-in or a stop came after this was said: a newer request owns the pin
            return Reply("none")
        session.ask(text)
        return Reply("answer", language=hint_language)

    def say(self, text: str) -> None:
        """Read an answer aloud, a sentence at a time, the way every other answer is spoken. A no-op without a speaker."""
        speaker = self._speaker_or_none()
        if speaker is None:
            return
        for sentence in split_sentences(text):
            speaker.say(sentence, language=detect_language(sentence))

    def cut_voice(self) -> None:
        """Stop the readout now, without dropping a question that is still being heard."""
        self._silence()

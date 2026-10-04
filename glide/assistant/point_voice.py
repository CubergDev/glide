"""Spoken questions about a pinned item: the voice loop's assistant, pointed at a `PointSession` instead of the router.

`VoiceLoop` (glide/speech/turns.py) hears a turn and hands its audio to `assistant.handle_audio`, which transcribes it
and handles the transcript. `PointAssistant` is an `Assistant` built with a `responder` that does not route at all:
anything that is not a stop phrase is a question about the pin. Everything else is the base class's, so there is exactly
one implementation of each: transcription, the stop phrases (the built-in ones and the user's `[speech] stop_phrases`,
which `build_voice` passes as `extra_stop_phrases`), the tickets that make a stop be for what was made before it and drop
a request a stop or a barge-in overtook, the speaker and its sentence queue. It never starts a computer task.

Only the public surface of `Assistant` is used here: the `responder` and `on_stop` constructor options, `say_aloud` and
`cut_voice`. tests/test_point_voice.py runs this class over the real `Assistant`.

A stop phrase ends the pin's answer and its readout (`on_stop` -> `PointSession.stop`); the pin and the microphone stay.
A question that a stop or a barge-in overtook is dropped by the base class before the responder sees it, and one that
is overtaken while the session takes it is ended again, so an answer to it is never shown or spoken. Anything the
session decides is late (its own generation check) never reaches `say_aloud`.

The session is bound after the loop is built (`bind`), because the session needs this assistant's voice and this
assistant needs the session. A question that arrives before `bind` is dropped, never queued.
"""

from __future__ import annotations

import threading

from .core import Assistant, Reply


class PointAssistant(Assistant):
    def __init__(self, config, **options) -> None:
        super().__init__(config, responder=self._ask, on_stop=self._stopped, **options)
        self._session = None
        self._bound = threading.Lock()

    def bind(self, session) -> None:
        with self._bound:
            self._session = session

    def _bound_session(self):
        with self._bound:
            return self._session

    def _ask(self, text: str, hint_language: str | None) -> Reply:
        session = self._bound_session()
        if session is None:
            return Reply("none")
        session.ask(text)  # supersedes the answer in flight and its readout
        return Reply("answer", language=hint_language)

    def _stopped(self) -> None:
        session = self._bound_session()
        if session is not None:
            session.stop()  # the answer and its readout end; the pin and the microphone stay

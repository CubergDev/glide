"""The assistants the app drives: a text one that always exists, and hands-free voice that starts when the app asks.

`Runtime` holds an `Assistant` for typed requests. Hands-free is a `VoiceLoop` made by `glide.speech.session.build_voice`,
which makes an assistant of its own that speaks through the sound device; while it runs, typed requests go to that one too,
so the person hears the answers. `stop` reaches both.

Turning hands-free on is what opens the microphone, so it happens only on the app's explicit setting, never at start-up. The one
function that builds the real device (`real_voice`) is named only by `cli.serve`: a `Runtime` made without a `voice_factory` has no
voice at all, and `tests/guards_core-socket.py` refuses `real_voice` itself. Microphone and speaker belong to this process, not to the app.
"""

from __future__ import annotations

import contextlib
import logging
import threading
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..assistant.core import IO, Assistant
from ..assistant.tasks import DEFAULT_RUNS_DIR
from ..speech.settings import SpeechSettings
from .settings import SettingsState, VoiceUnavailable
from .wire import clean_text

log = logging.getLogger("glide.app_server")

VoiceFactory = Callable[[Any, SpeechSettings, IO, bool], Any]


def real_voice(config: Any, settings: SpeechSettings, io: IO, act: bool) -> Any:
    """The voice loop over the real sound device, not yet started. Needs the speech extra. Opens the microphone when started."""
    from ..speech.session import build_voice

    return build_voice(config, settings, io=io, act=act)


class Runtime:
    def __init__(
        self,
        config: Any,
        make_io: Callable[[], IO],
        *,
        runs_dir: Path = DEFAULT_RUNS_DIR,
        voice_factory: VoiceFactory | None = None,
        assistant_factory: Callable[..., Assistant] = Assistant,
    ) -> None:
        self.config = config
        self._make_io = make_io
        self._voice_factory = voice_factory
        self._text = assistant_factory(config, io=make_io(), runs_dir=runs_dir, clarify=True)
        self._loop: Any = None
        self._muted = False
        self._lock = threading.RLock()

    @property
    def voice_available(self) -> bool:
        return self._voice_factory is not None

    @property
    def assistant(self) -> Assistant:
        """Where a typed request goes: the voice loop's assistant while hands-free is on, else the text one."""
        loop = self._loop
        return loop.assistant if loop is not None else self._text

    def assistants(self) -> list[Assistant]:
        loop = self._loop
        return [self._text] if loop is None else [self._text, loop.assistant]

    @property
    def hands_free(self) -> bool:
        return self._loop is not None

    @property
    def muted(self) -> bool:
        return self._muted

    @property
    def voice_failure(self) -> str | None:
        loop = self._loop
        return getattr(loop, "failure", None) if loop is not None else None

    # -- voice --------------------------------------------------------------------------------------

    def apply_voice(self, state: SettingsState) -> None:
        """Make the voice loop match `state`: start it, stop it, or restart it with new options. Raises `VoiceUnavailable`."""
        with self._lock:
            if self._loop is not None:
                self._stop_loop()  # a change of options restarts the loop; so does turning it off
            if not state.hands_free:
                return
            if self._voice_factory is None:
                raise VoiceUnavailable("hands-free voice is not available in this core")
            settings = replace(
                getattr(self.config, "voice", None) or SpeechSettings(),
                headset=state.headset,
                silence_ms=state.silence_ms,
                language=state.language,
            )
            loop = None
            try:
                loop = self._voice_factory(self.config, settings, self._make_io(), state.act_enabled)
                loop.start()
            except Exception as exc:  # no speech extra, no microphone, a model that is not there: the person is told why
                if loop is not None:
                    with contextlib.suppress(Exception):
                        loop.assistant.close()
                raise VoiceUnavailable(self._reason(exc)) from None
            if self._muted:
                loop.pause()
            self._loop = loop

    def mute(self, muted: bool) -> None:
        with self._lock:
            self._muted = muted
            loop = self._loop
            if loop is not None:
                loop.pause() if muted else loop.resume()

    def _stop_loop(self) -> None:
        loop, self._loop = self._loop, None
        if loop is None:
            return
        with contextlib.suppress(Exception):
            loop.stop()
        with contextlib.suppress(Exception):
            loop.assistant.close()

    def _reason(self, exc: BaseException) -> str:
        scrub = getattr(self.config, "scrub", None)
        text = str(exc)
        return clean_text(scrub(text) if callable(scrub) else text, 200) or type(exc).__name__

    # -- stopping --------------------------------------------------------------------------------------

    def stop(self) -> None:
        for assistant in self.assistants():
            assistant.stop()

    def interrupt_speech(self) -> None:
        for assistant in self.assistants():
            assistant.interrupt_speech()

    def close(self) -> None:
        with self._lock:
            self._stop_loop()
        with contextlib.suppress(Exception):
            self._text.close()

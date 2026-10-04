"""Point and ask, for the pet: the countdown, the pin, and the session about it. No Qt in this module.

`PointMode.start` runs on a worker thread: say what will be shared, wait out a short countdown so the person can
position the pointer, read the one item under it (`capture_point`), and either show what was read (a local preview, with
nothing sent anywhere) or open a `PointSession` and ask the first question. Later questions, typed or spoken, go to
the same session and the same pin until the person stops, points again, or the pin is two minutes old.

Nothing is shared by default: `share` is the person's explicit decision to send the question, the item's text and, if they
also ask for it, a small crop to the answer provider named in the first status line. A preview reads the item locally and
sends nothing. Everything captured, asked or answered stays in memory (point_session.py) and in the events it makes;
none of it is written to disk, whatever the recording setting.

Events go to `emit(kind, **data)`, with the kinds of point_session.py plus `status`, `selecting`, `selected` (point, text),
`preview` and `closed`. The pet turns them into `PetEvent("point", ...)`.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

from ..assistant.point_ask import PointSelection, PointUnavailable, capture_point
from ..assistant.point_session import PointSession
from ..assistant.point_voice import PointAssistant
from ..computer.config import writer_vision
from ..computer.writer import make_writer, provider
from ..providers.config import ConfigError

POINT_DELAY_S = 3.0  # time to position the pointer after pressing Ask
LABEL_CHARS = 1024
NO_PROVIDER = "Configure an answer provider first: the smart LLM chain in glide.toml has no usable slot."
TEXT_ONLY = "The configured answer model is text-only; ask without an image."
UNREADABLE = "Could not read this point. Check the desktop permissions."


class PointMode:
    def __init__(
        self,
        config,
        emit: Callable[..., None],
        *,
        start_voice: Callable[..., bool],
        end_voice: Callable[[], None],
        capture: Callable[..., PointSelection] = capture_point,
        delay_s: float = POINT_DELAY_S,
        vision: Callable[[], bool] | None = None,
    ) -> None:
        self._config, self._emit, self._start_voice, self._end_voice = config, emit, start_voice, end_voice
        self._capture, self._delay, self._vision = capture, delay_s, vision or writer_vision
        self._lock = threading.RLock()
        self._session: PointSession | None = None
        self._voiced = False  # the session was opened with a microphone, which the end of the pin must also end
        self._cancel: threading.Event | None = None  # set while a countdown or capture is under way
        self._counting_for_voice = False  # that countdown ends by opening a microphone

    @property
    def active(self) -> bool:
        """A countdown is running or a pin is held."""
        with self._lock:
            return self._cancel is not None or (self._session is not None and self._session.available)

    @property
    def holding(self) -> bool:
        """A session about a pin is open: a question goes to it rather than starting a new pin."""
        with self._lock:
            return self._session is not None and self._session.available

    def start(self, question: str = "", *, share: bool = False, with_image: bool = False, voice: bool = False) -> None:
        """Pin the item under the pointer after the countdown, then ask `question` about it if there is one.

        Returns at once. Pointing again while a pin is held replaces it (and forgets its exchanges); pointing while a
        countdown runs is ignored. `with_image` and `voice` need `share`: there is nothing to ask without it."""
        with self._lock:
            if self._cancel is not None:
                return
            self._cancel = cancel = threading.Event()
            self._counting_for_voice = share and voice
        options = (question.strip(), share, share and with_image, share and voice)
        threading.Thread(target=self._pin, args=(cancel, *options), name="glide-point", daemon=True).start()

    def cancel_voice_countdown(self) -> bool:
        """The microphone this pin's countdown was going to open is no longer wanted. A countdown for a typed question, or
        a pin already held, is left alone. True when a countdown was cancelled."""
        with self._lock:
            if self._cancel is None or not self._counting_for_voice:
                return False
            cancel, self._cancel, self._counting_for_voice = self._cancel, None, False
        cancel.set()
        self._emit("status", text="Voice question cancelled.")
        return True

    def ask(self, text: str) -> bool:
        """A follow-up about the pin. False, and a status line, when there is none to ask about."""
        with self._lock:
            session = self._session
        if session is None or not session.available:
            self._emit("status", text="Select a point first.")
            return False
        return session.ask(text)

    def stop(self) -> None:
        """Cancel the answer and its readout. The pin and the session stay."""
        with self._lock:
            session = self._session
        if session is not None:
            session.stop()

    def close(self) -> None:
        """Cancel a countdown, end the session, forget the pin. Returns at once."""
        with self._lock:
            cancel, self._cancel = self._cancel, None
            self._counting_for_voice = False
            session, self._session = self._session, None
            self._voiced = False
        if cancel is not None:
            cancel.set()
        if session is not None:
            session.close()
        self._emit("closed")

    # -- the worker --------------------------------------------------------------------------------

    def _pin(self, cancel: threading.Event, question: str, share: bool, with_image: bool, voice: bool) -> None:
        try:
            writer = self._disclose(share, with_image)
            if writer is None and share:
                return
            self._emit("selecting", text=f"Position the pointer. Reading in {self._delay:g} seconds.")
            if cancel.wait(self._delay):
                return
            selection = self._capture(with_image=with_image)
            if cancel.is_set():
                selection.close()
                return
            target = selection.target
            label = (target.label if target is not None and target.label else "Selected screen point")[:LABEL_CHARS]
            self._emit("selected", point=list(selection.point), text=label)
            if writer is None:
                self._preview(selection)
            elif not self._open(selection, writer, question, voice, cancel):
                selection.close()
                if not self.holding:
                    self._emit("closed")  # `selected` drew a pin; with no session behind it the view must not keep it
        except PointUnavailable as error:  # our own sentences: protected field, nothing under the pointer
            self._emit("error", text=str(error), closed=True)
        except Exception:  # the adapters' errors can name the screen; the person gets the fixed sentence
            self._emit("error", text=UNREADABLE, closed=True)
        finally:
            with self._lock:
                if self._cancel is cancel:
                    self._cancel = None
                    self._counting_for_voice = False

    def _disclose(self, share: bool, with_image: bool):
        """Say what is about to be shared, before anything is read. The writer, or None for a local preview."""
        if not share:
            self._emit("status", text="Local preview: reads the pointed item's accessibility text. Nothing is sent to a model.")
            return None
        try:
            writer = make_writer(self._config)
        except ConfigError as error:  # a slot of glide.toml that cannot be set up: its message names the file and slot, no key
            self._emit("error", text=self._scrub(str(error)), closed=True)
            return None
        if writer is None:
            self._emit("error", text=NO_PROVIDER, closed=True)
            return None
        if with_image and not self._vision():
            self._emit("error", text=TEXT_ONLY, closed=True)
            return None
        self._emit("status", text=f"Answer provider: {provider(writer)}")
        self._emit(
            "status",
            text="This request shares your question, the pointed item's text"
            + (" and a small image crop." if with_image else "."),
        )
        return writer

    def _scrub(self, text: str) -> str:
        scrub = getattr(self._config, "scrub", None)
        return " ".join((scrub(text) if callable(scrub) else text).split())

    def _preview(self, selection: PointSelection) -> None:
        packet = selection.target.packet() if selection.target is not None else {}
        self._emit("preview", text="\n".join(f"{name}: {value}" for name, value in packet.items() if value))
        selection.close()  # read, shown, and forgotten: nothing was asked, so nothing is held

    def _open(self, selection: PointSelection, writer, question: str, voice: bool, cancel: threading.Event) -> bool:
        with self._lock:
            session = self._session
        if session is not None and session.available and (self._voiced or not voice):
            session.repoint(selection)  # the same session and voice, a new pin, no old exchanges
        elif voice:
            if session is not None:
                session.close()  # a typed session cannot take a microphone: it ends, and a spoken one begins
            session = self._open_voice(selection, writer)
            if session is None:
                return False
        else:
            session = PointSession(selection, writer, self._session_event)
        with self._lock:
            if cancel.is_set():
                session.close()
                return True
            self._session = session
        if question:
            session.ask(question)
        return True

    def _open_voice(self, selection: PointSelection, writer) -> PointSession | None:
        made: list[PointSession] = []

        def bind(assistant: PointAssistant) -> None:
            session = PointSession(
                selection, writer, self._session_event, speak=assistant.say_aloud, cancel_speech=assistant.cut_voice
            )
            assistant.bind(session)
            made.append(session)

        started = self._start_voice(lambda config, io=None, **options: PointAssistant(config, io=io, **options), bind)
        if not (started and made):
            return None
        self._voiced = True
        return made[0]

    def _session_event(self, kind: str, **data) -> None:
        self._emit(kind, **data)
        if kind == "error" and data.get("closed") and self._voiced:
            self._voiced = False
            self._end_voice()  # the pin is gone, so a microphone left open would be listening for nothing

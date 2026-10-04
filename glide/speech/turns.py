"""The hands-free loop: frames in, turns out, each turn handed to `Assistant.handle_audio`.

The loop owns exactly three decisions and nothing else (no transcription, no routing, no speech):

1. Where a turn starts and ends: `TurnDetector` on the voice detector's probabilities. One utterance has one
   end, decided by silence (or by `pause()`), never by anything earlier. In particular nothing "commits
   early" on a short pause: that once caused a second end 154 ms after the first.
2. Self-correction. With `merge_window_s > 0` a turn is not ended by `silence_ms` alone but only after
   `merge_window_s` more of quiet, and speech inside that window continues the SAME turn. "Open Safari ...
   no, Chrome" is one request and the transcriber hears all of it. The audio is not edited: deciding that
   "no, Chrome" replaced "Safari" is the router's job, which sees the whole sentence.
3. Barge-in. When speech starts while Glide is audibly speaking, the voice is cut at once through
   `Assistant.interrupt_speech`. That call drops any request still being heard, so it is made only when no
   earlier turn is still waiting for its transcript; otherwise the earlier turn would be thrown away. (The
   new turn's own `handle_audio` cuts the voice anyway when it starts.) In speaker mode the device hears
   silence while Glide speaks, so barge-in by voice needs a headset.

Idle: with `idle_s > 0`, once nothing has happened for that long (no speech, nothing playing, no task, no
turn in flight) the microphone is switched off and `on_idle` is called. Only `resume()` turns it back on.

Every turn runs `handle_audio` on its own thread, so hearing the next turn never waits for the last answer.
The only exceptions are reported, as their type name only, through `assistant.io.warn`.
"""

from __future__ import annotations

import math
import queue
import threading
import time
from collections.abc import Callable, Iterator

from .audio import DeviceFault
from .vad import FRAME_S, Probability, TurnDetector, UtteranceTooLong

POLL_S = 0.05


class TurnAborted(RuntimeError):
    """Raised inside a turn's audio so that the transcriber stops and nothing is submitted."""


class _Turn:
    """One utterance's audio on its way to a worker thread."""

    def __init__(self) -> None:
        self.audio: queue.Queue = queue.Queue()
        self.heard = threading.Event()  # the transcript is final (or there will be none): safe to interrupt again
        self.thread: threading.Thread | None = None

    def chunks(self) -> Iterator[bytes]:
        while True:
            item = self.audio.get()
            if item is None:
                return
            if isinstance(item, BaseException):
                raise item
            yield item


class VoiceLoop:
    def __init__(
        self,
        assistant,
        device,
        vad: Probability,
        *,
        silence_ms: int = 600,
        merge_window_s: float = 0.0,
        idle_s: float = 0.0,
        language: str | None = None,
        act: bool = False,
        on_idle: Callable[[], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._assistant = assistant
        self._device = device
        self._vad = vad
        self._detector = TurnDetector(silence_ms)
        self._merge_frames = math.ceil(merge_window_s / FRAME_S) if merge_window_s > 0 else 0
        self._idle_s = idle_s
        self._language = language
        self._act = act
        self._on_idle = on_idle
        self._clock = clock
        self._stop = threading.Event()
        self._commands: queue.SimpleQueue[str] = queue.SimpleQueue()
        self._turn: _Turn | None = None
        self._turns: list[_Turn] = []
        self._merging = 0  # frames of quiet left in the merge window; 0 when not merging
        self._paused = False
        self._active_at = clock()
        self._local = threading.local()
        self._thread: threading.Thread | None = None
        self.failure: str | None = None
        self._io = assistant.io
        self._chained_heard = heard = self._io.heard

        def on_heard(text: str) -> None:
            turn = getattr(self._local, "turn", None)
            if turn is not None:
                turn.heard.set()
            heard(text)

        self._on_heard = self._io.heard = on_heard

    @property
    def assistant(self):
        return self._assistant

    # -- control (any thread) -------------------------------------------------------------------

    def start(self) -> None:
        self._thread = threading.Thread(target=self.run, name="glide-voice", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._commands.put("wake")
        if self._thread is not None and threading.current_thread() is not self._thread:
            self._thread.join(timeout=5)

    def pause(self) -> None:
        """End the turn being said as a silence would, then turn the microphone off. Answers and speech go on."""
        self._commands.put("pause")

    def resume(self) -> None:
        self._commands.put("resume")

    def join_turns(self, timeout: float = 5.0) -> None:
        """Wait for the worker threads of the turns heard so far."""
        end = time.monotonic() + timeout
        for turn in list(self._turns):
            if turn.thread is not None:
                turn.thread.join(max(0.0, end - time.monotonic()))

    # -- the loop -------------------------------------------------------------------------------

    def run(self) -> None:
        try:
            while not self._stop.is_set():
                self._apply_commands()
                if self._paused:
                    continue
                try:
                    frame = self._device.read()
                except DeviceFault as exc:
                    self._abort()
                    self.failure = str(exc)
                    self._assistant.io.warn(self.failure)
                    return
                if frame is not None:
                    self._frame(frame)
                self._check_idle()
        finally:
            self._abort()  # stopping is not an end of speech: a half-said command is discarded, never submitted
            if self._io.heard is self._on_heard:
                self._io.heard = self._chained_heard

    def _apply_commands(self) -> None:
        """Run the pause and resume asked for. While paused this waits for one, so a paused loop does not spin."""
        wait = POLL_S if self._paused else 0.0
        while True:
            try:
                command = self._commands.get(timeout=wait) if wait else self._commands.get_nowait()
            except queue.Empty:
                return
            wait = 0.0
            if command == "pause" and not self._paused:
                self._end_turn()
                self._device.pause_input()
                self._paused = True
            elif command == "resume" and self._paused:
                try:
                    self._device.resume_input()
                except Exception as exc:  # the microphone stays off, and the person is told
                    self._assistant.io.warn(f"the microphone could not be turned on ({type(exc).__name__})")
                    continue
                self._paused = False
                self._active_at = self._clock()

    def _frame(self, frame: bytes) -> None:
        try:
            started, data, commit = self._detector.feed(frame, self._vad(frame))
        except UtteranceTooLong as exc:
            self._abort()
            self._assistant.io.warn(str(exc))
            return
        turn = self._turn
        if turn is None:
            if started:
                self._begin(data)
            return
        if started:  # speech inside the merge window: the same turn goes on
            self._merging = 0
            turn.audio.put(frame)
            return
        if self._merging:
            turn.audio.put(frame)
            self._merging -= 1
            if not self._merging:
                self._end_turn()
            return
        if data:
            turn.audio.put(data)
        if commit:
            if self._merge_frames:
                self._merging = self._merge_frames
            else:
                self._end_turn()

    # -- turns ----------------------------------------------------------------------------------

    def _begin(self, first: bytes) -> None:
        self._active_at = self._clock()
        if self._device.playing and all(turn.heard.is_set() for turn in self._turns):
            self._assistant.interrupt_speech()  # barge-in, before this turn reads the assistant's epoch
        turn = self._turn = _Turn()
        turn.audio.put(first)
        turn.thread = threading.Thread(target=self._work, args=(turn,), name="glide-voice-turn", daemon=True)
        self._turns = [t for t in self._turns if t.thread is not None and t.thread.is_alive()] + [turn]
        turn.thread.start()

    def _end_turn(self) -> None:
        turn, self._turn = self._turn, None
        self._merging = 0
        self._detector.reset()
        if turn is not None:
            turn.audio.put(None)
            self._active_at = self._clock()

    def _abort(self) -> None:
        """Discard the turn in progress: the worker's audio raises instead of ending, so nothing half-heard is submitted."""
        turn, self._turn = self._turn, None
        self._merging = 0
        self._detector.reset()
        if turn is not None:
            turn.audio.put(TurnAborted())

    def _work(self, turn: _Turn) -> None:
        self._local.turn = turn
        try:
            self._assistant.handle_audio(turn.chunks(), act=self._act, wait=False, language=self._language)
        except TurnAborted:
            pass  # already reported by whoever aborted it
        except Exception as exc:  # one bad turn must not end hands-free listening; the type is all that is said
            self._assistant.io.warn(f"a voice turn failed ({type(exc).__name__})")
        finally:
            turn.heard.set()

    # -- idle -----------------------------------------------------------------------------------

    def _check_idle(self) -> None:
        if not self._idle_s or self._paused:
            return
        now = self._clock()
        busy = (
            self._turn is not None
            or self._device.playing
            or self._assistant.busy
            or any(t.thread is not None and t.thread.is_alive() for t in self._turns)
        )
        if busy:
            self._active_at = now
        elif now - self._active_at >= self._idle_s:
            self._commands.put("pause")
            if self._on_idle is not None:
                self._on_idle()

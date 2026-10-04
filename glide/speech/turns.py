"""The hands-free loop: frames in, turns out, each turn handed to `Assistant.handle_audio`.

The loop owns exactly four decisions and nothing else (no transcription, no routing, no speech):

1. Where a turn starts and ends: `TurnDetector` on the voice detector's probabilities. One utterance has one
   end, decided by silence (or by `pause()`), never by anything earlier. In particular nothing "commits
   early" on a short pause: that once caused a second end 154 ms after the first.
2. Self-correction. With `merge_window_s > 0` a turn is not ended by `silence_ms` alone but only after
   `merge_window_s` more of quiet, and speech inside that window continues the SAME turn. "Open Safari ...
   no, Chrome" is one request and the transcriber hears all of it. The audio is not edited: deciding that
   "no, Chrome" replaced "Safari" is the router's job, which sees the whole sentence.
3. Barge-in, by voice, through speakers or a headset. While Glide is speaking (or its echo may still be in the
   microphone), no frame can start a turn on the voice detector's say-so alone. `BargeInGate` asks for more:
   the cleaned frame (the device removed Glide's own voice with the far-end reference, see echo.py) must be
   voiced AND louder than the echo still expected in it by a margin, for `barge_min_voiced_ms`, and only once the
   canceller has measured how much echo it removes (`barge_min_erle_db`). A cough, a click or "mm-hm" is shorter
   than that and is dropped; Glide's own echo never gets louder than its residual, so it never qualifies. When
   it does qualify, in this order and at once: the voice is cut (`Assistant.interrupt_speech`, which also bumps
   the epoch so queued speech and any late answer are dropped, and cancels the answer being written), and then
   the new turn starts from the frames kept since a little before the sound began, so the first words are not lost.
   The epoch is bumped BEFORE the turn reads it: a turn that began first would be dropped by its own barge-in.
   `interrupt_speech` drops pending requests only when no earlier turn of the person's is still waiting for its
   transcript; otherwise the earlier turn would be thrown away.

   A spoken stop is allowed weaker evidence. A shorter and quieter sound opens a stop-only probe: the audio
   goes to the transcriber and `Assistant.handle_audio(stop_only=True)` acts on it only if it is a stop phrase
   (the router's deterministic fast path, plus the configured `stop_phrases`). Anything else it hears is dropped
   unheard, so Glide's own transcribed echo can never become a request. Probes are one at a time and rate limited.
   If the sound then turns out to be the person after all, the probe is abandoned and the ordinary turn takes over.

   With no canceller (headset, `echo_canceller = "none"`, or none available) nothing changes underneath: a
   headset hears no echo, so the voice detector and the minimum duration decide; half-duplex speaker mode
   hears silence while Glide speaks, so nothing qualifies until it has finished.

   A turn that the voice detector starts on its own (nothing of Glide's in the microphone) cuts an answer that is still
   being routed or written, or whose speech is queued but not yet audible, the same way: the device is silent then, so
   the gate cannot see it, and the answer would otherwise speak over the person's new request.

   The policy keeps counters and thresholds (`VoiceLoop.status()`): ERLE, delay, decisions. They are numbers,
   never audio or words, and are for the person to read, never to be spoken.

4. Whether a voice request may act. Hands-free listening has no wake word and no speaker check, so with `act` a request
   that becomes a computer task runs as a dry run first, and `TaskApproval` (approval.py) holds the window in which the
   confirm phrase, or `confirm()`, makes the same request real. The phrase's own turn is never routed as a request. A
   no (another sentence, a stop, talking over Glide, a pause, the time) is the default, and `confirm_tasks = false` is the
   explicit way to give it up.

Idle: with `idle_s > 0`, once nothing has happened for that long (no speech, nothing playing, no task, no
turn in flight) the microphone is switched off and `on_idle` is called. Only `resume()` turns it back on.

Every turn runs `handle_audio` on its own thread, so hearing the next turn never waits for the last answer.
The only exceptions are reported, as their type name only, through `assistant.io.warn`.
"""

from __future__ import annotations

import contextlib
import math
import queue
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Protocol

from ..assistant.audio_io import rms
from .approval import Pending, TaskApproval
from .audio import DeviceFault
from .echo import EchoStats
from .settings import SpeechSettings
from .vad import FRAME_S, START_PROBABILITY, STOP_PROBABILITY, Probability, TurnDetector, UtteranceTooLong

POLL_S = 0.05
DRY_RUN_WAIT_S = 120.0  # the longest a previewed task is waited for before it is no longer offered
PREVIEW_HEARD_S = 10.0  # the longest its description is waited for before the window opens anyway

# Barge-in policy. The three the person can change are `[speech]` keys (their defaults are in `SpeechSettings`); the rest
# are the shape of the policy.
PROBE_FRAMES = 3  # frames of weaker evidence that open a stop-only probe (96 ms)
PROBE_MARGIN_LESS_DB = 3.0  # the probe's margin is the policy's minus this (not below 3 dB)
PROBE_COOLDOWN_S = 1.0  # between one probe ending and the next opening
PROBE_QUIET_FRAMES = 12  # frames without evidence that end a probe (384 ms)
PROBE_MAX_FRAMES = 94  # about 3 s: a stop is short
HOLD_MAX_FRAMES = 31  # the canceller's statistics are frozen for a sound under judgement for at most this long (1 s)
SUSPECT_RATIO = 1.4  # a voiced frame this far above the echo expected in it may be the person: statistics freeze
SOUND_GAP_FRAMES = 12  # a sound goes on through dips this long (384 ms); the turn that interrupts starts from its first evidence
RUN_GAP_FRAMES = 2  # speech has dips: evidence may skip this many frames without ending the run
RETAINED_MIN = 0.25  # at least this share of the microphone's amplitude must survive cancelling: echo it understood does not
MIN_LEVEL_RMS = 60.0  # the cleaned level is never judged against less than this (16-bit scale)
BARGE_LEAD_FRAMES = (
    12  # the turn that interrupts starts this long (384 ms) before the sound that qualified: a voice can rise slowly
)
HISTORY_FRAMES = 64  # frames kept to start a turn from: 2 s


class TurnAborted(RuntimeError):
    """Raised inside a turn's audio so that the transcriber stops and nothing is submitted."""


@dataclass(frozen=True)
class Verdict:
    """What one frame did to the gate. `weak` and `strong` say whether this frame was evidence at each margin."""

    confirm: bool = False  # the person is talking over Glide: cut the voice and start the turn
    probe: bool = False  # enough weak evidence to listen for a stop
    weak: bool = False
    strong: bool = False


class _Run:
    """Evidence frames in a row, allowing a few misses: speech has dips, and a sound is not judged on one frame."""

    def __init__(self) -> None:
        self.count = 0
        self.gap = 0
        self.start = 0  # frame number of the first evidence frame
        self.ended_with = 0  # the count of the run that just ended, for the caller; 0 when none did

    def feed(self, frame_no: int, evidence: bool) -> None:
        self.ended_with = 0
        if evidence:
            if self.count == 0:
                self.start = frame_no
            self.count += 1
            self.gap = 0
        elif self.count:
            self.gap += 1
            if self.gap > RUN_GAP_FRAMES:
                self.ended_with, self.count, self.gap = self.count, 0, 0

    def clear(self) -> None:
        self.count = self.gap = self.ended_with = 0


class BargeInGate:
    """Decides, frame by frame while Glide may be heard in the microphone, whether a sound is the person.

    A frame is evidence when the voice detector says speech and, if there is an echo canceller, the canceller
    has measured at least `min_erle_db` of echo removal and the cleaned level is `margin_db` above the echo
    still expected in this frame. `strong` evidence (the policy's margin) held for `min_voiced_frames` confirms.
    `weak` evidence (a smaller margin) held for `probe_frames` frames beyond what is strong asks for a stop-only probe. Without a canceller
    there is nothing to compare with and the voice detector alone is the evidence, so only the duration protects.
    Pure policy: it never touches a device, a clock or a thread.
    """

    def __init__(
        self,
        *,
        min_voiced_ms: int = SpeechSettings.barge_min_voiced_ms,
        margin_db: float = SpeechSettings.barge_margin_db,
        min_erle_db: float = SpeechSettings.barge_min_erle_db,
        probe_frames: int = PROBE_FRAMES,
    ) -> None:
        self.min_voiced_frames = max(1, math.ceil(min_voiced_ms / 1000 / FRAME_S))
        self.probe_frames = min(probe_frames, self.min_voiced_frames)
        self.margin_db = margin_db
        self.probe_margin_db = max(3.0, margin_db - PROBE_MARGIN_LESS_DB)
        self.min_erle_db = min_erle_db
        self._strong = _Run()
        self._weak = _Run()
        self._suspect = _Run()
        self._sound_start: int | None = None  # the first frame of weak evidence of the sound being judged
        self._quiet_since = 0  # frames since the last weak evidence
        self._frame_no = 0
        self._candidate = False
        self._trusted: EchoStats | None = None  # the canceller's numbers from before the sound under judgement began
        self.counters = {"frames": 0, "candidates": 0, "rejected": 0, "confirmed": 0, "waiting_for_erle": 0}

    @property
    def armed(self) -> bool:
        """Something louder than the echo expected is voiced: the canceller's statistics must not learn from it."""
        return 0 < self._suspect.count <= HOLD_MAX_FRAMES

    @property
    def onset_back(self) -> int:
        """Frames from the current one back to the first evidence of the sound being judged, dips and all."""
        return 0 if self._sound_start is None else self._frame_no - self._sound_start

    @property
    def thresholds(self) -> dict[str, float]:
        return {
            "min_voiced_ms": round(self.min_voiced_frames * FRAME_S * 1000),
            "probe_ms": round(self.probe_frames * FRAME_S * 1000),
            "margin_db": self.margin_db,
            "probe_margin_db": self.probe_margin_db,
            "min_erle_db": self.min_erle_db,
        }

    def reset(self) -> None:
        self._close_candidate()
        self._strong.clear()
        self._weak.clear()
        self._suspect.clear()
        self._sound_start = None
        self._trusted = None

    def _close_candidate(self) -> None:
        if self._candidate:
            self.counters["rejected"] += 1
        self._candidate = False

    def feed(self, probability: float, level: float, echo: EchoStats | None) -> Verdict:
        self._frame_no += 1
        self.counters["frames"] += 1
        # The canceller has already learned from this frame by the time its numbers arrive, and a voice it cannot remove
        # drags the ERLE down within that one frame. So a frame is judged on the numbers from before it, and while a sound
        # is under judgement (`armed`) on the numbers from before the sound began: it is not held against itself.
        trusted = echo if self._trusted is None else self._trusted
        strong, weak, suspect = self._evidence(probability >= START_PROBABILITY, level, echo, trusted)
        self._note_sound(weak)
        self._suspect.feed(self._frame_no, suspect)
        self._strong.feed(self._frame_no, strong)
        self._weak.feed(self._frame_no, weak)
        if not self.armed:
            self._trusted = echo
        if self._weak.ended_with:
            self._close_candidate()
        return self._verdict(weak, strong)

    def _evidence(self, voiced: bool, level: float, echo: EchoStats | None, trusted: EchoStats | None) -> tuple[bool, bool, bool]:
        """Whether this frame is strong evidence, weak evidence, and a suspect (louder than the echo expected) of a person.
        `trusted` holds the canceller's measure of itself (ERLE, the echo expected); `echo` holds this frame's near level."""
        if not voiced or echo is None or trusted is None:
            return voiced, voiced, voiced  # with no canceller there is nothing to compare with: the voice detector decides
        if not trusted.measured or (trusted.erle_db or 0.0) < self.min_erle_db:
            self.counters["waiting_for_erle"] += 1
            return False, False, False
        floor = max(trusted.expected_residual_rms, MIN_LEVEL_RMS)
        kept = level >= RETAINED_MIN * echo.near_rms  # a frame the canceller all but removed was echo it understood
        return (
            kept and level >= floor * 10 ** (self.margin_db / 20),
            kept and level >= floor * 10 ** (self.probe_margin_db / 20),
            level >= floor * SUSPECT_RATIO,
        )

    def _note_sound(self, weak: bool) -> None:
        """Where the sound being judged began: its first weak evidence, and again after a gap too long to be a dip in speech."""
        if not weak:
            self._quiet_since += 1
            return
        if self._sound_start is None or self._quiet_since >= SOUND_GAP_FRAMES:
            self._sound_start = self._frame_no
        self._quiet_since = 0

    def _verdict(self, weak: bool, strong: bool) -> Verdict:
        if self._strong.count >= self.min_voiced_frames:
            self.counters["confirmed"] += 1
            self._candidate = False
            return Verdict(confirm=True, weak=weak, strong=strong)
        # a probe is for what is too weak to confirm: a loud voice that is confirming anyway needs none
        if self._weak.count >= self.probe_frames and not self._candidate:
            self._candidate = True
            self.counters["candidates"] += 1
        probe = self._weak.count >= self.probe_frames and self._weak.count - self._strong.count >= self.probe_frames
        return Verdict(probe=probe, weak=weak, strong=strong)


class _Turn:
    """One utterance's audio on its way to a worker thread."""

    def __init__(self, *, probe: bool = False) -> None:
        self.audio: queue.Queue = queue.Queue()
        self.frames = 0  # frames of the utterance so far, merge-window pauses included: the cap is on the whole turn
        self.heard = threading.Event()  # the transcript is final (or there will be none): safe to interrupt again
        self.thread: threading.Thread | None = None
        self.probe = probe  # listens for a stop only: it never makes a request
        self.confirmed: Pending | None = None  # set when what was heard is the person's yes to a previewed task

    def add(self, frame: bytes) -> None:
        self.frames += 1
        self.audio.put(frame)

    def chunks(self) -> Iterator[bytes]:
        while True:
            item = self.audio.get()
            if item is None:
                return
            if isinstance(item, BaseException):
                raise item
            yield item


def format_status(status: dict) -> str:
    """One line of `VoiceLoop.status()` for a person to read while trying the voice: numbers and thresholds, no content."""
    b, echo = status["barge_in"], status["echo"]
    t = b["thresholds"]
    parts = [
        "echo: none"
        if echo is None
        else (
            f"echo: {echo['canceller']} erle {'?' if echo['erle_db'] is None else echo['erle_db']} dB"
            f" delay {'?' if echo['delay_ms'] is None else round(echo['delay_ms'])} ms latency {echo['latency_ms']} ms"
            f" reference {echo['far_rms']} residual {echo['out_rms']} expected {echo['expected_residual_rms']}"
            f" underruns {echo['reference_underruns']}"
        ),
        f"barge-in: confirmed {b['confirmed']} probes {b['probes']} candidates {b['candidates']} rejected {b['rejected']}"
        f" waiting-for-erle {b['waiting_for_erle']} of {b['frames']} frames",
        f"needs {t['min_voiced_ms']} ms, {t['margin_db']} dB over the echo, erle {t['min_erle_db']} dB (probe {t['probe_ms']} ms, {t['probe_margin_db']} dB)",
    ]
    return " | ".join(parts)


class Device(Protocol):
    """What the loop needs of the microphone and the speaker. `FullDuplexDevice` is one; tests give a fake."""

    playing: bool  # audio is queued or still sounding
    echo_active: bool  # Glide's voice may be in the microphone (only with an echo canceller)
    guarding: bool  # the microphone is being blanked while Glide speaks (speaker mode with no canceller): frames are zeros
    echo: EchoStats | None  # the canceller's latest numbers, or None when there is none
    echo_name: str | None
    reference_underruns: int
    output_faults: int  # blocks the sound card reported a problem with while speaking

    def read(self, timeout: float | None = ...) -> bytes | None: ...
    def pause_input(self) -> None: ...
    def resume_input(self) -> None: ...
    def hold_echo_stats(self, hold: bool) -> None: ...


class VoiceLoop:
    def __init__(
        self,
        assistant,
        device: Device,
        vad: Probability,
        *,
        silence_ms: int = SpeechSettings.silence_ms,
        merge_window_s: float = SpeechSettings.merge_window_s,
        idle_s: float = SpeechSettings.idle_s,
        language: str | None = None,
        act: bool = False,
        on_idle: Callable[[], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        barge_min_voiced_ms: int = SpeechSettings.barge_min_voiced_ms,
        barge_margin_db: float = SpeechSettings.barge_margin_db,
        barge_min_erle_db: float = SpeechSettings.barge_min_erle_db,
        confirm_tasks: bool = SpeechSettings.confirm_tasks,
        confirm_phrase: str = SpeechSettings.confirm_phrase,
        confirm_timeout_s: float = SpeechSettings.confirm_timeout_s,
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
        self._output_faults = 0  # how many the person has been told of: once is enough
        self._skip_tail = False  # an overlong utterance was discarded: what is left of it is not a request
        self._tail_quiet = 0
        self._gate = BargeInGate(min_voiced_ms=barge_min_voiced_ms, margin_db=barge_margin_db, min_erle_db=barge_min_erle_db)
        self._recent: deque[bytes] = deque(maxlen=HISTORY_FRAMES)  # what a barge-in turn is started from
        self._probe: _Turn | None = None
        self._probe_quiet = self._probe_frames = 0
        self._probe_ready_at = 0.0
        self._probes_opened = 0
        self._held = False
        self._paused = False
        self._active_at = clock()
        self._local = threading.local()
        self._thread: threading.Thread | None = None
        self.failure: str | None = None
        self._io = assistant.io
        # with `act`, a request to use the machine runs as a dry run until the person says the phrase (approval.py)
        self._approval = (
            TaskApproval(confirm_phrase, confirm_timeout_s, clock=clock, show=lambda text: self._io.show(text))
            if act and confirm_tasks
            else None
        )
        self._chained_heard = heard = self._io.heard

        def on_heard(text: str) -> None:
            turn = getattr(self._local, "turn", None)
            if turn is not None:
                turn.heard.set()
                if self._approval is not None:
                    turn.confirmed = self._approval.answer(text)
                    if turn.confirmed is not None:
                        self._assistant.interrupt_speech()  # this turn is the yes, not a request: nothing is routed or answered
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

    @property
    def awaiting_confirmation(self) -> bool:
        """A previewed task is waiting for the person's yes (the phrase, or `confirm()`)."""
        return self._approval is not None and self._approval.waiting

    def confirm(self) -> bool:
        """The keyed yes: run the task that is waiting for one, for real. False when none is, or its time has run out."""
        pending = self._approval.confirm() if self._approval is not None else None
        if pending is None:
            return False
        turn = _Turn()
        turn.heard.set()
        self._start(turn, lambda _turn: self._run_confirmed(pending, standalone=True))
        return True

    @property
    def ended(self) -> bool:
        """The thread made by `start()` has finished: the loop stopped, or a microphone fault ended it."""
        return self._thread is not None and not self._thread.is_alive()

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
                if self._approval is not None:
                    self._approval.expire()
                if self._paused:
                    continue
                try:
                    frame = self._device.read()
                except DeviceFault as exc:
                    self._fail(str(exc))
                    return
                self._note_output_faults()
                if frame is not None:
                    try:
                        self._frame(frame)
                    except Exception as exc:  # the voice detector failing must not leave a dead thread and a live microphone
                        self._fail(f"Voice detection failed ({type(exc).__name__}); hands-free listening stopped.")
                        return
                self._check_idle()
        finally:
            self._abort()  # stopping is not an end of speech: a half-said command is discarded, never submitted
            if self._io.heard is self._on_heard:
                self._io.heard = self._chained_heard

    def _fail(self, message: str) -> None:
        """Listening cannot go on: the turn in progress is discarded, the person is told, and the microphone is turned off."""
        self._abort()
        self.failure = message
        self._assistant.io.warn(message)
        with contextlib.suppress(Exception):
            self._device.pause_input()

    def _note_output_faults(self) -> None:
        """The speaker had trouble (an underflow): what was said may have been cut or garbled. Said once, never counted aloud."""
        faults = self._device.output_faults
        if faults and not self._output_faults:
            self._assistant.io.warn("The speaker reported a problem; Glide's speech may have been cut off or garbled.")
        self._output_faults = faults

    def _pause(self) -> None:
        self._cancel_confirmation()
        self._end_turn()
        self._settle_gate()
        self._recent.clear()  # nothing from before the pause is the lead-in of what follows it
        self._skip_tail = False
        self._device.pause_input()
        self._paused = True

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
                self._pause()
            elif command == "resume" and self._paused:
                try:
                    self._device.resume_input()
                except Exception as exc:  # the microphone stays off, and the person is told
                    self._assistant.io.warn(f"the microphone could not be turned on ({type(exc).__name__})")
                    continue
                self._paused = False
                self._active_at = self._clock()

    def _frame(self, frame: bytes) -> None:
        probability = self._vad(frame)
        self._recent.append(frame)
        if self._turn is not None and self._device.guarding:
            # Glide began to speak over a command in half-duplex: the microphone now hands over zeros, which the turn would
            # read as the silence that ends it. Half a command is never sent.
            self._abort()
            self._assistant.io.warn("Glide began to speak over your command; it was not sent.")
            return
        if self._turn is None and self._speaking():
            self._gated(frame, probability)
        else:
            self._settle_gate()
            self._detect(frame, probability)

    def _detect(self, frame: bytes, probability: float) -> None:
        """Glide is not being heard, or a turn is running: the voice detector's probability decides where turns begin and end."""
        if self._skip_tail:  # until the silence that would have ended the utterance, nothing may begin a turn
            self._tail_quiet = self._tail_quiet + 1 if probability < STOP_PROBABILITY else 0
            self._skip_tail = self._tail_quiet < self._detector.silence_frames
            self._detector.feed(frame, 0.0)
            return
        try:
            started, data, commit = self._detector.feed(frame, probability)
            turn = self._turn
            if turn is None:
                if started:
                    self._start_turn(data)
                return
            self._extend(turn, frame, started, data, commit)
        except UtteranceTooLong as exc:
            self._abort()
            self._skip_tail = True
            self._tail_quiet = 0
            self._assistant.io.warn(str(exc))

    def _extend(self, turn: _Turn, frame: bytes, started: bool, data: bytes, commit: bool) -> None:
        """One more frame for the turn running: it goes on through a merge window, and ends on the silence that closes it."""
        if started:  # speech inside the merge window: the same turn goes on
            self._merging = 0
            turn.add(frame)
        elif self._merging:
            turn.add(frame)
        elif data:
            turn.add(data)
        if turn.frames > self._detector.max_frames:  # the detector only counts one stretch of speech: a turn is every stretch
            raise self._detector.too_long()
        if self._merging:
            self._merging -= 1
            if not self._merging:
                self._end_turn()
        elif commit:
            if self._merge_frames:
                self._merging = self._merge_frames
            else:
                self._end_turn()

    # -- barge-in -------------------------------------------------------------------------------

    def _speaking(self) -> bool:
        """Glide's voice is, or may still be, in the microphone."""
        return self._device.echo_active or self._device.playing

    def _gated(self, frame: bytes, probability: float) -> None:
        """A frame while Glide speaks and no turn is running: nothing starts a turn until the gate says it is the person."""
        verdict = self._gate.feed(probability, rms(frame), self._device.echo)
        self._detector.feed(frame, 0.0)  # keeps the lead-in current for when the speaking ends; never starts a turn here
        self._hold(self._gate.armed)
        if self._probe is not None:
            self._feed_probe(frame, verdict.weak)
        if verdict.confirm:
            self._barge_in()
        elif verdict.probe and self._probe is None:
            self._open_probe()

    def _settle_gate(self) -> None:
        """Glide is not speaking, or a turn is already running: whatever the gate was judging is over."""
        if self._gate.armed or self._probe is not None or self._held:
            self._gate.reset()
            self._hold(False)
            self._end_probe()

    def _hold(self, hold: bool) -> None:
        if hold != self._held:
            self._held = hold
            self._device.hold_echo_stats(hold)

    def _since_onset(self, extra: int = 0) -> bytes:
        """The frames from a lead-in before the sound being judged began up to the current one."""
        count = self._gate.onset_back + 1 + BARGE_LEAD_FRAMES + extra
        return b"".join(list(self._recent)[-count:])

    def _barge_in(self) -> None:
        audio = self._since_onset()
        self._gate.reset()
        self._hold(False)
        self._abandon_probe()
        self._interrupt()
        self._detector.feed(self._recent[-1], 1.0)  # the detector now runs the turn: this frame is its first
        self._begin(audio)

    def _interrupt(self) -> None:
        """Cut the voice and the answer being written. Pending requests are dropped too, unless an earlier turn of the
        person's is still waiting for its transcript: that one is wanted."""
        spare = any(not turn.heard.is_set() for turn in self._turns if not turn.probe)
        self._cancel_confirmation()  # a task waiting for its yes is not confirmed by a person who is now talking over Glide
        self._assistant.interrupt_speech(drop_pending=not spare)

    def _start_turn(self, first: bytes) -> None:
        """The voice detector began a turn with nothing of Glide's in the microphone. An answer the person has not heard
        yet (being routed or written, or its speech queued) must not start speaking over them: cut it first."""
        if self._answer_in_flight():
            self._interrupt()
        self._begin(first)

    def _answer_in_flight(self) -> bool:
        answering = any(t.heard.is_set() and t.thread is not None and t.thread.is_alive() for t in self._turns if not t.probe)
        return answering or not self._assistant.wait_idle(0.0)

    def _open_probe(self) -> None:
        now = self._clock()
        if now < self._probe_ready_at or any(t.probe and not t.heard.is_set() for t in self._turns):
            return  # one at a time: the last probe's transcript is still on its way
        self._probe = probe = _Turn(probe=True)
        self._probe_quiet = self._probe_frames = 0
        self._probes_opened += 1
        probe.audio.put(self._since_onset())
        self._start(probe)

    def _feed_probe(self, frame: bytes, evidence: bool) -> None:
        self._probe.audio.put(frame)
        self._probe_frames += 1
        self._probe_quiet = 0 if evidence else self._probe_quiet + 1
        if self._probe_quiet >= PROBE_QUIET_FRAMES or self._probe_frames >= PROBE_MAX_FRAMES:
            self._end_probe()

    def _end_probe(self) -> None:
        """Let the probe's transcript finish: if it is a stop, it acts on it; otherwise it is dropped unheard."""
        probe, self._probe = self._probe, None
        if probe is not None:
            probe.audio.put(None)
            self._probe_ready_at = self._clock() + PROBE_COOLDOWN_S

    def _abandon_probe(self) -> None:
        """The sound was the person: the ordinary turn takes over, and the probe's audio is thrown away."""
        probe, self._probe = self._probe, None
        if probe is not None:
            probe.audio.put(TurnAborted())

    def status(self) -> dict:
        """Counters and thresholds of the barge-in policy and the canceller's numbers: content-free, for a person to read.
        Never audio, never a word of what was said."""
        echo = self._device.echo
        return {
            "barge_in": {**self._gate.counters, "probes": self._probes_opened, "thresholds": self._gate.thresholds},
            "echo": None
            if echo is None
            else {
                "canceller": self._device.echo_name,
                "reference_underruns": self._device.reference_underruns,
                "measured": echo.measured,
                "erle_db": None if echo.erle_db is None else round(echo.erle_db, 1),
                "delay_ms": echo.delay_ms,
                "latency_ms": round(echo.latency_ms, 1),
                "far_rms": round(echo.far_rms),
                "out_rms": round(echo.out_rms),
                "expected_residual_rms": round(echo.expected_residual_rms),
            },
        }

    # -- turns ----------------------------------------------------------------------------------

    def _begin(self, first: bytes) -> None:
        self._active_at = self._clock()
        turn = self._turn = _Turn()
        turn.frames = 1  # the detector counts the frame that began the turn, not the pre-roll or the lead-in
        turn.audio.put(first)
        self._start(turn)

    def _start(self, turn: _Turn, target: Callable[[_Turn], None] | None = None) -> None:
        turn.thread = threading.Thread(target=target or self._work, args=(turn,), name="glide-voice-turn", daemon=True)
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
        self._gate.reset()
        self._hold(False)
        self._abandon_probe()
        self._cancel_confirmation()
        if turn is not None:
            turn.audio.put(TurnAborted())

    def _work(self, turn: _Turn) -> None:
        self._local.turn = turn
        try:
            # a request that is wanted for real waits for its yes: until then it is a dry run, which only says its first move
            reply = self._assistant.handle_audio(
                turn.chunks(), act=self._act and self._approval is None, wait=False, language=self._language, stop_only=turn.probe
            )
            if turn.confirmed is not None:
                self._run_confirmed(turn.confirmed)
            elif self._approval is not None and reply.route == "computer":
                self._offer(reply)
        except TurnAborted:
            pass  # already reported by whoever aborted it
        except Exception as exc:  # one bad turn must not end hands-free listening; the type is all that is said
            self._assistant.io.warn(f"a voice turn failed ({type(exc).__name__})")
        finally:
            turn.heard.set()

    def _offer(self, reply) -> None:
        """A request began a computer task, as a dry run: once it has ended and its description has been heard, the person
        may confirm it. A task that was stopped, failed or was not a dry run is not offered."""
        task = reply.task
        if task is None or not reply.heard or not task.wait(DRY_RUN_WAIT_S):
            return
        result = task.result
        if result is None or result.stopped or result.outcome != "dry run":
            return
        self._assistant.wait_idle(PREVIEW_HEARD_S)
        self._approval.open(reply.heard, reply.language)

    def _run_confirmed(self, pending: Pending, *, standalone: bool = False) -> None:
        """The yes was given: the same request again, now for real. It is routed again, and may be answered differently."""
        try:
            self._assistant.handle_text(pending.text, act=True, wait=False, hint_language=pending.language)
        except Exception as exc:
            if not standalone:
                raise
            self._assistant.io.warn(f"a voice turn failed ({type(exc).__name__})")

    def _cancel_confirmation(self) -> None:
        if self._approval is not None:
            self._approval.cancel()

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
            self._pause()  # the microphone is off before anyone is called: a callback that blocks or raises cannot leave it on
            if self._on_idle is not None:
                try:
                    self._on_idle()
                except Exception as exc:
                    self._assistant.io.warn(f"the idle callback failed ({type(exc).__name__})")

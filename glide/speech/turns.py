"""The hands-free loop: frames in, turns out, each turn handed to `Assistant.handle_audio`.

The loop owns exactly three decisions and nothing else (no transcription, no routing, no speech):

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

   The policy keeps counters and thresholds (`VoiceLoop.status()`): ERLE, delay, decisions. They are numbers,
   never audio or words, and are for the person to read, never to be spoken.

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
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass

from ..assistant.audio_io import rms
from .audio import DeviceFault
from .echo import EchoStats
from .settings import SpeechSettings
from .vad import FRAME_S, START_PROBABILITY, Probability, TurnDetector, UtteranceTooLong

POLL_S = 0.05

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

    def _close_candidate(self) -> None:
        if self._candidate:
            self.counters["rejected"] += 1
        self._candidate = False

    def feed(self, probability: float, level: float, echo: EchoStats | None) -> Verdict:
        self._frame_no += 1
        self.counters["frames"] += 1
        voiced = probability >= START_PROBABILITY
        strong = weak = suspect = voiced
        if voiced and echo is not None:
            if not echo.measured or (echo.erle_db or 0.0) < self.min_erle_db:
                strong = weak = suspect = False
                self.counters["waiting_for_erle"] += 1
            else:
                floor = max(echo.expected_residual_rms, MIN_LEVEL_RMS)
                kept = level >= RETAINED_MIN * echo.near_rms
                strong = kept and level >= floor * 10 ** (self.margin_db / 20)
                weak = kept and level >= floor * 10 ** (self.probe_margin_db / 20)
                suspect = level >= floor * SUSPECT_RATIO
        if weak:
            if self._sound_start is None or self._quiet_since >= SOUND_GAP_FRAMES:
                self._sound_start = self._frame_no
            self._quiet_since = 0
        else:
            self._quiet_since += 1
        self._suspect.feed(self._frame_no, suspect)
        self._strong.feed(self._frame_no, strong)
        self._weak.feed(self._frame_no, weak)
        if self._weak.ended_with:
            self._close_candidate()
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
        self.heard = threading.Event()  # the transcript is final (or there will be none): safe to interrupt again
        self.thread: threading.Thread | None = None
        self.probe = probe  # listens for a stop only: it never makes a request

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


class VoiceLoop:
    def __init__(
        self,
        assistant,
        device,
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
                self._settle_gate()
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
        probability = self._vad(frame)
        self._recent.append(frame)
        if self._turn is None and self._speaking():
            self._gated(frame, probability)
            return
        self._settle_gate()
        try:
            started, data, commit = self._detector.feed(frame, probability)
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

    # -- barge-in -------------------------------------------------------------------------------

    def _speaking(self) -> bool:
        """Glide's voice is, or may still be, in the microphone."""
        return bool(getattr(self._device, "echo_active", False)) or self._device.playing

    def _gated(self, frame: bytes, probability: float) -> None:
        """A frame while Glide speaks and no turn is running: nothing starts a turn until the gate says it is the person."""
        verdict = self._gate.feed(probability, rms(frame), getattr(self._device, "echo", None))
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
            freeze = getattr(self._device, "hold_echo_stats", None)
            if freeze is not None:
                freeze(hold)

    def _since_onset(self, extra: int = 0) -> bytes:
        """The frames from a lead-in before the sound being judged began up to the current one."""
        count = self._gate.onset_back + 1 + BARGE_LEAD_FRAMES + extra
        return b"".join(list(self._recent)[-count:])

    def _barge_in(self) -> None:
        audio = self._since_onset()
        self._gate.reset()
        self._hold(False)
        self._abandon_probe()
        # drop pending requests only when no earlier turn of the person's is still waiting for its transcript
        wanted = all(turn.heard.is_set() for turn in self._turns if not turn.probe)
        self._assistant.interrupt_speech(drop_pending=wanted)
        self._detector.feed(self._recent[-1], 1.0)  # the detector now runs the turn: this frame is its first
        self._begin(audio)

    def _open_probe(self) -> None:
        now = self._clock()
        if now < self._probe_ready_at:
            return
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
        echo = getattr(self._device, "echo", None)
        return {
            "barge_in": {**self._gate.counters, "probes": self._probes_opened, "thresholds": self._gate.thresholds},
            "echo": None
            if echo is None
            else {
                "canceller": getattr(self._device, "echo_name", None),
                "reference_underruns": getattr(self._device, "reference_underruns", 0),
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
        turn.audio.put(first)
        self._start(turn)

    def _start(self, turn: _Turn) -> None:
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
        self._gate.reset()
        self._hold(False)
        self._abandon_probe()
        if turn is not None:
            turn.audio.put(TurnAborted())

    def _work(self, turn: _Turn) -> None:
        self._local.turn = turn
        try:
            self._assistant.handle_audio(turn.chunks(), act=self._act, wait=False, language=self._language, stop_only=turn.probe)
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

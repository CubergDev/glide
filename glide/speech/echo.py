"""Echo cancellation: take Glide's own voice out of the microphone so that the person can talk over it through speakers.

The full-duplex device (audio.py) already knows exactly what it handed to the speaker, and that is the far-end
reference an echo canceller needs. Every canceller here has the same small shape:

    cleaned = canceller.process(near, far)   # one 512-sample frame each, 16 kHz mono 16-bit PCM, same length out
    canceller.reset()                        # forget the room (a new device, or the microphone was off)
    canceller.stats                          # content-free numbers: ERLE so far, the delay, the levels
    canceller.close()

`near` is what the microphone heard and `far` is what the speaker was playing at the same moment, taken in step
with it, so the echo in `near` arrives somewhat *after* the matching part of `far` (the delay, tens to a few
hundred milliseconds). The cleaned frame has the same length, and may lag the input by `latency_samples`.

Two implementations, chosen by `[speech] echo_canceller`:

- `webrtc`: the WebRTC audio-processing module (AEC3) through the `livekit` package (the `aec` extra). Its
  delay search, double-talk handling and residual suppression are the best of the candidates measured. The wrapper
  itself uses no numpy (frames are bytes), though the livekit package installs it.
- `nlms`: a partitioned-block frequency-domain adaptive filter in numpy (the `speech` extra), kept as the
  fallback with no further dependency. A foreground filter produces the output and a background filter adapts
  freely; the foreground is replaced only when the background is measurably better, so a burst of the person's
  own voice during adaptation (double talk) cannot drag the output off. It reaches roughly 15 to 25 dB of echo
  return loss enhancement after a few seconds; it covers an echo path of `BLOCK_SAMPLES * PARTITIONS` samples
  (384 ms, delay and room tail together), and a longer path is only partly cancelled.

`auto` takes `webrtc` when it imports, then `nlms`, and says which fallback it took; with neither, speaker mode
stays half duplex and says so. macOS VoiceProcessingIO is the native option for the SwiftUI app and is not built
here: see docs/voice-echo-cancellation.md.

What `stats` means. `erle_db` is the long-run echo return loss enhancement: how much quieter the cleaned signal
is than the microphone, over the frames where the speaker was sounding. It starts unmeasured and settles over
seconds, so a barge-in policy must not trust it earlier (`EchoStats.measured`). `expected_residual_rms` is the cleaned
level this frame would have if it held only echo: the larger of the reference's recent loudness times the 95th
percentile of residual-over-reference seen in the last 1.5 s, and the 90th percentile of the cleaned level itself over
the same time. High percentiles, not averages, because a canceller's residual comes in bursts and Glide's own echo
must never be louder than what is expected of it. The barge-in policy compares the cleaned level against it.
Statistics are frozen while `hold` is set, which the policy does while it is judging whether a sound is the person:
the person's own voice must not teach the canceller that echo is louder than it is.

Importing this module never loads numpy or livekit; building a canceller does, and fails with `EchoError`.
"""

from __future__ import annotations

import atexit
import gc
import math
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass

from ..assistant.audio_io import rms
from .vad import FRAME_BYTES, FRAME_S, SAMPLE_RATE

ECHO_CHOICES = ("auto", "none", "webrtc", "nlms")

FAR_ACTIVE_RMS = 40.0  # the speaker is sounding when the reference is at least this loud (16-bit scale)
ERLE_WINDOW_S = 1.0  # time constant of the long-run ERLE
ERLE_MIN_FRAMES = 30  # about one second of frames with echo possible before the ERLE is trusted
RATIO_WINDOW = 48  # frames of echo kept for the percentiles below: 1.5 s
RATIO_PERCENTILE = 0.95  # of cleaned level over recent reference level
LEVEL_PERCENTILE = 0.90  # of the cleaned level itself
PATH_FRAMES = 16  # how long Glide's voice can still be arriving after it was handed to the speaker: 512 ms
ENVELOPE_FRAMES = 64  # the delay is read off the loudness envelopes of the last two seconds
ENVELOPE_LAGS = 16
ENVELOPE_EVERY = 16
ENVELOPE_MIN_CORRELATION = 0.35


class EchoError(RuntimeError):
    """The canceller could not be set up or failed. The message never carries audio."""


@dataclass(frozen=True)
class EchoStats:
    """A content-free snapshot: levels and ratios only, never audio."""

    far_rms: float = 0.0
    far_recent_rms: float = 0.0  # the loudest reference within the room's memory (`PATH_FRAMES`): echo may still be coming
    near_rms: float = 0.0
    out_rms: float = 0.0
    expected_residual_rms: float = 0.0
    erle_db: float | None = None  # None until `measured`
    measured: bool = False
    delay_ms: float | None = None  # the echo path as paired by the device, to a frame (32 ms); None until seen
    latency_ms: float = 0.0  # how far the cleaned frames lag the microphone
    frames: int = 0


class EchoCanceller(ABC):
    name = "echo"
    latency_samples = 0

    def __init__(self) -> None:
        self.hold = False
        self._reset_stats()

    # -- the interface --------------------------------------------------------------------------

    def process(self, near: bytes, far: bytes) -> bytes:
        if len(near) != FRAME_BYTES or len(far) != FRAME_BYTES:
            raise ValueError("an echo canceller takes one 512-sample frame of each stream")
        out = self._process(near, far)
        self._track(near, far, out)
        return out

    def reset(self) -> None:
        self._reset_state()
        self._reset_stats()

    def close(self) -> None:
        """Release whatever the implementation holds. The canceller is not used again."""
        self.hold = False

    @property
    def stats(self) -> EchoStats:
        return self._stats

    # -- what an implementation provides --------------------------------------------------------

    @abstractmethod
    def _process(self, near: bytes, far: bytes) -> bytes: ...

    @abstractmethod
    def _reset_state(self) -> None: ...

    # -- statistics (shared, so every implementation is measured the same way) ------------------

    def _reset_stats(self) -> None:
        self._near_energy = self._out_energy = 0.0  # over the frames where echo may be in the microphone
        self._tracked = 0
        self._count = 0
        self._ratios: deque[float] = deque(maxlen=RATIO_WINDOW)  # cleaned level over recent reference level, per frame
        self._levels: deque[float] = deque(maxlen=RATIO_WINDOW)  # the cleaned level, per frame
        self._recent_far: deque[float] = deque(maxlen=PATH_FRAMES)
        self._envelope: deque[tuple[float, float]] = deque(maxlen=ENVELOPE_FRAMES)
        self._delay_ms: float | None = None
        self._stats = EchoStats(latency_ms=self.latency_samples * 1000 / SAMPLE_RATE)

    def _track(self, near: bytes, far: bytes, out: bytes) -> None:
        near_rms, far_rms, out_rms = rms(near), rms(far), rms(out)
        self._count += 1
        self._recent_far.append(far_rms)
        far_recent = max(self._recent_far)
        if far_recent >= FAR_ACTIVE_RMS and not self.hold:  # echo may be in the microphone: the room outlasts the sound
            self._learn(near_rms, out_rms, far_recent)
        measured = self._tracked >= ERLE_MIN_FRAMES
        self._envelope.append((far_rms, near_rms))
        if self._count % ENVELOPE_EVERY == 0:
            found = self._estimate_delay()
            self._delay_ms = self._delay_ms if found is None else found
        self._stats = EchoStats(
            far_rms=far_rms,
            far_recent_rms=far_recent,
            near_rms=near_rms,
            out_rms=out_rms,
            expected_residual_rms=self._expected_residual(far_recent) if measured else 0.0,
            erle_db=self._erle() if measured else None,
            measured=measured,
            delay_ms=self._delay_ms,
            latency_ms=self.latency_samples * 1000 / SAMPLE_RATE,
            frames=self._count,
        )

    def _learn(self, near_rms: float, out_rms: float, far_recent: float) -> None:
        """Take a frame in which echo may be present into the long-run energies and the recent ratios and levels."""
        weight = 1.0 if self._tracked == 0 else min(1.0, FRAME_S / ERLE_WINDOW_S)
        self._near_energy += weight * (near_rms**2 - self._near_energy)
        self._out_energy += weight * (out_rms**2 - self._out_energy)
        self._tracked += 1
        self._ratios.append(out_rms / far_recent)
        self._levels.append(out_rms)

    def _erle(self) -> float | None:
        if self._tracked and self._out_energy > 0 and self._near_energy > 0:
            return 10 * math.log10(max(self._near_energy / self._out_energy, 1e-3))
        return None

    def _expected_residual(self, far_recent: float) -> float:
        """The echo that should be left in this frame is the larger of two things, each a high percentile of what was seen
        lately and not an average, because the residual of a canceller comes in bursts and a bar at the average would be
        crossed by Glide's own echo: the reference's recent loudness (the room's echo outlasts the sound that made it) times
        the usual residual-over-reference ratio, which follows the speaker's loudness; and the cleaned level itself, which
        holds what does not follow it (a residual that hangs on while the reference dips). A path change shows within a few
        frames, since a handful of large values is all it takes to move a percentile. Statistics are frozen while a voice is
        being judged, so the voice does not teach them that echo is louder."""
        if not self._ratios:
            return 0.0
        ratios, levels = sorted(self._ratios), sorted(self._levels)
        return max(
            far_recent * ratios[-max(1, round(len(ratios) * (1 - RATIO_PERCENTILE)))],
            levels[min(len(levels) - 1, int(len(levels) * LEVEL_PERCENTILE))],
        )

    def _estimate_delay(self) -> float | None:
        """The lag, in whole frames, at which the microphone's loudness follows the reference's best."""
        if len(self._envelope) < ENVELOPE_FRAMES:
            return None
        far = [f for f, _ in self._envelope]
        near = [n for _, n in self._envelope]
        if sum(f >= FAR_ACTIVE_RMS for f in far) < ENVELOPE_FRAMES // 4:
            return None
        best, best_lag = ENVELOPE_MIN_CORRELATION, None
        for lag in range(ENVELOPE_LAGS):
            a, b = far[: len(far) - lag], near[lag:]
            ma, mb = sum(a) / len(a), sum(b) / len(b)
            va = sum((x - ma) ** 2 for x in a)
            vb = sum((y - mb) ** 2 for y in b)
            if va <= 0 or vb <= 0:
                continue
            corr = sum((x - ma) * (y - mb) for x, y in zip(a, b, strict=True)) / math.sqrt(va * vb)
            if corr > best:
                best, best_lag = corr, lag
        return None if best_lag is None else best_lag * FRAME_S * 1000


# -- WebRTC AEC3 ------------------------------------------------------------------------------------

BLOCK_SAMPLES_10MS = SAMPLE_RATE // 100  # the module works on exactly 10 ms
BLOCK_BYTES_10MS = BLOCK_SAMPLES_10MS * 2


def _drop(holder: list) -> None:
    """Let go of the native module now: its handle must be dropped while the library is still loaded, not at teardown."""
    if holder:
        holder.clear()
        gc.collect()


class WebRtcCanceller(EchoCanceller):
    """AEC3 from the WebRTC audio-processing module, through `livekit.rtc` (the `aec` extra).

    Only echo cancellation is on: no noise suppression, no gain control, no high-pass filter, so the cleaned
    signal is the microphone's with the echo taken out and nothing else changed. Frames of 512 samples are
    re-cut into the 10 ms blocks the module needs; the cleaned stream lags the microphone by `latency_samples`,
    measured against this module (the 10 ms prefill that keeps every output frame full plus the module's own
    block delay) and re-checked by a test.
    """

    name = "webrtc"
    latency_samples = 288

    def __init__(self) -> None:
        super().__init__()
        try:
            from livekit import rtc
        except ImportError as exc:
            raise EchoError("echo cancellation 'webrtc' needs the livekit package (the aec extra)") from exc
        self._rtc = rtc
        self._holder: list = []  # the module lives only here, so the exit hook can drop it
        self._registered = False
        self._reset_state()

    def _reset_state(self) -> None:
        self._release()
        try:
            apm = self._rtc.AudioProcessingModule(
                echo_cancellation=True, noise_suppression=False, high_pass_filter=False, auto_gain_control=False
            )
            apm.set_stream_delay_ms(0)  # AEC3 searches for the delay itself
            self._holder.append(apm)
            if not self._registered:  # after the library's own exit hook exists, so ours runs first (they run last in, first out)
                atexit.register(_drop, self._holder)
                self._registered = True
        except Exception as exc:
            raise EchoError(f"the WebRTC echo canceller could not start ({type(exc).__name__})") from exc
        self._near = bytearray()
        self._far = bytearray()
        self._ready = bytearray(BLOCK_BYTES_10MS)  # prefilled: an output frame is always complete

    def _process(self, near: bytes, far: bytes) -> bytes:
        rtc, apm = self._rtc, self._holder[0]
        self._near += near
        self._far += far
        produced = bytearray(self._ready)
        try:
            while len(self._near) >= BLOCK_BYTES_10MS:
                reference = rtc.AudioFrame(bytes(self._far[:BLOCK_BYTES_10MS]), SAMPLE_RATE, 1, BLOCK_SAMPLES_10MS)
                capture = rtc.AudioFrame(bytes(self._near[:BLOCK_BYTES_10MS]), SAMPLE_RATE, 1, BLOCK_SAMPLES_10MS)
                apm.process_reverse_stream(reference)
                apm.process_stream(capture)
                produced += bytes(capture.data)
                del self._near[:BLOCK_BYTES_10MS], self._far[:BLOCK_BYTES_10MS]
        except Exception as exc:
            raise EchoError(f"the WebRTC echo canceller failed ({type(exc).__name__})") from exc
        self._ready = produced[FRAME_BYTES:]
        return bytes(produced[:FRAME_BYTES])

    def _release(self) -> None:
        _drop(self._holder)

    def close(self) -> None:
        self._release()


# -- numpy partitioned-block frequency-domain filter ------------------------------------------------

BLOCK_SAMPLES = 128  # one adaptation step is 8 ms
PARTITIONS = 48  # 48 blocks of 128 samples: an echo path of up to 384 ms
STEP = 1.0  # background filter step size, normalised by the reference power over all partitions
SWITCH_DB = 1.5  # the background must beat the foreground by this much before the foreground takes it over
REVERT_RATIO = 4.0  # a background this much worse than the foreground is re-seeded from it
ENERGY_WINDOW = 0.9
REGULARISATION = 1.0  # per sample of the filter, in squared 16-bit units


class FrequencyDomainCanceller(EchoCanceller):
    """A two-path partitioned-block frequency-domain adaptive filter (overlap-save), numpy only."""

    name = "nlms"
    latency_samples = 0

    def __init__(self) -> None:
        super().__init__()
        try:
            import numpy as np
        except ImportError as exc:
            raise EchoError("echo cancellation 'nlms' needs numpy (the speech extra)") from exc
        self._np = np
        self._reset_state()

    def _reset_state(self) -> None:
        np = self._np
        shape = (PARTITIONS, BLOCK_SAMPLES + 1)
        self._foreground = np.zeros(shape, dtype=np.complex128)
        self._background = np.zeros(shape, dtype=np.complex128)
        self._history = np.zeros(shape, dtype=np.complex128)  # the reference's spectra, newest first
        self._previous = np.zeros(BLOCK_SAMPLES)
        self._power = np.zeros(BLOCK_SAMPLES + 1)
        self._foreground_energy = self._background_energy = 1e-6

    def _process(self, near: bytes, far: bytes) -> bytes:
        np = self._np
        near_samples = np.frombuffer(near, dtype="<i2").astype(np.float64)
        far_samples = np.frombuffer(far, dtype="<i2").astype(np.float64)
        out = np.empty_like(near_samples)
        for start in range(0, len(near_samples), BLOCK_SAMPLES):
            stop = start + BLOCK_SAMPLES
            out[start:stop] = self._block(near_samples[start:stop], far_samples[start:stop])
        if not np.all(np.isfinite(out)):  # a filter that blew up is thrown away, and this frame passes through
            self._reset_state()
            return near
        return np.clip(np.rint(out), -32768, 32767).astype("<i2").tobytes()

    def _block(self, near, far):
        np = self._np
        B = BLOCK_SAMPLES
        self._history = np.roll(self._history, 1, axis=0)
        self._history[0] = np.fft.rfft(np.concatenate((self._previous, far)))
        self._previous = far
        foreground = near - np.fft.irfft(np.sum(self._foreground * self._history, axis=0))[B:]
        if not np.any(far):
            return foreground  # nothing is playing: nothing to learn from, and nothing to take out
        background = near - np.fft.irfft(np.sum(self._background * self._history, axis=0))[B:]
        spectra = np.abs(self._history) ** 2
        self._power = 0.8 * self._power + 0.2 * spectra[0]
        norm = np.maximum(spectra.sum(axis=0), self._power * PARTITIONS) + REGULARISATION * PARTITIONS * B
        error = np.fft.rfft(np.concatenate((np.zeros(B), background)))
        gradient = np.fft.irfft(np.conj(self._history) * (error / norm)[None, :], axis=1)
        gradient[:, B:] = 0
        self._background += STEP * np.fft.rfft(gradient, axis=1)
        keep = ENERGY_WINDOW
        self._foreground_energy = keep * self._foreground_energy + (1 - keep) * float(np.sum(foreground**2))
        self._background_energy = keep * self._background_energy + (1 - keep) * float(np.sum(background**2))
        if self._foreground_energy > 10 ** (SWITCH_DB / 10) * self._background_energy:
            self._foreground = self._background.copy()
            self._foreground_energy = self._background_energy
        elif self._background_energy > REVERT_RATIO * self._foreground_energy:
            self._background = self._foreground.copy()
            self._background_energy = self._foreground_energy
        return foreground


CANCELLERS: dict[str, type[EchoCanceller]] = {"webrtc": WebRtcCanceller, "nlms": FrequencyDomainCanceller}


def make_canceller(name: str, warn=lambda message: None) -> EchoCanceller | None:
    """The canceller `[speech] echo_canceller` asks for, or None for `none`.

    A named canceller that cannot be built is an error. `auto` tries them in order and says which fallback it took,
    or that there is none (speaker mode is then half duplex, and nothing can interrupt Glide by voice).
    """
    if name == "none":
        return None
    if name != "auto":
        return CANCELLERS[name]()
    failures = []
    for choice, cls in CANCELLERS.items():
        try:
            canceller = cls()
        except EchoError as exc:
            failures.append(str(exc))
            continue
        if failures:
            warn(f"echo cancellation falls back to '{choice}': {failures[0]}")
        return canceller
    warn(
        "no echo canceller is available, so speaker mode is half duplex (install the aec or speech extra): " + "; ".join(failures)
    )
    return None

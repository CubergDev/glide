"""Microphone capture and speaker playback, with the sound card behind a small swappable backend.

Audio is raw PCM, 16-bit little-endian, mono (providers/base.py). `Player` plays what it is given on a
thread of its own and can be cut off at any moment; `Microphone` hands out about 100 ms of audio at a
time so a transcriber can work while the person is still speaking; `Endpointer` decides from energy
alone when an utterance is over. None of them knows what a sound card is: each takes a backend, and
tests give it a fake. The real backends use the `sounddevice` package, imported only when one is built,
so this module imports fine (and the tests run) on a machine that does not have it.

Nothing here opens a device unless a real backend is constructed.
"""

from __future__ import annotations

import contextlib
import math
import queue
import sys
import threading
import time
from array import array
from collections.abc import Callable, Iterator
from operator import mul
from typing import Protocol

SAMPLE_RATE = 16000  # what the transcribers are given
CHUNK_MS = 100  # how much audio one capture chunk holds: small enough to stream, large enough to be cheap
SAMPLE_BYTES = 2  # 16-bit mono
PLAY_BLOCK_MS = 50  # playback is written in pieces this long, so a cancel never waits for more than one of them
MAX_RECORDING_S = 60.0


class AudioUnavailable(RuntimeError):
    """There is no way to reach a microphone or speaker: the `sounddevice` package or PortAudio is missing."""


def load_sounddevice():
    """The `sounddevice` module, imported on first use. It needs PortAudio, so a missing library is an OSError."""
    try:
        import sounddevice
    except (ImportError, OSError) as exc:
        raise AudioUnavailable(
            f"audio needs the 'sounddevice' package and PortAudio ({type(exc).__name__}); install it with: uv add sounddevice"
        ) from exc
    return sounddevice


# -- Backends ---------------------------------------------------------------------------------------


class OutputBackend(Protocol):
    """A speaker. `write` and `flush` are called from the player's thread only; `abort` from any thread."""

    def write(self, pcm: bytes, sample_rate: int) -> None:
        """Queue audio and return when the device has room for more. The rate may change between calls."""
        ...

    def abort(self) -> None:
        """Throw away whatever has not been played and make `write` return now."""
        ...

    def flush(self) -> None:
        """Block until everything written has been played."""
        ...

    def close(self) -> None: ...


class InputBackend(Protocol):
    def start(self, sample_rate: int, frames: int) -> None: ...

    def read(self) -> bytes:
        """Block until `frames` frames have been captured and return them; empty means the device is gone."""
        ...

    def stop(self) -> None: ...


class SoundDeviceOutput:
    """The default speaker, through `sounddevice`. One output stream per sample rate, reopened when it changes."""

    def __init__(self, device: int | str | None = None) -> None:
        self._sd = load_sounddevice()
        self._device = device
        self._stream = None
        self._rate: int | None = None

    def write(self, pcm: bytes, sample_rate: int) -> None:
        self._ready(sample_rate).write(pcm)

    def abort(self) -> None:
        stream = self._stream
        if stream is not None:
            stream.abort()  # unlike stop(), this discards the buffered audio instead of playing it out

    def flush(self) -> None:
        stream = self._stream
        if stream is not None and stream.active:
            stream.stop()  # waits until the pending buffers have been played; the next write starts it again

    def close(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            with contextlib.suppress(Exception):
                stream.abort()
                stream.close()

    def _ready(self, rate: int):
        stream = self._stream
        if stream is not None and self._rate != rate:
            stream.stop()  # the sentence before this one finishes at its own rate
            stream.close()
            stream = self._stream = None
        if stream is None:
            stream = self._sd.RawOutputStream(samplerate=rate, channels=1, dtype="int16", device=self._device)
            self._stream, self._rate = stream, rate
        if not stream.active:
            stream.start()
        return stream


class SoundDeviceInput:
    """The default microphone, through `sounddevice`: blocking reads of fixed-size blocks, copied out as bytes."""

    def __init__(self, device: int | str | None = None) -> None:
        self._sd = load_sounddevice()
        self._device = device
        self._stream = None
        self._frames = 0

    def start(self, sample_rate: int, frames: int) -> None:
        self._frames = frames
        self._stream = self._sd.RawInputStream(
            samplerate=sample_rate, channels=1, dtype="int16", blocksize=frames, device=self._device
        )
        self._stream.start()

    def read(self) -> bytes:
        data, _overflowed = self._stream.read(self._frames)  # an overflow means a dropped block; the speech goes on
        return bytes(data)

    def stop(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            with contextlib.suppress(Exception):
                stream.abort()
                stream.close()


# -- Playback ---------------------------------------------------------------------------------------

_CLOSE = object()


class _Flush:
    """A marker behind the audio: when the player reaches it, everything before it has been written."""

    def __init__(self, epoch: int) -> None:
        self.epoch = epoch


class Player:
    """Plays chunks of speech on its own thread and stops the instant it is told to.

    `play` only queues, so a caller streaming speech never waits on the sound card. Each queued chunk
    carries the epoch it was queued in; `cancel` bumps the epoch, empties the queue and aborts the
    device, and the thread discards anything from an older epoch. Audio is written in `PLAY_BLOCK_MS`
    pieces and the epoch is checked between them, so even a backend that cannot abort stops within one
    piece. A chunk queued after a cancel is played in full.
    """

    def __init__(
        self,
        backend: OutputBackend | None = None,
        *,
        block_ms: int = PLAY_BLOCK_MS,
        on_error: Callable[[BaseException], None] | None = None,
    ) -> None:
        self._backend = backend if backend is not None else SoundDeviceOutput()
        self._block_ms = block_ms
        self._on_error = on_error
        self._lock = threading.RLock()
        self._cond = threading.Condition(self._lock)
        self._items: queue.Queue = queue.Queue()
        self._epoch = 0
        self._busy = 0  # chunks and flush markers queued or being written
        self._thread: threading.Thread | None = None
        self._closed = False

    def __repr__(self) -> str:
        return f"<Player busy={self._busy}>"

    @property
    def busy(self) -> bool:
        return self._busy > 0

    def play(self, pcm: bytes, sample_rate: int) -> None:
        """Queue one chunk. Returns at once."""
        pcm = pcm[: len(pcm) - len(pcm) % SAMPLE_BYTES]  # a whole number of samples: half a sample plays as noise
        if not pcm:
            return
        with self._cond:
            if self._closed:
                return
            self._busy += 1
            self._items.put((self._epoch, pcm, sample_rate))
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="glide-player", daemon=True)
                self._thread.start()

    def cancel(self) -> None:
        """Silence now: queued audio is dropped and the device is aborted. Safe from any thread, at any time."""
        with self._cond:
            self._epoch += 1
            self._drain()
            try:
                self._backend.abort()
            except Exception as exc:  # the audio is already dropped; say so, but the cancel itself has happened
                self._report(exc)
            self._cond.notify_all()

    def wait_idle(self, timeout: float | None = None) -> bool:
        """Block until everything queued has been played out. False if `timeout` ran out first."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            if self._thread is None:
                return True
            self._busy += 1
            self._items.put(_Flush(self._epoch))
            while self._busy > 0:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._cond.wait(remaining)
        return True

    def close(self) -> None:
        with self._cond:
            if self._closed:
                return
            self._closed = True
            self._epoch += 1
            self._drain()
            with contextlib.suppress(Exception):
                self._backend.abort()
            self._items.put(_CLOSE)
            thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
        with contextlib.suppress(Exception):
            self._backend.close()

    # -- the player's thread --------------------------------------------------------------------

    def _drain(self) -> None:
        """Empty the queue. Called with the lock held."""
        close_requested = False
        while True:
            try:
                item = self._items.get_nowait()
            except queue.Empty:
                break
            if item is _CLOSE:
                close_requested = True
            else:
                self._busy -= 1
        if close_requested:
            self._items.put(_CLOSE)

    def _run(self) -> None:
        while True:
            item = self._items.get()
            if item is _CLOSE:
                return
            try:
                if isinstance(item, _Flush):
                    if item.epoch == self._epoch:
                        self._backend.flush()
                else:
                    epoch, pcm, rate = item
                    self._write(epoch, pcm, rate)
            except Exception as exc:
                if (item.epoch if isinstance(item, _Flush) else item[0]) == self._epoch:
                    self._report(exc)  # an error after a cancel is the abort itself, not a fault
            finally:
                with self._cond:
                    self._busy -= 1
                    self._cond.notify_all()

    def _write(self, epoch: int, pcm: bytes, rate: int) -> None:
        step = max(SAMPLE_BYTES, rate * self._block_ms // 1000 * SAMPLE_BYTES)
        for start in range(0, len(pcm), step):
            if epoch != self._epoch:
                return
            self._backend.write(pcm[start : start + step], rate)

    def _report(self, exc: BaseException) -> None:
        if self._on_error is not None:
            with contextlib.suppress(Exception):
                self._on_error(exc)


# -- Capture ----------------------------------------------------------------------------------------


def rms(chunk: bytes) -> float:
    """The loudness of 16-bit mono PCM: the root mean square of its samples, 0 for silence, 32767 at most."""
    samples = array("h")
    samples.frombytes(chunk[: len(chunk) - len(chunk) % SAMPLE_BYTES])
    if sys.byteorder == "big":
        samples.byteswap()
    if not samples:
        return 0.0
    return math.sqrt(sum(map(mul, samples, samples)) / len(samples))


def chunked(pcm: bytes, *, sample_rate: int = SAMPLE_RATE, chunk_ms: int = CHUNK_MS) -> Iterator[bytes]:
    """PCM cut into capture-sized chunks, the way a `Microphone` would have delivered it."""
    size = sample_rate * chunk_ms // 1000 * SAMPLE_BYTES
    for start in range(0, len(pcm), size):
        yield pcm[start : start + size]


class NoiseFloor:
    """Loudness against a floor that follows the room: a level is speech when it is above `threshold` and above
    `ratio` times the running average of the levels that were not (`learn`), so a noisy room raises the bar by itself.
    Shared by `Endpointer` (push to talk) and `EnergyProbability` (hands free)."""

    def __init__(self, *, threshold: float = 500.0, ratio: float = 3.0, floor_weight: float = 0.1) -> None:
        self.threshold = threshold
        self.ratio = ratio
        self.floor_weight = floor_weight
        self.floor = 0.0
        self._measured = False

    def is_speech(self, level: float) -> bool:
        return level > max(self.threshold, self.floor * self.ratio)

    def learn(self, level: float) -> None:
        """Fold the level of a chunk that was not speech into the floor (the first one is the floor)."""
        weight = self.floor_weight if self._measured else 1.0
        self.floor = (1 - weight) * self.floor + weight * level
        self._measured = True

    def reset(self) -> None:
        self.floor = 0.0
        self._measured = False


class Endpointer:
    """Decides from loudness alone when an utterance is over. Feed it each captured chunk in turn.

    Speech is a stretch of at least `min_speech_s` louder than `max(threshold, ratio * noise floor)`,
    where the floor is a running average of the chunks that were not speech, so a noisy room raises the
    bar by itself. The utterance ends `silence_s` after the last loud chunk. If nobody speaks within
    `no_speech_s`, or the recording reaches `max_s`, it ends anyway; `reason` says which of
    "silence", "no speech" or "max" it was, and `heard` whether anything was said.
    """

    def __init__(
        self,
        *,
        sample_rate: int = SAMPLE_RATE,
        threshold: float = 500.0,
        ratio: float = 3.0,
        silence_s: float = 0.8,
        min_speech_s: float = 0.2,
        no_speech_s: float = 6.0,
        max_s: float = 30.0,
        floor_weight: float = 0.1,
    ) -> None:
        self.sample_rate = sample_rate
        self.silence_s = silence_s
        self.min_speech_s = min_speech_s
        self.no_speech_s = no_speech_s
        self.max_s = max_s
        self.noise = NoiseFloor(threshold=threshold, ratio=ratio, floor_weight=floor_weight)
        self.heard = False
        self.reason: str | None = None
        self._samples = 0  # counted in samples, not seconds: ten chunks of 0.1 s must add up to exactly 1 s
        self._voiced = 0  # samples of speech in the current run
        self._quiet = 0  # samples since the last speech

    @property
    def elapsed(self) -> float:
        return self._samples / self.sample_rate

    def _enough(self, seconds: float, samples: int) -> bool:
        return samples >= round(seconds * self.sample_rate)

    @property
    def floor(self) -> float:
        return self.noise.floor

    def is_speech(self, level: float) -> bool:
        return self.noise.is_speech(level)

    def feed(self, chunk: bytes) -> bool:
        """Take the next chunk. True once the utterance is over (see `reason`)."""
        samples = len(chunk) // SAMPLE_BYTES
        self._samples += samples
        level = rms(chunk)
        if self.is_speech(level):
            self._voiced += samples
            self._quiet = 0
            if self._enough(self.min_speech_s, self._voiced):
                self.heard = True
        else:
            self._voiced = 0
            self._quiet += samples
            self.noise.learn(level)
        if self.heard and self._enough(self.silence_s, self._quiet):
            self.reason = "silence"
        elif not self.heard and self._enough(self.no_speech_s, self._samples):
            self.reason = "no speech"
        elif self._enough(self.max_s, self._samples):
            self.reason = "max"
        return self.reason is not None


class Microphone:
    """Captures speech in chunks of about 100 ms, ready to feed to a transcriber while it is still being said."""

    def __init__(self, backend: InputBackend | None = None, *, sample_rate: int = SAMPLE_RATE, chunk_ms: int = CHUNK_MS) -> None:
        self._backend = backend if backend is not None else SoundDeviceInput()
        self.sample_rate = sample_rate
        self.chunk_ms = chunk_ms

    @property
    def chunk_bytes(self) -> int:
        return self.sample_rate * self.chunk_ms // 1000 * SAMPLE_BYTES

    def record(
        self, stop: threading.Event | None = None, *, endpointer: Endpointer | None = None, max_s: float = MAX_RECORDING_S
    ) -> Iterator[bytes]:
        """Chunks of PCM until `stop` is set, the endpointer says the utterance is over, or `max_s` has passed.

        The device is opened when the first chunk is asked for and always closed again, also when the
        consumer gives up early. Ending the iterator is what tells a transcriber that the utterance is
        complete, so the caller sets `stop` rather than abandoning it.
        """
        frames = self.sample_rate * self.chunk_ms // 1000
        self._backend.start(self.sample_rate, frames)
        try:
            captured = 0.0
            while not (stop is not None and stop.is_set()) and captured < max_s:
                chunk = self._backend.read()
                if not chunk:
                    return
                captured += len(chunk) / (SAMPLE_BYTES * self.sample_rate)
                yield chunk
                if endpointer is not None and endpointer.feed(chunk):
                    return
        finally:
            self._backend.stop()

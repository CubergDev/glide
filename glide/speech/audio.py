"""One microphone and one speaker as a single full-duplex device, with the echo guard and the headset mode.

Both streams are callback-driven, and the callbacks run on the sound card's threads, so they only move bytes
between locked buffers and never call the assistant. Everything else talks to the device through four small
surfaces:

- `read()` gives the loop one 32 ms frame (512 samples, 16 kHz, mono, 16-bit) at a time, the size the voice
  detector needs, however the sound card chunks its callbacks.
- `play`, `cancel`, `wait_idle` and `close` are the shape `Speaker` expects of a player (assistant/audio_io.py),
  so `IO(player=device)` makes the assistant speak through this device.
- `pause_input` and `resume_input` stop and restart the microphone itself, not just its use (idle handling).
- `headset` says whether the microphone may stay open while Glide speaks.

The echo guard. With speakers the microphone hears Glide, and a voice detector cannot tell that from a person.
So in speaker mode (the default) every microphone frame is replaced with silence while audio is queued, while
it plays, and for `echo_tail_s` after it ends so the room's reverberation dies away. Frames that were captured
but not yet read when speech begins are dropped. That makes speaker mode half duplex: **you cannot interrupt
Glide by voice through speakers**, because nothing you say is heard until it has finished. With a headset
(`headset = true`) the microphone is not guarded and talking over Glide is heard, which is what barge-in needs.
Whether the guard is on is decided from the device's own state (audio queued or playing), never from a flag a
stale caller could clear, so a late "finished" from an old utterance cannot unlock the microphone mid-sentence.

Nothing here opens a device until `start()`, and `start()` builds its streams from factories that tests replace.
"""

from __future__ import annotations

import contextlib
import queue
import threading
import time
from array import array
from collections import deque
from collections.abc import Callable
from typing import Protocol

from ..assistant.audio_io import AudioUnavailable, _sounddevice
from .vad import FRAME_BYTES, FRAME_SAMPLES, SAMPLE_RATE

SAMPLE_BYTES = 2
MAX_QUEUED_FRAMES = 100  # about 3 s of unread microphone audio: more than that means the loop has stopped reading
READ_POLL_S = 0.05


class DeviceFault(RuntimeError):
    """The sound card reported a problem (an overflow, a lost device). The turn in progress is not trustworthy."""


class Stream(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def close(self) -> None: ...


InputFactory = Callable[[Callable[[bytes, bool], None]], Stream]  # (on_input) -> an input stream, not yet started
OutputFactory = Callable[[Callable[[int], bytes], int], Stream]  # (on_output, sample_rate) -> an output stream


def resample(pcm: bytes, source_rate: int, target_rate: int) -> bytes:
    """16-bit mono PCM at another rate, by linear interpolation. Whole samples only; a stray odd byte is dropped.

    Good enough for speech heard through a speaker, and it needs no package. Each chunk is converted on its
    own, which can leave a faint tick at a seam; a TTS that can produce `target_rate` itself avoids it.
    """
    pcm = pcm[: len(pcm) - len(pcm) % SAMPLE_BYTES]
    if source_rate == target_rate or not pcm:
        return pcm
    samples = array("h")
    samples.frombytes(pcm)
    count = len(samples)
    produced = max(1, round(count * target_rate / source_rate))
    step = source_rate / target_rate
    out = array("h")
    for i in range(produced):
        position = i * step
        left = min(int(position), count - 1)
        right = min(left + 1, count - 1)
        fraction = position - left
        out.append(round(samples[left] * (1 - fraction) + samples[right] * fraction))
    return out.tobytes()


def sounddevice_factories(input_device=None, output_device=None) -> tuple[InputFactory, OutputFactory]:
    """Stream factories over `sounddevice`. Raises `AudioUnavailable` when it or PortAudio is missing."""
    sd = _sounddevice()

    def make_input(on_input: Callable[[bytes, bool], None]) -> Stream:
        def callback(indata, frames, timing, status) -> None:
            on_input(bytes(indata), bool(status))

        return sd.RawInputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=FRAME_SAMPLES, device=input_device, callback=callback
        )

    def make_output(on_output: Callable[[int], bytes], rate: int) -> Stream:
        def callback(outdata, frames, timing, status) -> None:
            outdata[:] = on_output(len(outdata))

        return sd.RawOutputStream(
            samplerate=rate, channels=1, dtype="int16", blocksize=FRAME_SAMPLES, device=output_device, callback=callback
        )

    return make_input, make_output


class FullDuplexDevice:
    def __init__(
        self,
        *,
        output_rate: int = 24000,
        headset: bool = False,
        echo_tail_s: float = 0.3,
        clock: Callable[[], float] = time.monotonic,
        input_factory: InputFactory | None = None,
        output_factory: OutputFactory | None = None,
        input_device: int | str | None = None,
        output_device: int | str | None = None,
    ) -> None:
        self.output_rate = output_rate
        self.headset = headset
        self.echo_tail_s = echo_tail_s
        self._clock = clock
        self._input_factory = input_factory
        self._output_factory = output_factory
        self._devices = (input_device, output_device)
        self._cond = threading.Condition(threading.Lock())
        self._frames: queue.Queue[bytes] = queue.Queue(maxsize=MAX_QUEUED_FRAMES)
        self._carry = b""
        self._output: deque[bytes] = deque()
        self._queued = 0  # bytes waiting to be played
        self._playing_until = 0.0  # the device is still sounding the last block it was handed until then
        self._quiet_until = 0.0  # the echo guard holds until then
        self._fault: str | None = None
        self._paused = False
        self._closed = False
        self._input: Stream | None = None
        self._output_stream: Stream | None = None

    def __repr__(self) -> str:
        return f"<FullDuplexDevice headset={self.headset} playing={self.playing} paused={self._paused}>"

    # -- state ----------------------------------------------------------------------------------

    @property
    def echo_guard(self) -> bool:
        return not self.headset

    @property
    def playing(self) -> bool:
        """Audio is queued or the device is still sounding the last of it."""
        with self._cond:
            return self._playing_locked()

    def _playing_locked(self) -> bool:
        return self._queued > 0 or self._clock() < self._playing_until

    def _guarded_locked(self) -> bool:
        return self.echo_guard and (self._queued > 0 or self._clock() < self._quiet_until)

    @property
    def input_paused(self) -> bool:
        return self._paused

    @property
    def fault(self) -> str | None:
        return self._fault

    # -- lifecycle ------------------------------------------------------------------------------

    def start(self) -> None:
        """Open the speaker and then the microphone. On any failure everything opened is closed again."""
        if self._input_factory is None or self._output_factory is None:
            make_input, make_output = sounddevice_factories(*self._devices)
            self._input_factory = self._input_factory or make_input
            self._output_factory = self._output_factory or make_output
        try:
            self._output_stream = self._output_factory(self._on_output, self.output_rate)
            self._output_stream.start()
            if not self._paused:
                self._input = self._input_factory(self._on_input)
                self._input.start()
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        with self._cond:
            if self._closed:
                return
            self._closed = True
            self._clear_output_locked()
            self._paused = True
            self._cond.notify_all()
        for stream in (self._input, self._output_stream):
            _shut(stream)
        self._input = self._output_stream = None

    # -- microphone -----------------------------------------------------------------------------

    def _on_input(self, data: bytes, fault: bool = False) -> None:
        """The sound card's input callback: reframe, guard against echo, queue. Never blocks."""
        with self._cond:
            if self._paused or self._closed:
                return
            if fault:
                self._fault = "Microphone overflow; the incomplete command was discarded."
                self._carry = b""
                return
            guarded = self._guarded_locked()
            buffer = self._carry + data
            while len(buffer) >= FRAME_BYTES:
                frame, buffer = buffer[:FRAME_BYTES], buffer[FRAME_BYTES:]
                try:
                    self._frames.put_nowait(bytes(FRAME_BYTES) if guarded else frame)
                except queue.Full:
                    self._fault = "Microphone overflow; the incomplete command was discarded."
                    buffer = b""
                    break
            self._carry = buffer

    def read(self, timeout: float | None = READ_POLL_S) -> bytes | None:
        """The next frame, or None if none came within `timeout`. Raises `DeviceFault` if the card reported one."""
        if self._fault is not None:
            raise DeviceFault(self._fault)
        try:
            frame = self._frames.get(timeout=timeout) if timeout else self._frames.get_nowait()
        except queue.Empty:
            if self._fault is not None:
                raise DeviceFault(self._fault) from None
            return None
        return frame

    def _drain_frames(self) -> None:
        while True:
            try:
                self._frames.get_nowait()
            except queue.Empty:
                break
        self._carry = b""

    def pause_input(self) -> None:
        """Turn the microphone itself off. Frames already captured are dropped, and callbacks that still arrive are ignored."""
        with self._cond:
            if self._paused:
                return
            self._paused = True
            self._drain_frames()
        stream, self._input = self._input, None
        _shut(stream)

    def resume_input(self) -> None:
        """Turn the microphone back on, with nothing stale in the queue. A failure leaves it paused and raises."""
        with self._cond:
            if not self._paused or self._closed:
                return
            self._drain_frames()
            self._fault = None
        if self._input_factory is None:
            self._paused = False
            return
        stream = self._input_factory(self._on_input)
        try:
            stream.start()
        except Exception:
            _shut(stream)
            raise
        with self._cond:
            self._input = stream
            self._paused = False

    # -- speaker (the shape `Speaker` expects of a player) --------------------------------------

    def play(self, pcm: bytes, sample_rate: int) -> None:
        """Queue one chunk of speech and return at once. Audio at another rate is converted to the device's."""
        data = resample(pcm, sample_rate, self.output_rate)
        if not data:
            return
        with self._cond:
            if self._closed:
                return
            if self.echo_guard and self._queued == 0 and not self._playing_locked():
                self._drain_frames()  # speech begins: whatever the microphone caught before it is not for the loop
            for start in range(0, len(data), 4096):
                piece = data[start : start + 4096]
                self._output.append(piece)
                self._queued += len(piece)

    def _on_output(self, size: int) -> bytes:
        """The sound card's output callback: the next `size` bytes, padded with silence. Never blocks."""
        block = bytearray(size)
        with self._cond:
            offset = 0
            while self._output and offset < size:
                piece = self._output.popleft()
                take = min(len(piece), size - offset)
                block[offset : offset + take] = piece[:take]
                if take < len(piece):
                    self._output.appendleft(piece[take:])
                self._queued -= take
                offset += take
            if offset:
                now = self._clock()
                self._playing_until = now + offset / SAMPLE_BYTES / self.output_rate
                self._quiet_until = self._playing_until + self.echo_tail_s
            if self._queued == 0:
                self._cond.notify_all()
        return bytes(block)

    def _clear_output_locked(self) -> None:
        self._output.clear()
        self._queued = 0
        self._cond.notify_all()

    def cancel(self) -> None:
        """Silence now: queued audio is dropped. The echo guard stays up for the tail, since what the device already
        has will still be heard for a moment."""
        with self._cond:
            was_sounding = self._playing_locked()
            self._clear_output_locked()
            if was_sounding and self.echo_guard:
                self._quiet_until = max(self._quiet_until, self._clock() + self.echo_tail_s)
            self._playing_until = min(self._playing_until, self._clock())

    def wait_idle(self, timeout: float | None = None) -> bool:
        """Block until everything queued has been handed to the sound card. False if `timeout` ran out first."""
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cond:
            while self._queued > 0 and not self._closed:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._cond.wait(remaining)
        return True


def _shut(stream: Stream | None) -> None:
    if stream is None:
        return
    for name in ("stop", "close"):
        with contextlib.suppress(Exception):  # a device that is already gone cannot be stopped, and has nothing left to release
            getattr(stream, name)()


__all__ = ["AudioUnavailable", "DeviceFault", "FullDuplexDevice", "resample", "sounddevice_factories"]

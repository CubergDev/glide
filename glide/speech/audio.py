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
Two ways to deal with it:

- With an echo canceller (`canceller=`, see echo.py) the device keeps the microphone open while Glide speaks.
  It records what the speaker was handed (the far-end reference) next to every microphone frame, and `read()`
  returns the frame with the echo taken out. The person can then be heard over Glide: that is barge-in by
  voice through speakers. The room's echo tail is the canceller's job, so nothing is blanked and nothing the
  person says right after Glide stops is lost to a tail.
- Without one, in speaker mode (the default when no canceller can be built), every microphone frame is
  replaced with silence while audio is queued, while it plays, and for `echo_tail_s` after it ends. Frames that
  were captured but not yet read when speech begins are dropped. That makes speaker mode half duplex: nothing
  said over Glide is heard until it has finished.

With a headset (`headset = true`) the microphone is not guarded and there is no canceller: talking over Glide
is heard as it is. Whether the guard is on is decided from the device's own state (audio queued or playing),
never from a flag a stale caller could clear, so a late "finished" from an old utterance cannot unlock the
microphone mid-sentence.

The reference is cut on the callbacks' threads like everything else there: bytes only. The output callback
appends exactly what it handed the card (silence included), converted to 16 kHz by a resampler that keeps its place
from block to block (no seams), and numbers every sample from the start. Each microphone frame is numbered the
same way and paired with the reference samples at its own numbers, shifted by the count the speaker had reached
when the first microphone frame arrived. That makes the reference never older than the microphone, so the echo
in a frame always lags it, which is the one thing a canceller cannot work without; the shift only adds a little
to the echo delay the canceller sees (`echo.stats.delay_ms`), well inside what either canceller covers. A frame
is held back in `read()` (about one frame, 32 ms) until the speaker has handed over the samples that pair with
it, so the reference is never short; a speaker that has stalled is counted (`reference_underruns`), not waited
for beyond `REF_STALL_S` once. The canceller itself runs in `read()`, on the loop's thread.

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

from ..assistant.audio_io import SAMPLE_BYTES, AudioUnavailable, load_sounddevice
from .echo import FAR_ACTIVE_RMS, EchoCanceller, EchoError, EchoStats
from .settings import SpeechSettings
from .vad import FRAME_BYTES, FRAME_SAMPLES, SAMPLE_RATE

MAX_QUEUED_FRAMES = 100  # about 3 s of unread microphone audio: more than that means the loop has stopped reading
READ_POLL_S = 0.05
REF_MAX_FRAMES = MAX_QUEUED_FRAMES + 8  # reference kept as long as a frame can wait in the queue: none is paired with zeros
REF_STALL_S = 0.25  # a frame whose reference has still not come after this long goes on without it (and is counted)


class DeviceFault(RuntimeError):
    """The sound card reported a problem (an overflow, a lost device). The turn in progress is not trustworthy."""


class Stream(Protocol):
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def close(self) -> None: ...


InputFactory = Callable[[Callable[[bytes, bool], None]], Stream]  # (on_input) -> an input stream, not yet started
OutputFactory = Callable[[Callable[[int, bool], bytes], int], Stream]  # (on_output(size, fault), sample_rate) -> an output stream


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


class _ToSixteenKilohertz:
    """16-bit mono blocks of a continuous stream converted to 16 kHz by linear interpolation, keeping its place across
    blocks so that no block boundary leaves a seam. Only the reference passes through here, never what is played."""

    def __init__(self, source_rate: int) -> None:
        self._step = source_rate / SAMPLE_RATE
        self._position = 0.0  # where the next output sample falls, in source samples after the last one seen
        self._last = 0

    def feed(self, pcm: bytes) -> bytes:
        if self._step == 1.0:
            return pcm
        samples = array("h")
        samples.frombytes(pcm[: len(pcm) - len(pcm) % SAMPLE_BYTES])
        if not samples:
            return b""
        count = len(samples)
        out = array("h")
        position = self._position
        while position < count:
            index = int(position)  # 0 is the sample before this block, 1 its first
            left = self._last if index == 0 else samples[index - 1]
            right = samples[index]
            out.append(round(left + (right - left) * (position - index)))
            position += self._step
        self._position = position - count
        self._last = samples[-1]
        return out.tobytes()


def sounddevice_factories(input_device=None, output_device=None) -> tuple[InputFactory, OutputFactory]:
    """Stream factories over `sounddevice`. Raises `AudioUnavailable` when it or PortAudio is missing."""
    sd = load_sounddevice()

    def make_input(on_input: Callable[[bytes, bool], None]) -> Stream:
        def callback(indata, frames, timing, status) -> None:
            on_input(bytes(indata), bool(status))

        return sd.RawInputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="int16", blocksize=FRAME_SAMPLES, device=input_device, callback=callback
        )

    def make_output(on_output: Callable[[int, bool], bytes], rate: int) -> Stream:
        def callback(outdata, frames, timing, status) -> None:
            outdata[:] = on_output(len(outdata), bool(status))

        return sd.RawOutputStream(
            samplerate=rate, channels=1, dtype="int16", blocksize=FRAME_SAMPLES, device=output_device, callback=callback
        )

    return make_input, make_output


class FullDuplexDevice:
    def __init__(
        self,
        *,
        output_rate: int = SpeechSettings.output_rate,
        headset: bool = SpeechSettings.headset,
        echo_tail_s: float = SpeechSettings.echo_tail_s,
        clock: Callable[[], float] = time.monotonic,
        input_factory: InputFactory | None = None,
        output_factory: OutputFactory | None = None,
        input_device: int | str | None = None,
        output_device: int | str | None = None,
        canceller: EchoCanceller | None = None,
    ) -> None:
        self.output_rate = output_rate
        self.headset = headset
        self.echo_tail_s = echo_tail_s
        self._clock = clock
        self._input_factory = input_factory
        self._output_factory = output_factory
        self._devices = (input_device, output_device)
        self._cond = threading.Condition(threading.Lock())
        self._canceller = None if headset else canceller  # a headset has no echo to take out
        self._frames: queue.Queue[tuple[bytes, int]] = queue.Queue(
            maxsize=MAX_QUEUED_FRAMES
        )  # (microphone, its reference's number)
        self._waiting: tuple[bytes, int, float] | None = None  # a frame held back for its reference, and since when
        self._ref_stalled = False  # a frame went on without its reference and none has been paired since: do not wait again
        self._reference = bytearray()  # the newest of what the speaker was handed, at 16 kHz
        self._reference_origin = 0  # the number of the sample at the front of `_reference`
        self._reference_total = 0  # samples handed to the speaker so far, at 16 kHz
        self._to_16k = _ToSixteenKilohertz(output_rate)
        self._input_shift: int | None = None  # the speaker's count when the first microphone frame arrived
        self._input_total = 0  # microphone samples so far
        self.reference_underruns = 0  # frames that went to the canceller with part of their reference missing
        self.output_faults = 0  # blocks the sound card reported a problem with (an underflow): speech may have glitched
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
        """The microphone is blanked while Glide speaks: speaker mode with nothing to take the echo out."""
        return not self.headset and self._canceller is None

    @property
    def guarding(self) -> bool:
        """The microphone is being blanked right now (`echo_guard`, and Glide is speaking or has only just stopped)."""
        with self._cond:
            return self._guarded_locked()

    @property
    def echo(self) -> EchoStats | None:
        """The canceller's latest numbers, or None when there is none."""
        return None if self._canceller is None else self._canceller.stats

    @property
    def echo_name(self) -> str | None:
        return None if self._canceller is None else self._canceller.name

    @property
    def echo_active(self) -> bool:
        """Glide's voice may be in the microphone: it is playing, or it has only just stopped. Only with a canceller;
        without one the microphone is blanked (or a headset hears none) and this is False."""
        if self._canceller is None:
            return False
        with self._cond:
            if self._queued > 0 or self._clock() < self._quiet_until:
                return True
        return self._canceller.stats.far_recent_rms >= FAR_ACTIVE_RMS  # the room's echo outlasts the tail setting

    def hold_echo_stats(self, hold: bool) -> None:
        """Freeze (or release) the canceller's statistics: the policy sets it while judging whether a sound is the person."""
        if self._canceller is not None:
            self._canceller.hold = hold

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
        if self._canceller is not None:
            self._canceller.close()

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
                    self._frames.put_nowait((bytes(FRAME_BYTES) if guarded else frame, self._reference_number_locked()))
                except queue.Full:
                    self._fault = "Microphone overflow; the incomplete command was discarded."
                    buffer = b""
                    break
            self._carry = buffer

    def _reference_number_locked(self) -> int:
        """The number of the first reference sample that pairs with the microphone frame being queued (0 with no canceller)."""
        if self._canceller is None:
            return 0
        if self._input_shift is None:
            self._input_shift = self._reference_total
        number = self._input_shift + self._input_total
        self._input_total += FRAME_SAMPLES
        return number

    def _reference_for(self, number: int) -> bytes:
        """The 32 ms of reference that starts at sample `number`, zero-filled where it is not there."""
        with self._cond:  # the origin moves on the speaker's thread, so the offset is taken under the same lock
            start = (number - self._reference_origin) * SAMPLE_BYTES
            if start >= 0:
                piece = bytes(self._reference[start : start + FRAME_BYTES])
            else:  # older than anything kept: the frame waited far too long
                piece = bytes(min(-start, FRAME_BYTES)) + bytes(self._reference[: max(0, FRAME_BYTES + start)])
        if len(piece) < FRAME_BYTES:
            self.reference_underruns += 1
            piece += bytes(FRAME_BYTES - len(piece))
        return piece

    def _reference_ready(self, number: int, timeout: float | None) -> bool:
        end = number + FRAME_SAMPLES
        with self._cond:
            if self._reference_total < end and timeout and not self._closed:
                self._cond.wait_for(lambda: self._reference_total >= end or self._closed, timeout)
            return self._reference_total >= end

    def read(self, timeout: float | None = READ_POLL_S) -> bytes | None:
        """The next frame (echo taken out when there is a canceller), or None if none was ready within `timeout`.
        Raises `DeviceFault` if the card reported one, or the canceller failed."""
        if self._fault is not None:
            raise DeviceFault(self._fault)
        if self._waiting is None:
            try:
                frame, number = self._frames.get(timeout=timeout) if timeout else self._frames.get_nowait()
            except queue.Empty:
                if self._fault is not None:
                    raise DeviceFault(self._fault) from None
                return None
            self._waiting = (frame, number, self._clock())
        frame, number, since = self._waiting
        if self._canceller is None:
            self._waiting = None
            return frame
        # A speaker that has stalled is waited for once: while it stays stalled the frames behind go on without a
        # reference at once, or the queue would fill faster than a wait of REF_STALL_S per frame could drain it.
        if self._reference_ready(number, 0 if self._ref_stalled else timeout):
            self._ref_stalled = False
        elif self._ref_stalled or self._clock() - since >= REF_STALL_S:
            self._ref_stalled = True
        else:
            return None  # its reference has not been handed to the speaker yet: ask again in a moment
        self._waiting = None
        try:
            return self._canceller.process(frame, self._reference_for(number))
        except EchoError as exc:
            self._fault = str(exc)
            raise DeviceFault(self._fault) from None

    def _drain_frames(self) -> None:
        self._waiting = None
        self._ref_stalled = False
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
            self._input_shift = None
            self._input_total = 0
        if self._canceller is not None:
            self._canceller.reset()  # the microphone was off: whatever it had learned of the room is stale
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
            if not self._closed:
                self._input, self._paused = stream, False
                return
        _shut(stream)  # close() ran while the stream was starting: nothing may be left open

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

    def _on_output(self, size: int, fault: bool = False) -> bytes:
        """The sound card's output callback: the next `size` bytes, padded with silence. Never blocks."""
        block = bytearray(size)
        with self._cond:
            self.output_faults += fault
            offset = 0
            while self._output and offset < size:
                piece = self._output.popleft()
                take = min(len(piece), size - offset)
                block[offset : offset + take] = piece[:take]
                if take < len(piece):
                    self._output.appendleft(piece[take:])
                self._queued -= take
                offset += take
            if self._canceller is not None:  # exactly what the card is handed, silence included, so the reference stays in step
                piece = self._to_16k.feed(bytes(block))
                self._reference += piece
                self._reference_total += len(piece) // SAMPLE_BYTES
                dropped = max(0, len(self._reference) - REF_MAX_FRAMES * FRAME_BYTES)
                del self._reference[:dropped]
                self._reference_origin += dropped // SAMPLE_BYTES
                self._cond.notify_all()
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

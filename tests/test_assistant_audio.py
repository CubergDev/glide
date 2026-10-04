"""Playback, capture and end-of-speech detection, against fake sound cards. No device is ever opened."""

from __future__ import annotations

import importlib
import sys
import threading
from array import array
from types import ModuleType

import pytest
from test_assistant_fakes import wait_until

from glide.assistant import audio_io
from glide.assistant.audio_io import (
    AudioUnavailable,
    Endpointer,
    Microphone,
    Player,
    SoundDeviceInput,
    SoundDeviceOutput,
    chunked,
    rms,
)

RATE = 1000  # a slow "sample rate" so a block of 50 ms is 50 samples and tests stay tiny


def tone(level: int, seconds: float, rate: int = 16000) -> bytes:
    """PCM that alternates between +level and -level, whose loudness (rms) is exactly `level`."""
    count = int(seconds * rate)
    samples = array("h", [level if i % 2 == 0 else -level for i in range(count)])
    return samples.tobytes()


class FakeOutput:
    """A speaker that records every write. `hold` makes writes wait, as a full device buffer does."""

    def __init__(self, hold: threading.Event | None = None) -> None:
        self.writes: list[tuple[int, int]] = []  # (bytes, rate)
        self.aborts = 0
        self.flushes = 0
        self.closed = False
        self.hold = hold
        self.entered = threading.Event()
        self._lock = threading.Lock()

    def write(self, pcm: bytes, sample_rate: int) -> None:
        self.entered.set()
        if self.hold is not None:
            self.hold.wait(3)
        with self._lock:
            self.writes.append((len(pcm), sample_rate))

    def abort(self) -> None:
        self.aborts += 1
        if self.hold is not None:
            self.hold.set()  # an aborted device lets a blocked write return

    def flush(self) -> None:
        self.flushes += 1

    def close(self) -> None:
        self.closed = True


class FakeInput:
    def __init__(self, chunks) -> None:
        self.chunks = list(chunks)
        self.started: tuple[int, int] | None = None
        self.stopped = 0

    def start(self, sample_rate: int, frames: int) -> None:
        self.started = (sample_rate, frames)

    def read(self) -> bytes:
        return self.chunks.pop(0) if self.chunks else b""

    def stop(self) -> None:
        self.stopped += 1


# -- the player -------------------------------------------------------------------------------------


def test_audio_is_written_in_short_blocks_at_the_rate_it_came_with():
    out = FakeOutput()
    player = Player(out, block_ms=50)
    player.play(bytes(2 * 120), RATE)  # 120 samples: blocks of 50, 50 and 20 samples
    assert player.wait_idle(timeout=3)
    assert out.writes == [(100, RATE), (100, RATE), (40, RATE)]
    assert out.flushes == 1  # the device was drained once everything was written
    player.close()


def test_a_chunk_ending_in_half_a_sample_is_trimmed_to_whole_samples():
    out = FakeOutput()
    player = Player(out, block_ms=50)
    player.play(bytes(7), RATE)
    assert player.wait_idle(timeout=3)
    assert out.writes == [(6, RATE)]
    player.close()


def test_the_rate_changes_between_chunks_and_each_write_says_which():
    out = FakeOutput()
    player = Player(out, block_ms=1000)
    player.play(bytes(20), 24000)
    player.play(bytes(20), 22050)
    assert player.wait_idle(timeout=3)
    assert [rate for _, rate in out.writes] == [24000, 22050]
    player.close()


def test_cancel_stops_a_chunk_part_way_and_drops_everything_queued():
    hold = threading.Event()
    out = FakeOutput(hold=hold)
    player = Player(out, block_ms=50)
    player.play(bytes(2 * 500), RATE)  # ten blocks
    player.play(bytes(2 * 500), RATE)  # queued behind it
    assert out.entered.wait(3)  # the first block is "in the device"
    player.cancel()
    assert out.aborts == 1
    assert wait_until(lambda: not player.busy)
    assert len(out.writes) <= 1  # the block that was in flight, and nothing after the cancel
    player.close()


def test_audio_played_after_a_cancel_is_heard_in_full():
    out = FakeOutput()
    player = Player(out, block_ms=50)
    player.cancel()
    player.play(bytes(2 * 100), RATE)
    assert player.wait_idle(timeout=3)
    assert sum(n for n, _ in out.writes) == 200
    player.close()


def test_cancel_with_nothing_playing_is_harmless_and_cuts_the_device():
    out = FakeOutput()
    player = Player(out)
    player.cancel()
    player.cancel()
    assert out.aborts == 2
    assert player.wait_idle(timeout=1)  # nothing was ever played, so nothing to wait for
    player.close()


def test_wait_idle_times_out_while_the_device_is_still_taking_audio():
    hold = threading.Event()
    player = Player(FakeOutput(hold=hold), block_ms=50)
    player.play(bytes(100), RATE)
    assert player.wait_idle(timeout=0.05) is False
    hold.set()
    assert player.wait_idle(timeout=3)
    player.close()


def test_a_failing_device_is_reported_and_the_next_chunk_still_plays():
    errors = []

    class Flaky(FakeOutput):
        def write(self, pcm, sample_rate):
            if not self.writes and not self.entered.is_set():
                self.entered.set()
                raise OSError("device busy")
            super().write(pcm, sample_rate)

    out = Flaky()
    player = Player(out, block_ms=1000, on_error=errors.append)
    player.play(bytes(20), RATE)
    assert wait_until(lambda: errors)
    player.play(bytes(20), RATE)
    assert player.wait_idle(timeout=3)
    assert out.writes == [(20, RATE)]
    player.close()


def test_closing_aborts_the_device_stops_the_thread_and_refuses_new_audio():
    out = FakeOutput()
    player = Player(out)
    player.play(bytes(20), RATE)
    player.close()
    assert out.closed and out.aborts >= 1
    player.play(bytes(20), RATE)  # ignored
    assert not player.busy


# -- loudness and end of speech ---------------------------------------------------------------------


def test_rms_is_the_loudness_of_the_samples():
    assert rms(b"") == 0.0
    assert rms(bytes(3200)) == 0.0
    assert rms(tone(1000, 0.1)) == pytest.approx(1000)
    assert rms(tone(1000, 0.1) + b"\x01") == pytest.approx(1000)  # a stray odd byte is ignored


def feed_all(endpointer: Endpointer, *parts: bytes) -> list[bool]:
    return [endpointer.feed(c) for part in parts for c in chunked(part)]


def test_speech_then_silence_ends_the_utterance_after_the_hang_time():
    ep = Endpointer(silence_s=0.5)
    assert not any(feed_all(ep, tone(0, 0.3), tone(3000, 1.0)))  # a quiet start and then speech: not over
    assert ep.heard
    done = feed_all(ep, tone(0, 0.5))
    assert done == [False, False, False, False, True]  # five quiet chunks of 100 ms
    assert ep.reason == "silence"


def test_a_pause_shorter_than_the_hang_time_does_not_end_it():
    ep = Endpointer(silence_s=0.8)
    assert not any(feed_all(ep, tone(3000, 0.5), tone(0, 0.5), tone(3000, 0.5), tone(0, 0.3)))
    assert ep.reason is None


def test_a_click_is_not_speech():
    ep = Endpointer(min_speech_s=0.2, no_speech_s=1.0)
    feed_all(ep, tone(0, 0.2), tone(5000, 0.1), tone(0, 0.5))  # one loud 100 ms chunk: shorter than min_speech_s
    assert not ep.heard


def test_nobody_speaking_ends_the_recording_with_no_speech():
    ep = Endpointer(no_speech_s=1.0)
    done = feed_all(ep, tone(0, 1.0))
    assert done[-1] and ep.reason == "no speech" and not ep.heard


def test_a_recording_that_never_stops_is_cut_at_the_maximum():
    ep = Endpointer(max_s=1.0, silence_s=10)
    done = feed_all(ep, tone(3000, 2.0))
    assert done.index(True) == 9 and ep.reason == "max"


def test_a_noisy_room_raises_the_bar_by_itself():
    ep = Endpointer(threshold=500, ratio=3.0, no_speech_s=10)
    feed_all(ep, tone(400, 1.0))  # steady background noise, under the fixed threshold
    assert not ep.is_speech(1000)  # 3 x the learned floor is above 1000
    assert ep.is_speech(2000)


# -- the microphone ---------------------------------------------------------------------------------


def test_the_microphone_yields_100_ms_chunks_and_always_stops_the_device():
    pcm = tone(1000, 0.35)
    backend = FakeInput(chunked(pcm))
    mic = Microphone(backend)
    chunks = list(mic.record())
    assert backend.started == (16000, 1600)
    assert [len(c) for c in chunks] == [3200, 3200, 3200, 1600]
    assert b"".join(chunks) == pcm
    assert backend.stopped == 1
    assert mic.chunk_bytes == 3200


def test_setting_stop_ends_the_chunks_and_that_end_is_what_the_consumer_sees():
    backend = FakeInput([bytes(3200)] * 50)
    stop = threading.Event()
    got = []
    for chunk in Microphone(backend).record(stop):
        got.append(chunk)
        if len(got) == 3:
            stop.set()
    assert len(got) == 3
    assert backend.stopped == 1


def test_the_endpointer_ends_a_recording_after_the_chunk_that_decided_it():
    backend = FakeInput(chunked(tone(3000, 0.3) + tone(0, 2.0)))
    ep = Endpointer(silence_s=0.3)
    chunks = list(Microphone(backend).record(endpointer=ep))
    assert len(chunks) == 6  # 3 of speech and 3 of silence, then the endpointer said it was over
    assert ep.reason == "silence"


def test_the_device_is_closed_when_the_consumer_gives_up_early():
    backend = FakeInput([bytes(3200)] * 10)
    stream = Microphone(backend).record()
    next(stream)
    stream.close()
    assert backend.stopped == 1


def test_a_recording_is_capped_in_length():
    backend = FakeInput([bytes(3200)] * 100)
    assert len(list(Microphone(backend).record(max_s=0.35))) == 4


# -- the real backends, against a fake `sounddevice` ------------------------------------------------


STREAMS: list[FakeStream] = []


class FakeStream:
    def __init__(self, kind: str, **kw) -> None:
        self.kind = kind
        self.kw = kw
        self.active = False
        self.calls: list[str] = []
        self.written: list[bytes] = []
        STREAMS.append(self)

    def start(self) -> None:
        self.active = True
        self.calls.append("start")

    def stop(self) -> None:
        self.active = False
        self.calls.append("stop")

    def abort(self) -> None:
        self.active = False
        self.calls.append("abort")

    def close(self) -> None:
        self.calls.append("close")

    def write(self, data) -> None:
        self.written.append(bytes(data))

    def read(self, frames: int):
        return memoryview(bytes(frames * 2)), False


@pytest.fixture
def fake_sounddevice(monkeypatch):
    STREAMS.clear()
    module = ModuleType("sounddevice")
    module.RawOutputStream = lambda **kw: FakeStream("out", **kw)
    module.RawInputStream = lambda **kw: FakeStream("in", **kw)
    monkeypatch.setitem(sys.modules, "sounddevice", module)
    return module


def test_importing_the_module_does_not_import_sounddevice(monkeypatch):
    monkeypatch.delitem(sys.modules, "sounddevice", raising=False)
    monkeypatch.delitem(sys.modules, "glide.assistant.audio_io", raising=False)
    fresh = importlib.import_module("glide.assistant.audio_io")
    assert "sounddevice" not in sys.modules
    assert fresh.SAMPLE_RATE == 16000
    monkeypatch.setitem(sys.modules, "glide.assistant.audio_io", audio_io)  # the other tests keep their module


@pytest.mark.parametrize("backend", [SoundDeviceOutput, SoundDeviceInput])
def test_a_missing_package_is_a_clear_error_not_an_import_error(monkeypatch, backend):
    monkeypatch.setitem(sys.modules, "sounddevice", None)  # makes `import sounddevice` raise ImportError
    with pytest.raises(AudioUnavailable) as caught:
        backend()
    assert "sounddevice" in str(caught.value) and "uv sync --extra speech" in str(caught.value)


def test_a_missing_portaudio_library_is_the_same_clear_error(monkeypatch):
    class NoPortAudio:
        """What importing `sounddevice` does when the PortAudio library is not installed: an OSError."""

        @staticmethod
        def find_spec(name, path=None, target=None):
            if name == "sounddevice":
                raise OSError("PortAudio library not found")
            return None

    monkeypatch.delitem(sys.modules, "sounddevice", raising=False)
    monkeypatch.setattr(sys, "meta_path", [NoPortAudio, *sys.meta_path])
    with pytest.raises(AudioUnavailable) as caught:
        SoundDeviceOutput()
    assert "OSError" in str(caught.value)


def test_the_output_opens_one_stream_per_rate_aborts_to_cut_and_stops_to_drain(fake_sounddevice):
    out = SoundDeviceOutput(device="speakers")
    out.write(b"\x00\x01", 24000)
    out.write(b"\x00\x02", 24000)
    (first,) = STREAMS
    assert first.kw == {"samplerate": 24000, "channels": 1, "dtype": "int16", "device": "speakers"}
    assert first.calls == ["start"] and first.written == [b"\x00\x01", b"\x00\x02"]
    out.abort()
    assert first.calls[-1] == "abort"  # discards the buffer; stop() would have played it out
    out.write(b"\x00\x03", 24000)
    assert first.calls[-2:] == ["abort", "start"]  # the same stream is started again after an abort
    out.flush()
    assert first.calls[-1] == "stop"  # drains
    out.write(b"\x00\x04", 22050)  # a new rate: the old stream is finished and a new one opened
    assert len(STREAMS) == 2 and STREAMS[1].kw["samplerate"] == 22050
    assert "close" in first.calls
    out.close()


def test_the_input_reads_blocks_and_copies_them_out_as_bytes(fake_sounddevice):
    mic = SoundDeviceInput()
    mic.start(16000, 1600)
    (stream,) = STREAMS
    assert stream.kw["blocksize"] == 1600 and stream.kw["samplerate"] == 16000 and stream.calls == ["start"]
    data = mic.read()
    assert isinstance(data, bytes) and len(data) == 3200
    mic.stop()
    assert stream.calls[-2:] == ["abort", "close"]


def test_the_default_player_and_microphone_need_the_package(monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice", None)
    with pytest.raises(AudioUnavailable):
        Player()
    with pytest.raises(AudioUnavailable):
        Microphone()

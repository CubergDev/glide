"""The full-duplex device against fake streams and a fake clock: echo guard, headset mode, pause and resume."""

from __future__ import annotations

import pytest
from guards_voice import no_real_audio  # noqa: F401

from glide.speech.audio import DeviceFault, FullDuplexDevice, resample
from glide.speech.vad import FRAME_BYTES

SPEECH = b"\x01\x00" * (FRAME_BYTES // 2)  # one frame that is not silence
SILENCE = bytes(FRAME_BYTES)
RATE = 24000


class FakeStream:
    def __init__(self, log, name):
        self.log, self.name = log, name

    def start(self):
        self.log.append((self.name, "start"))

    def stop(self):
        self.log.append((self.name, "stop"))

    def close(self):
        self.log.append((self.name, "close"))


class Rig:
    def __init__(self, **kw):
        self.now = 10.0
        self.log = []
        self.on_input = None
        self.on_output = None
        self.device = FullDuplexDevice(
            output_rate=RATE, clock=lambda: self.now, input_factory=self._input, output_factory=self._output, **kw
        )
        self.device.start()

    def _input(self, callback):
        self.on_input = callback
        return FakeStream(self.log, "in")

    def _output(self, callback, rate):
        self.on_output = callback
        return FakeStream(self.log, "out")

    def pull(self, size=FRAME_BYTES):
        """The sound card asks the speaker for `size` bytes; the clock moves by what that block lasts."""
        data = self.on_output(size)
        self.now += size / 2 / RATE
        return data

    def hear(self, data=SPEECH):
        self.on_input(data, False)
        return self.device.read(timeout=None)


def test_the_microphone_is_reframed_into_32_ms_frames_whatever_the_card_delivers():
    rig = Rig()
    rig.on_input(SPEECH[:300], False)
    assert rig.device.read(timeout=None) is None
    rig.on_input(SPEECH[300:] + SPEECH[:100], False)
    assert rig.device.read(timeout=None) == SPEECH
    assert rig.device.read(timeout=None) is None


def test_speaker_mode_hears_silence_while_speaking_and_for_the_tail_after():
    rig = Rig()
    rig.device.play(b"\x05\x00" * 4000, RATE)
    assert rig.hear() == SILENCE  # audio is queued
    while rig.device._queued:
        rig.pull()
    assert rig.hear() == SILENCE  # the device is still sounding the last block
    rig.now += 0.1
    assert rig.hear() == SILENCE  # reverberation
    rig.now += 0.4
    assert rig.hear() == SPEECH  # the tail is over


def test_frames_captured_before_speech_begins_are_dropped_in_speaker_mode():
    rig = Rig()
    rig.on_input(SPEECH, False)
    rig.device.play(b"\x05\x00" * 100, RATE)
    assert rig.device.read(timeout=None) is None


def test_a_cancel_keeps_the_guard_up_for_the_tail_because_the_card_still_has_sound():
    rig = Rig()
    rig.device.play(b"\x05\x00" * 4000, RATE)
    rig.pull()
    rig.device.cancel()
    assert rig.device._queued == 0
    assert rig.hear() == SILENCE
    rig.now += 0.5
    assert rig.hear() == SPEECH


def test_headset_mode_keeps_hearing_the_person_while_glide_speaks():
    rig = Rig(headset=True)
    rig.device.play(b"\x05\x00" * 4000, RATE)
    assert rig.device.playing
    assert rig.hear() == SPEECH  # this is what lets a spoken interruption through


def test_a_late_finish_of_an_old_utterance_cannot_unlock_the_guard_mid_sentence():
    """The guard follows the device's own queue, so there is no flag an old utterance could clear."""
    rig = Rig()
    rig.device.play(b"\x05\x00" * 4000, RATE)
    rig.device.wait_idle(timeout=0)  # a stale waiter returning false changes nothing
    assert rig.hear() == SILENCE


def test_output_is_padded_with_silence_and_in_order():
    rig = Rig()
    rig.device.play(b"\x01\x00\x02\x00", RATE)
    assert rig.pull(8) == b"\x01\x00\x02\x00" + bytes(4)
    assert rig.device.wait_idle(timeout=0)


def test_audio_at_another_rate_is_converted_to_the_devices():
    rig = Rig()
    rig.device.play(b"\x00\x00\x10\x00" * 100, 12000)  # 200 samples at half the rate
    assert rig.device._queued == 400 * 2


def test_resample_keeps_whole_samples_and_is_the_identity_at_the_same_rate():
    pcm = b"\x01\x00\x02\x00\x03"
    assert resample(pcm, 16000, 16000) == b"\x01\x00\x02\x00"
    assert len(resample(b"\x00\x00" * 160, 16000, 24000)) == 240 * 2
    assert resample(b"", 16000, 24000) == b""


def test_an_overflow_discards_the_turn_and_surfaces_as_a_fault():
    rig = Rig()
    rig.on_input(SPEECH, True)
    with pytest.raises(DeviceFault, match="overflow"):
        rig.device.read(timeout=None)


def test_pause_stops_the_microphone_and_ignores_late_callbacks_resume_starts_it_clean():
    rig = Rig()
    rig.on_input(SPEECH, False)
    rig.device.pause_input()
    assert ("in", "stop") in rig.log and ("in", "close") in rig.log
    rig.on_input(SPEECH, False)  # a callback already in flight
    assert rig.device.read(timeout=None) is None
    rig.device.resume_input()
    assert rig.log.count(("in", "start")) == 2 and not rig.device.input_paused
    assert rig.hear() == SPEECH


def test_a_failed_resume_leaves_the_microphone_paused():
    rig = Rig()
    rig.device.pause_input()

    def broken(callback):
        raise OSError("device gone")

    rig.device._input_factory = broken
    with pytest.raises(OSError):
        rig.device.resume_input()
    assert rig.device.input_paused
    rig.on_input(SPEECH, False)
    assert rig.device.read(timeout=None) is None


def test_a_start_that_fails_closes_what_it_opened():
    log = []

    def input_factory(callback):
        raise OSError("no microphone")

    device = FullDuplexDevice(input_factory=input_factory, output_factory=lambda cb, rate: FakeStream(log, "out"))
    with pytest.raises(OSError):
        device.start()
    assert ("out", "close") in log


def test_close_is_idempotent_and_drops_queued_audio():
    rig = Rig()
    rig.device.play(b"\x05\x00" * 100, RATE)
    rig.device.close()
    rig.device.close()
    assert rig.device._queued == 0
    rig.device.play(b"\x05\x00" * 100, RATE)
    assert rig.device._queued == 0


def test_a_close_that_lands_while_the_microphone_is_being_resumed_leaves_no_open_stream():
    """resume_input builds and starts the stream outside the lock; a close() in that gap used to leak the new stream."""
    import threading

    rig = Rig()
    rig.device.pause_input()
    entered, release = threading.Event(), threading.Event()

    class SlowStream(FakeStream):
        def start(self):
            entered.set()
            assert release.wait(5)
            super().start()

    rig.device._input_factory = lambda callback: SlowStream(rig.log, "late")
    resumer = threading.Thread(target=rig.device.resume_input)
    resumer.start()
    assert entered.wait(5)
    rig.device.close()  # lands after the stream began to start and before resume stored it
    release.set()
    resumer.join(5)
    assert not resumer.is_alive()
    assert ("late", "start") in rig.log
    assert ("late", "stop") in rig.log and ("late", "close") in rig.log  # nothing is left open
    assert rig.device.input_paused and rig.device._input is None

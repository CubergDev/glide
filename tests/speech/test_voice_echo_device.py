"""The full-duplex device with an echo canceller: what it hands the canceller, when, and what it stops doing. Settings of
the new keys and the choice of canceller. Pure Python: the canceller here is a recording fake, so no numpy is needed.
How well a real canceller cancels is proved in test_voice_echo_dsp.py.
"""

from __future__ import annotations

import itertools
from array import array

import pytest
from guards_voice import no_real_audio  # noqa: F401

import glide.speech.echo as echo_module
from glide.providers.config import ConfigError
from glide.speech.audio import REF_STALL_S, DeviceFault, FullDuplexDevice, _ToSixteenKilohertz
from glide.speech.echo import FAR_ACTIVE_RMS, EchoCanceller, EchoError, EchoStats, make_canceller
from glide.speech.settings import SpeechSettings
from glide.speech.vad import FRAME_BYTES, FRAME_S


class Recorder(EchoCanceller):
    """Passes the microphone through and records every (near, far) pair it is given."""

    name = "recorder"

    def __init__(self):
        super().__init__()
        self.pairs: list[tuple[bytes, bytes]] = []
        self.resets = 0
        self.closed = False
        self.fail = False

    def _process(self, near, far):
        if self.fail:
            raise EchoError("boom")
        self.pairs.append((near, far))
        return near

    def _reset_state(self):
        self.resets += 1

    def close(self):
        self.closed = True


class _Stream:
    def start(self): ...
    def stop(self): ...
    def close(self): ...


class Rig:
    def __init__(self, *, rate=16000, **kw):
        self.now = 10.0
        self.rate = rate
        self.canceller = kw.pop("canceller", Recorder())
        self.on_input = self.on_output = None
        self.device = FullDuplexDevice(
            output_rate=rate,
            clock=lambda: self.now,
            input_factory=self._input,
            output_factory=self._output,
            canceller=self.canceller,
            **kw,
        )
        self.device.start()

    def _input(self, callback):
        self.on_input = callback
        return _Stream()

    def _output(self, callback, rate):
        self.on_output = callback
        return _Stream()

    def pull(self, samples=512):
        """The sound card asks the speaker for a block; the clock moves by what it lasts."""
        data = self.on_output(samples * 2)
        self.now += samples / self.rate
        return data

    def hear(self, frame):
        self.on_input(frame, False)
        return self.device.read(timeout=None)


def frame_of(value: int) -> bytes:
    return array("h", [value]).tobytes() * (FRAME_BYTES // 2)


def test_the_microphone_stays_open_while_glide_speaks_and_the_echo_guard_is_off():
    rig = Rig(echo_tail_s=0.3)
    rig.device.play(frame_of(500) * 4, 16000)
    rig.pull()
    assert rig.device.playing and not rig.device.echo_guard
    rig.hear(frame_of(300))
    rig.pull()
    assert rig.device.read(timeout=None) == frame_of(300)  # not blanked, and the canceller got it
    assert rig.canceller.pairs and rig.canceller.pairs[0][0] == frame_of(300)


def test_a_headset_has_no_canceller_and_no_guard():
    rig = Rig(headset=True)
    assert rig.device.echo is None and not rig.device.echo_active and not rig.device.echo_guard
    assert rig.hear(frame_of(300)) == frame_of(300) and rig.canceller.pairs == []


def test_each_microphone_frame_is_paired_with_the_reference_that_is_never_older_than_it():
    """Microphone frame k goes with the frame the speaker was handed one frame after the one the microphone began with."""
    rig = Rig()
    far = [frame_of(1000 + 100 * i) for i in range(8)]
    rig.device.play(b"".join(far), 16000)
    given = [rig.pull()]  # the speaker runs first, as it does when the device starts
    assert given[0] == far[0]
    mics = [frame_of(7 * (i + 1)) for i in range(5)]
    cleaned = []
    for mic in mics:
        out = rig.hear(mic)
        if out is None:  # the frame waits until the speaker has handed over what pairs with it
            given.append(rig.pull())
            out = rig.device.read(timeout=None)
        cleaned.append(out)
    assert cleaned == mics
    assert [pair[0] for pair in rig.canceller.pairs] == mics
    assert [pair[1] for pair in rig.canceller.pairs] == given[1 : 1 + len(mics)]  # mic k with far k+1


def test_a_frame_waits_for_its_reference_and_a_stalled_speaker_is_counted_not_waited_for():
    rig = Rig()
    rig.pull()
    assert rig.hear(frame_of(5)) is None  # its reference has not been handed over yet
    assert rig.device.read(timeout=None) is None
    rig.now += REF_STALL_S + 0.01
    assert rig.device.read(timeout=None) == frame_of(5)  # goes on without it
    assert rig.device.reference_underruns == 1 and rig.canceller.pairs[-1][1] == bytes(FRAME_BYTES)


def test_a_loop_that_falls_far_behind_still_gets_every_frame_with_its_own_reference():
    """Frames queue up while the loop is busy (a pause, a busy CPU) and are paired with their reference later, however late."""
    rig = Rig()
    far = [frame_of(1000 + i) for i in range(100)]
    rig.device.play(b"".join(far), 16000)
    mics = [frame_of(5 + i) for i in range(90)]
    given = [rig.pull()]
    for mic in mics:  # the speaker and microphone callbacks go on; nothing reads
        rig.on_input(mic, False)
        given.append(rig.pull())
    got = []
    while (out := rig.device.read(timeout=None)) is not None:
        got.append(out)
    assert got == mics and rig.device.reference_underruns == 0
    assert [pair[1] for pair in rig.canceller.pairs] == given[1 : 1 + len(mics)]


def test_the_reference_of_a_24_khz_speaker_is_one_continuous_16_khz_signal():
    rig = Rig(rate=24000)
    ramp = array("h", range(0, 24000)).tobytes()  # sample i has the value i, so 16 kHz samples step by 1.5
    rig.device.play(ramp, 24000)
    rig.pull(512)
    for _ in range(14):
        out = rig.hear(frame_of(1))
        while out is None:
            rig.pull(512)
            out = rig.device.read(timeout=None)
    reference = array("h")
    for _, far in rig.canceller.pairs:
        reference.extend(array("h", far))
    steps = [b - a for a, b in itertools.pairwise(reference)]
    assert steps and set(steps) <= {1, 2}  # no seam, no hole, at any block boundary


def test_the_reference_converter_is_the_same_whole_or_in_pieces_and_leaves_16_khz_alone():
    data = array("h", range(0, 3000)).tobytes()
    whole = _ToSixteenKilohertz(48000).feed(data)
    converter = _ToSixteenKilohertz(48000)
    pieces = b"".join(converter.feed(data[i : i + 600]) for i in range(0, len(data), 600))
    assert pieces == whole and len(whole) == len(data) // 3
    assert _ToSixteenKilohertz(16000).feed(data) == data


def test_speech_beginning_does_not_throw_away_what_the_microphone_caught():
    rig = Rig()
    rig.pull()
    rig.on_input(frame_of(11), False)  # caught before Glide speaks
    rig.device.play(frame_of(500) * 4, 16000)
    rig.pull()
    assert rig.device.read(timeout=None) == frame_of(11)  # still there: starting to speak drained nothing


def test_cancel_silences_the_speaker_within_the_next_block_and_does_not_hold_the_microphone():
    rig = Rig(echo_tail_s=0.3)
    rig.device.play(frame_of(900) * 20, 16000)
    assert any(rig.pull())  # sounding
    rig.device.cancel()
    assert not any(rig.pull())  # the very next block is silence
    assert not rig.device.echo_guard
    rig.hear(frame_of(40))
    rig.pull()
    assert rig.device.read(timeout=None) == frame_of(40)  # no tail blanks what the person says right after the cut


def test_echo_is_possible_while_audio_is_queued_for_the_tail_and_while_the_rooms_echo_is_still_coming():
    rig = Rig(echo_tail_s=0.3)
    assert not rig.device.echo_active
    rig.device.play(frame_of(900) * 4, 16000)
    assert rig.device.echo_active  # queued
    for _ in range(4):
        rig.pull()
    assert rig.device.echo_active  # inside the tail
    rig.now += 0.5
    assert not rig.device.echo_active
    rig.canceller._stats = EchoStats(far_recent_rms=FAR_ACTIVE_RMS + 1)
    assert rig.device.echo_active  # the canceller still sees the reference within the room's memory


def test_the_cancellers_numbers_and_hold_are_reachable_through_the_device():
    rig = Rig()
    assert rig.device.echo is rig.canceller.stats
    rig.device.hold_echo_stats(True)
    assert rig.canceller.hold is True
    rig.device.hold_echo_stats(False)
    assert rig.canceller.hold is False
    FullDuplexDevice().hold_echo_stats(True)  # no canceller: nothing to do, nothing raised


def test_resuming_the_microphone_resets_the_canceller_and_pairs_from_the_start_again():
    rig = Rig()
    rig.pull()
    rig.hear(frame_of(1))
    rig.device.pause_input()
    resets = rig.canceller.resets
    rig.device.resume_input()
    assert rig.canceller.resets == resets + 1
    assert rig.device._input_shift is None and rig.device._input_total == 0 and rig.device._waiting is None


def test_a_canceller_that_fails_ends_the_loop_with_a_device_fault_not_a_wrong_frame():
    rig = Rig()
    rig.canceller.fail = True
    rig.pull()
    rig.hear(frame_of(1))
    rig.pull()
    with pytest.raises(DeviceFault, match="boom"):
        rig.device.read(timeout=None)
    with pytest.raises(DeviceFault):
        rig.device.read(timeout=None)


def test_closing_the_device_closes_the_canceller():
    rig = Rig()
    rig.device.close()
    assert rig.canceller.closed


def test_a_frame_is_32_ms():
    assert pytest.approx(0.032) == FRAME_S


# -- the canceller's statistics, shared by every implementation --------------------------------------


class PassThrough(EchoCanceller):
    name = "pass"

    def _process(self, near, far):
        return near

    def _reset_state(self): ...


class Halver(PassThrough):
    def _process(self, near, far):
        return array("h", [s // 10 for s in array("h", near)]).tobytes()  # takes 20 dB out of everything


def test_the_erle_is_unmeasured_at_first_then_read_off_the_frames_where_the_speaker_sounds():
    canceller = Halver()
    far, near = frame_of(2000), frame_of(1000)
    for _ in range(10):
        canceller.process(near, far)
    assert canceller.stats.erle_db is None and not canceller.stats.measured
    for _ in range(40):
        canceller.process(near, far)
    assert canceller.stats.measured and canceller.stats.erle_db == pytest.approx(20.0, abs=0.1)
    assert canceller.stats.far_recent_rms == pytest.approx(2000)


def test_a_silent_reference_teaches_the_canceller_nothing_and_a_hold_freezes_what_it_has_learned():
    canceller = Halver()
    for _ in range(60):
        canceller.process(frame_of(1000), bytes(FRAME_BYTES))
    assert canceller.stats.erle_db is None
    for _ in range(60):
        canceller.process(frame_of(1000), frame_of(2000))
    learned = canceller.stats.erle_db
    canceller.hold = True
    for _ in range(60):
        canceller.process(frame_of(1000), frame_of(2000))
    assert canceller.stats.erle_db == learned


def test_the_expected_residual_follows_the_reference_level_and_is_zero_before_the_erle_is_trusted():
    canceller = Halver()
    canceller.process(frame_of(1000), frame_of(2000))
    assert canceller.stats.expected_residual_rms == 0.0
    for _ in range(60):
        canceller.process(frame_of(1000), frame_of(2000))
    assert canceller.stats.expected_residual_rms == pytest.approx(100.0, rel=0.05)  # what is left of 1000 at 20 dB


def test_the_echo_delay_is_read_off_the_loudness_envelopes_to_a_frame():
    canceller = PassThrough()
    sequence = [
        200,
        3000,
        400,
        2500,
        100,
        3500,
        300,
        2800,
        150,
        3200,
        250,
        2600,
    ] * 12  # loud and quiet, no pattern in a short lag
    delay = 4
    heard = [0] * delay + sequence
    for k, level in enumerate(sequence):
        canceller.process(frame_of(heard[k] + 1), frame_of(level))
    assert canceller.stats.delay_ms == pytest.approx(delay * FRAME_S * 1000)


def test_frames_of_the_wrong_size_are_refused():
    with pytest.raises(ValueError):
        PassThrough().process(bytes(10), bytes(FRAME_BYTES))


def test_reset_forgets_the_room():
    canceller = Halver()
    for _ in range(60):
        canceller.process(frame_of(1000), frame_of(2000))
    canceller.reset()
    assert canceller.stats.erle_db is None and canceller.stats.frames == 0


# -- settings and the choice of canceller -----------------------------------------------------------


def test_the_new_keys_have_defaults_that_keep_headset_and_speaker_semantics():
    s = SpeechSettings.from_mapping(None)
    assert (s.echo_canceller, s.barge_min_voiced_ms, s.barge_margin_db, s.barge_min_erle_db, s.stop_phrases) == (
        "auto",
        190,
        8.0,
        6.0,
        (),
    )


def test_the_new_keys_are_read_and_a_list_of_phrases_becomes_a_tuple():
    s = SpeechSettings.from_mapping(
        {
            "echo_canceller": "nlms",
            "barge_min_voiced_ms": 250,
            "barge_margin_db": 10,
            "stop_phrases": ["hold on", "enough already"],
        }
    )
    assert s.echo_canceller == "nlms" and s.barge_min_voiced_ms == 250 and s.barge_margin_db == 10.0
    assert s.stop_phrases == ("hold on", "enough already")


@pytest.mark.parametrize(
    "table",
    [
        {"echo_canceller": "magic"},
        {"echo_canceller": 1},
        {"barge_min_voiced_ms": 50},
        {"barge_min_voiced_ms": 2000},
        {"barge_margin_db": 1},
        {"barge_margin_db": 40},
        {"barge_min_erle_db": -1},
        {"stop_phrases": "stop"},
        {"stop_phrases": [1]},
        {"stop_phrases": [""]},
        {"stop_phrases": ["!!!"]},
    ],
)
def test_a_bad_new_key_is_a_config_error_naming_the_table(table):
    with pytest.raises(ConfigError, match=r"\[speech\]"):
        SpeechSettings.from_mapping(table)


class _Boom(EchoCanceller):
    name = "boom"

    def __init__(self):
        raise EchoError("not installed")

    def _process(self, near, far): ...
    def _reset_state(self): ...


class _Good(PassThrough):
    name = "good"


def test_none_builds_nothing_and_a_named_canceller_that_cannot_be_built_is_an_error(monkeypatch):
    monkeypatch.setattr(echo_module, "CANCELLERS", {"webrtc": _Boom, "nlms": _Good})
    assert make_canceller("none") is None
    with pytest.raises(EchoError):
        make_canceller("webrtc")
    assert isinstance(make_canceller("nlms"), _Good)


def test_auto_takes_the_first_that_builds_and_says_which_fallback_it_took(monkeypatch):
    warned = []
    monkeypatch.setattr(echo_module, "CANCELLERS", {"webrtc": _Boom, "nlms": _Good})
    assert isinstance(make_canceller("auto", warned.append), _Good)
    assert len(warned) == 1 and "nlms" in warned[0] and "not installed" in warned[0]
    warned.clear()
    monkeypatch.setattr(echo_module, "CANCELLERS", {"webrtc": _Good, "nlms": _Boom})
    assert isinstance(make_canceller("auto", warned.append), _Good) and warned == []


def test_auto_with_nothing_available_says_so_and_leaves_speaker_mode_half_duplex(monkeypatch):
    warned = []
    monkeypatch.setattr(echo_module, "CANCELLERS", {"webrtc": _Boom, "nlms": _Boom})
    assert make_canceller("auto", warned.append) is None
    assert len(warned) == 1 and "half duplex" in warned[0]


def test_the_status_of_a_device_without_a_canceller_is_empty_and_the_legacy_guard_is_unchanged():
    device = FullDuplexDevice(input_factory=lambda cb: _Stream(), output_factory=lambda cb, rate: _Stream())
    assert device.echo is None and device.echo_guard and not device.echo_active

"""Barge-in by voice: the gate's policy, and the voice loop's behaviour around it, over a scripted device that reports echo
statistics. Pure Python (no numpy, no sound card): what the numbers mean and how a real canceller produces them is
proved in test_voice_echo_dsp.py against simulated rooms.

What this file pins:
- a sound has to be voiced, louder than the echo still expected, long enough, and only once the canceller has measured
  itself, before it is the person: Glide's own echo, a cough and "mm-hm" do not interrupt;
- the order is: cut the voice and bump the epoch, THEN start the turn, from frames kept since before the sound began;
- a weaker sound opens a stop-only probe, which acts on a stop phrase (built-in or configured) and drops anything else
  unheard, and which the ordinary turn replaces if the sound turns out to be the person;
- the recorded lessons still hold: one end per utterance, no dropped segments, no counters spoken, self-correction merges.
"""

from __future__ import annotations

import json
import sys
import threading
from array import array
from pathlib import Path
from types import SimpleNamespace

from guards_voice import no_real_audio  # noqa: F401

sys.path.insert(0, str(Path(__file__).parent.parent))

from test_assistant_fakes import WAIT, FakeConfig, FakeLLM, FakeTTS, wait_until
from test_voice_loop import ScriptedDevice, quiet, vad

from glide.assistant.audio_io import rms
from glide.assistant.core import IO, Assistant
from glide.providers.base import Transcript
from glide.speech.echo import EchoCanceller, EchoStats
from glide.speech.session import watch
from glide.speech.turns import (
    BARGE_LEAD_FRAMES,
    HISTORY_FRAMES,
    PROBE_COOLDOWN_S,
    BargeInGate,
    TurnAborted,
    VoiceLoop,
    format_status,
)
from glide.speech.vad import FRAME_BYTES, FRAME_S


def tone(amplitude: int) -> bytes:
    """One frame whose RMS is `amplitude`."""
    return array("h", [amplitude]).tobytes() * (FRAME_BYTES // 2)


def stats(*, expected=50.0, near=0.0, erle=25.0, measured=True) -> EchoStats:
    return EchoStats(near_rms=near, expected_residual_rms=expected, erle_db=erle if measured else None, measured=measured)


# -- the gate ---------------------------------------------------------------------------------------


def feed(gate, frames, level, echo, probability=1.0):
    return [gate.feed(probability, level, echo) for _ in range(frames)]


def test_thresholds_follow_the_settings_in_whole_frames():
    gate = BargeInGate(min_voiced_ms=190, margin_db=8, min_erle_db=6)
    assert gate.min_voiced_frames == 6 and gate.thresholds["min_voiced_ms"] == 192
    assert gate.probe_margin_db == 5 and gate.probe_frames == 3
    assert BargeInGate(min_voiced_ms=96, margin_db=3).probe_margin_db == 3  # never below 3 dB


def test_a_voice_over_a_loud_enough_margin_confirms_on_the_frame_that_makes_the_minimum():
    gate = BargeInGate()
    verdicts = feed(gate, gate.min_voiced_frames, 1000.0, stats(expected=50.0))
    assert [v.confirm for v in verdicts] == [False] * (gate.min_voiced_frames - 1) + [True]
    assert gate.counters["confirmed"] == 1


def test_a_sound_shorter_than_the_minimum_is_backchannel_and_does_not_confirm():
    gate = BargeInGate()
    verdicts = feed(gate, gate.min_voiced_frames - 1, 1000.0, stats()) + feed(gate, 6, 0.0, stats(), probability=0.0)
    assert not any(v.confirm for v in verdicts)
    assert gate.counters["rejected"] == 1 and gate.counters["confirmed"] == 0


def test_a_brief_dropout_inside_speech_does_not_start_the_count_again_but_a_long_one_does():
    gate = BargeInGate()
    echo = stats()
    pattern = [1000.0] * 3 + [0.0] * 2 + [1000.0] * 3  # two dropped frames inside the allowed gap
    assert any(gate.feed(1.0 if level else 0.0, level, echo).confirm for level in pattern)
    gate = BargeInGate()
    pattern = [1000.0] * 3 + [0.0] * 3 + [1000.0] * 3
    assert not any(gate.feed(1.0 if level else 0.0, level, echo).confirm for level in pattern)


def test_nothing_confirms_before_the_canceller_has_measured_itself_or_removes_enough():
    gate = BargeInGate(min_erle_db=6)
    assert not any(v.confirm for v in feed(gate, 20, 3000.0, stats(measured=False)))
    assert gate.counters["waiting_for_erle"] == 20
    assert not any(v.confirm for v in feed(BargeInGate(min_erle_db=6), 20, 3000.0, stats(erle=4.0)))


def test_the_cleaned_level_must_clear_the_echo_expected_by_the_margin():
    echo = stats(expected=500.0)  # bar at 8 dB: about 1256
    assert not any(v.confirm for v in feed(BargeInGate(margin_db=8), 20, 1100.0, echo))
    assert any(v.confirm for v in feed(BargeInGate(margin_db=8), 20, 1300.0, echo))


def test_a_frame_the_canceller_all_but_removed_is_echo_it_understood_not_a_voice():
    echo = stats(expected=50.0, near=10000.0)  # the microphone held 10000 and 1000 is left: 20 dB removed
    assert not any(v.confirm for v in feed(BargeInGate(), 20, 1000.0, echo))


def test_the_level_is_never_judged_against_less_than_the_floor():
    assert not any(v.confirm for v in feed(BargeInGate(margin_db=8), 20, 100.0, stats(expected=0.0)))


def test_without_a_canceller_the_voice_detector_and_the_duration_decide():
    gate = BargeInGate()
    assert any(v.confirm for v in feed(gate, 8, 5.0, None))
    assert not any(v.confirm for v in feed(BargeInGate(), 8, 5.0, None, probability=0.2))


def test_weaker_evidence_asks_for_a_stop_probe_and_stronger_evidence_does_not():
    weak = feed(BargeInGate(), 6, 420.0, stats(expected=200.0))  # bars: weak 356, strong 502
    assert [v.probe for v in weak].index(True) == 2 and not any(v.confirm for v in weak)
    loud = feed(BargeInGate(), 6, 1500.0, stats(expected=200.0))
    assert not any(v.probe for v in loud) and any(v.confirm for v in loud)


def test_the_gate_freezes_the_cancellers_statistics_while_a_sound_louder_than_the_echo_is_judged_for_a_while_only():
    gate = BargeInGate()
    echo = stats(expected=100.0)
    gate.feed(0.0, 0.0, echo)
    assert not gate.armed
    gate.feed(1.0, 1000.0, echo)
    assert gate.armed
    for _ in range(40):
        gate.feed(1.0, 150.0, echo)  # voiced, a little above the echo, for longer than a statistic should be held
    assert not gate.armed


class _Scripted(EchoCanceller):
    """A canceller that hands back what it is told: the loud voice survives it, the echo is removed."""

    name = "scripted"

    def __init__(self) -> None:
        super().__init__()
        self.out = b""

    def _process(self, near, far):
        return self.out

    def _reset_state(self) -> None:
        pass


def test_the_first_frames_of_a_voice_over_the_echo_are_judged_before_they_teach_the_canceller_anything():
    """PR14-4175428961: the canceller learned from the voice frame before the gate read its numbers, so one frame took
    the ERLE from 29 dB under the bar and every later frame was 'waiting for erle': a person talking over Glide never got in."""
    canceller, gate = _Scripted(), BargeInGate()
    far = tone(3000)
    for _ in range(80):  # echo only: 1500 in the microphone, 50 left of it
        canceller.out = tone(50)
        gate.feed(0.0, rms(canceller.process(tone(1500), far)), canceller.stats)
    assert canceller.stats.erle_db > 25
    verdicts = []
    for _ in range(gate.min_voiced_frames):  # a voice 12 dB over the echo, which no canceller removes
        canceller.out = tone(6000)
        out = canceller.process(tone(7500), far)
        verdicts.append(gate.feed(1.0, rms(out), canceller.stats))
        canceller.hold = gate.armed  # what the loop does after every frame
    assert verdicts[-1].confirm, canceller.stats


def test_reset_forgets_the_sound_being_judged():
    gate = BargeInGate()
    feed(gate, 3, 1000.0, stats())
    gate.reset()
    assert not gate.armed and gate.counters["rejected"] == 1


# -- the loop ---------------------------------------------------------------------------------------


class StopSTT:
    """One scripted transcript per stream, in order; `partials[i]` is said (as an interim result) before the final."""

    def __init__(self, finals, partials=None):
        self.finals = list(finals)
        self.partials = partials or {}
        self.audio: dict[int, bytes] = {}
        self.aborted: list[int] = []
        self.events: list[str] = []
        self._lock = threading.Lock()
        self.count = 0

    @property
    def streams(self):
        return list(self.audio)

    def stream(self, chunks, *, sample_rate=16000, language=None):
        with self._lock:
            index = self.count
            self.count += 1
            self.events.append("stream")
        heard = bytearray()
        self.audio[index] = heard
        try:
            for chunk in chunks:
                heard.extend(chunk)
        except TurnAborted:
            self.aborted.append(index)
            raise
        if index in self.partials:
            yield Transcript(self.partials[index], "en", "fake:stt", "fake", 0.0, partial=True)
        yield Transcript(self.finals[index], "en", "fake:stt", "fake", 0.0)

    def transcribe(self, audio, **kw):
        raise AssertionError("no batch request expected")


class EchoDevice(ScriptedDevice):
    """A scripted device that reports echo statistics (`stats` is an EchoStats, or a callable of the frame index) and
    goes quiet when cancelled, as the real one does."""

    def __init__(self, script, *, stats=None, active=True, playing=True):
        super().__init__(script, playing=playing)
        self.stats = stats
        self.echo_active = active
        self.index = 0
        self.holds: list[bool] = []
        self.cancels = 0
        self.now = 1000.0

    @property
    def echo(self):
        return self.stats(self.index) if callable(self.stats) else self.stats

    def read(self, timeout=None):
        frame = super().read(timeout)
        self.index += 1
        self.now += FRAME_S
        return frame

    def hold_echo_stats(self, hold):
        self.holds.append(hold)

    def cancel(self):
        self.cancels += 1
        self.playing = False
        self.echo_active = False

    def play(self, pcm, sample_rate): ...

    def wait_idle(self, timeout=None):
        return True

    def close(self): ...


def rig(script, stt, *, stats=None, active=True, playing=True, stop_phrases=(), **loop_kw):
    llm, tts = FakeLLM(), FakeTTS()
    config = FakeConfig(llm=llm, stt=stt, tts=tts)
    device = EchoDevice(script, stats=stats, active=active, playing=playing)
    heard, warned = [], []
    io = IO(player=device, heard=heard.append, warn=warned.append)
    assistant = Assistant(config, io=io, extra_stop_phrases=stop_phrases)
    loop = VoiceLoop(assistant, device, vad, clock=lambda: device.now, **loop_kw)
    device.loop = loop
    loop.stop_soon = loop._stop.set
    return SimpleNamespace(loop=loop, device=device, assistant=assistant, heard=heard, warned=warned, stt=stt, config=config)


def run(r):
    r.loop.run()
    r.loop.join_turns(WAIT)
    return r


def test_glides_own_echo_never_interrupts_however_long_it_lasts():
    stt = StopSTT(["never asked"])
    r = run(rig([tone(1000)] * 60 + quiet(700), stt, stats=stats(expected=800.0)))  # bars: strong 2000, weak 1424
    assert r.assistant._epoch == 0 and r.device.cancels == 0
    assert stt.count == 0 and r.heard == []
    assert r.loop.status()["barge_in"]["confirmed"] == 0 and r.loop.status()["barge_in"]["probes"] == 0


def test_a_voice_over_the_echo_cuts_the_voice_bumps_the_epoch_and_is_not_dropped_by_it():
    stt = StopSTT(["what time is it"])
    r = run(rig(quiet(320) + [tone(1000)] * 12 + quiet(700), stt, stats=stats(expected=50.0)))
    assert r.device.cancels >= 1 and r.assistant._epoch == 1
    assert r.heard == ["what time is it"]  # the epoch moved first, so the turn that caused it is the one that counts


def test_the_voice_is_cut_before_the_turn_starts_listening():
    stt = StopSTT(["what time is it"])
    r = rig(quiet(320) + [tone(1000)] * 12 + quiet(700), stt, stats=stats(expected=50.0))
    original = r.assistant.interrupt_speech
    r.assistant.interrupt_speech = lambda **kw: (stt.events.append("interrupt"), original(**kw))[1]
    run(r)
    assert stt.events == ["interrupt", "stream"]


def test_the_turn_starts_from_a_lead_in_before_the_sound_so_the_first_words_are_kept():
    stt = StopSTT(["what time is it"])
    run(rig(quiet(960) + [tone(1000)] * 12 + quiet(700), stt, stats=stats(expected=50.0)))
    audio = bytes(stt.audio[0])
    lead = BARGE_LEAD_FRAMES * FRAME_BYTES
    assert audio[:lead] == bytes(lead)  # the lead-in: silence before the voice
    assert audio[lead : lead + 6 * FRAME_BYTES] == tone(1000) * 6  # every frame of the sound up to the decision
    assert audio.count(tone(1000)) == 12  # and the rest of it, each once


def test_the_lead_in_covers_a_sound_that_took_a_while_to_qualify():
    """The voice rises: weak frames first, then loud ones. The turn includes the weak start."""
    stt = StopSTT(["what time is it"])
    run(rig(quiet(320) + [tone(150)] * 4 + [tone(1000)] * 10 + quiet(700), stt, stats=stats(expected=50.0)))
    assert bytes(stt.audio[0]).count(tone(150)) == 4


def test_a_short_sound_is_backchannel_nothing_is_cut_and_nothing_is_heard():
    stt = StopSTT(["mm"])
    r = run(rig(quiet(320) + [tone(1000)] * 4 + quiet(1000), stt, stats=stats(expected=50.0)))
    assert r.assistant._epoch == 0 and r.device.cancels == 0 and stt.count == 0
    assert r.loop.status()["barge_in"]["rejected"] == 1


def test_no_voice_interrupts_before_the_canceller_has_measured_itself_and_the_next_one_after_it_does():
    unmeasured = stats(measured=False)
    stt = StopSTT(["first"])
    r = run(rig([tone(1000)] * 20 + quiet(700), stt, stats=unmeasured))
    assert r.assistant._epoch == 0 and stt.count == 0
    assert r.loop.status()["barge_in"]["waiting_for_erle"] == 20

    stt = StopSTT(["second"])
    r = run(rig([tone(1000)] * 20 + quiet(700), stt, stats=lambda i: unmeasured if i < 8 else stats(expected=50.0)))
    assert r.assistant._epoch == 1 and r.heard == ["second"]


def test_when_glide_is_not_speaking_the_voice_detector_alone_starts_the_turn_as_before():
    stt = StopSTT(["hello"])
    r = run(rig([tone(1000)] * 3 + quiet(700), stt, stats=stats(expected=800.0), active=False, playing=False))
    assert r.heard == ["hello"] and r.assistant._epoch == 0  # nothing was barged into


def test_with_no_canceller_a_headset_hears_a_voice_after_the_minimum_duration_and_not_before():
    short = run(rig([tone(1000)] * 4 + quiet(700), StopSTT(["x"]), stats=None))
    assert short.assistant._epoch == 0 and short.device.cancels == 0
    long = run(rig([tone(1000)] * 10 + quiet(700), StopSTT(["x"]), stats=None))
    assert long.assistant._epoch == 1 and long.device.cancels >= 1


def test_the_cancellers_statistics_are_held_while_a_voice_is_judged_and_released_after():
    r = run(rig(quiet(320) + [tone(1000)] * 12 + quiet(700), StopSTT(["x"]), stats=stats(expected=50.0)))
    assert r.device.holds[0] is True and r.device.holds[-1] is False


def test_a_correction_after_a_barge_in_is_one_turn_through_the_merge_window():
    stt = StopSTT(["open safari no wait chrome"])
    script = quiet(320) + [tone(1000)] * 10 + quiet(1000) + [tone(1000)] * 8 + quiet(2400)
    r = run(rig(script, stt, stats=stats(expected=50.0), merge_window_s=1.5))
    assert stt.count == 1 and r.heard == ["open safari no wait chrome"]
    assert bytes(stt.audio[0]).count(tone(1000)) == 18  # both halves reached the one request


def test_two_barge_ins_in_a_row_are_each_heard_none_dropped():
    stt = StopSTT(["first", "second"])
    first = quiet(320) + [tone(1000)] * 10 + quiet(700)
    second = quiet(100) + [tone(1000)] * 10 + quiet(700)
    r = rig(first + second, stt, stats=stats(expected=50.0))

    def glide_answers_again(left):  # as the second sound begins, Glide is speaking once more
        if left == len(second):
            r.device.playing = r.device.echo_active = True

    r.device.on_frame = glide_answers_again
    run(r)
    assert r.heard == ["first", "second"] and r.assistant._epoch >= 1 and r.device.cancels >= 2


# -- the stop probe ---------------------------------------------------------------------------------

WEAK = tone(420)  # with expected=200: above the probe's bar (about 356), below the policy's (about 502)
WEAK_ECHO = stats(expected=200.0)


def test_a_quiet_stop_opens_a_probe_that_stops_everything_and_never_confirms_a_barge_in():
    stt = StopSTT(["stop"])
    r = run(rig(quiet(320) + [WEAK] * 8 + quiet(700), stt, stats=WEAK_ECHO))
    assert r.loop.status()["barge_in"]["probes"] == 1 and r.loop.status()["barge_in"]["confirmed"] == 0
    assert r.heard == ["stop"] and r.device.cancels >= 1 and r.assistant._epoch >= 1
    assert r.config.calls.llm == 0  # a stop is the router's fast path: no model is asked


def test_a_stop_that_is_still_being_said_silences_the_voice_at_once():
    stt = StopSTT(["stop"], partials={0: "stop"})
    r = run(rig(quiet(320) + [WEAK] * 8 + quiet(700), stt, stats=WEAK_ECHO))
    assert r.device.cancels >= 1


def test_anything_else_a_probe_hears_is_dropped_unheard():
    stt = StopSTT(["what is the weather"], partials={0: "what is"})
    r = rig(quiet(320) + [WEAK] * 8 + quiet(700), stt, stats=WEAK_ECHO)
    interim = []
    r.assistant.io.partial = interim.append
    run(r)
    assert stt.count == 1  # it was listened to
    assert r.heard == [] and interim == []  # not shown as it was being heard either
    assert r.device.cancels == 0 and r.assistant._epoch == 0
    assert r.config.calls.llm == 0 and r.config.calls.tts == 0  # not routed, not answered, not spoken


def test_glides_own_words_transcribed_by_a_probe_are_never_a_request():
    """The probe may hear Glide's own echo. Whatever it transcribes, if it is not a stop, nothing happens."""
    stt = StopSTT(["the museum opens at nine"])
    r = run(rig(quiet(320) + [WEAK] * 8 + quiet(700), stt, stats=WEAK_ECHO))
    assert r.heard == [] and r.config.calls.llm == 0 and not r.assistant.busy


def test_a_configured_stop_phrase_stops_through_a_probe():
    stt = StopSTT(["hold on please"])
    r = run(rig(quiet(320) + [WEAK] * 8 + quiet(700), stt, stats=WEAK_ECHO, stop_phrases=("hold on please",)))
    assert r.heard == ["hold on please"] and r.device.cancels >= 1


def test_without_that_phrase_configured_the_same_words_are_not_a_stop():
    stt = StopSTT(["hold on please"])
    r = run(rig(quiet(320) + [WEAK] * 8 + quiet(700), stt, stats=WEAK_ECHO))
    assert r.heard == [] and r.device.cancels == 0


def test_a_sound_that_turns_out_to_be_the_person_replaces_the_probe_with_the_ordinary_turn():
    stt = StopSTT(["ignored", "what time is it"])
    r = run(rig(quiet(320) + [WEAK] * 4 + [tone(1000)] * 10 + quiet(700), stt, stats=WEAK_ECHO))
    assert stt.aborted == [0]  # the probe's audio was thrown away, not submitted
    assert r.heard == ["what time is it"] and r.assistant._epoch == 1


def test_probes_are_one_at_a_time_and_rate_limited():
    weak_run = [WEAK] * 6
    gap = lambda frames: [bytes(FRAME_BYTES)] * frames  # noqa: E731
    script = quiet(320) + weak_run + gap(14) + weak_run + gap(14) + weak_run + gap(60) + weak_run + quiet(700)
    # the probe ends 12 quiet frames after its sound; the next sound begins 2 frames later, inside the cooldown
    r = rig(script, StopSTT(["x"] * 4), stats=WEAK_ECHO)
    # a probe is over when its transcript is: let each one finish before the next sound, as a real transcriber would
    r.device.on_frame = lambda left: (
        r.loop._probe is None and wait_until(lambda: all(t.heard.is_set() for t in r.loop._turns if t.probe))
    )
    run(r)
    assert r.loop.status()["barge_in"]["probes"] == 2
    assert 60 * FRAME_S > PROBE_COOLDOWN_S  # the long quiet really is longer than the cooldown


class _HeldSTT(StopSTT):
    """Like StopSTT, but the first stream's transcript is held back until `release` is set."""

    def __init__(self, finals):
        super().__init__(finals)
        self.release = threading.Event()

    def stream(self, chunks, *, sample_rate=16000, language=None):
        for transcript in super().stream(chunks, sample_rate=sample_rate, language=language):
            if len(self.audio) == 1 and not transcript.partial:
                self.release.wait(WAIT)
            yield transcript


def test_a_second_probe_never_opens_while_the_first_is_still_being_transcribed():
    """PR14-4175586153: the cooldown ran from the end of the probe's audio, not of its transcript, so a slow transcriber
    could have two streams of stop-only audio open at once."""
    stt = _HeldSTT(["x", "y"])
    weak_run, gap = [WEAK] * 6, lambda frames: [bytes(FRAME_BYTES)] * frames
    script = quiet(320) + weak_run + gap(14) + gap(60) + weak_run + gap(30)  # the second sound is long after the cooldown
    r = rig(script, stt, stats=WEAK_ECHO)
    r.device.on_frame = lambda left: stt.release.set() if left == 0 else None
    run(r)
    assert stt.count == 1 and r.loop.status()["barge_in"]["probes"] == 1  # the first one was still waiting for its transcript


def test_a_probe_never_outlives_the_sound_it_listens_to():
    stt = StopSTT(["x"])
    r = run(rig(quiet(320) + [WEAK] * 100 + quiet(700), stt, stats=WEAK_ECHO))
    assert stt.count == 1 and r.loop.status()["barge_in"]["probes"] == 1


# -- assistant: a stop-only listen ------------------------------------------------------------------


def make_assistant(stt, **kw):
    llm, tts, player = FakeLLM(), FakeTTS(), EchoDevice([])
    heard = []
    config = FakeConfig(llm=llm, stt=stt, tts=tts)
    assistant = Assistant(config, io=IO(player=player, heard=heard.append), **kw)
    return assistant, config, heard, player


def test_stop_only_acts_on_a_stop_and_on_nothing_else():
    assistant, config, heard, player = make_assistant(StopSTT(["stop talking"]))
    assert assistant.handle_audio([b"\x01\x00" * 800], stop_only=True).route == "stop"
    assert heard == ["stop talking"] and player.cancels >= 1

    assistant, config, heard, player = make_assistant(StopSTT(["tell me a joke"]))
    reply = assistant.handle_audio([b"\x01\x00" * 800], stop_only=True)
    assert reply.route == "none" and heard == [] and config.calls.llm == 0 and player.cancels == 0


def test_a_configured_stop_phrase_is_part_of_the_fast_path_for_text_too():
    assistant, config, _, player = make_assistant(StopSTT([]), extra_stop_phrases=("hold on please",))
    assert assistant.handle_text("Hold on, please!").route == "stop"
    assert config.calls.llm == 0 and player.cancels >= 1
    assistant, config, _, player = make_assistant(StopSTT([]))
    assert assistant.handle_text("hold on please").route != "stop"


def test_interrupt_speech_can_leave_pending_requests_alone():
    assistant, _, _, player = make_assistant(StopSTT([]))
    assistant.interrupt_speech(drop_pending=False)
    assert assistant._epoch == 0 and player.cancels == 1
    assistant.interrupt_speech()
    assert assistant._epoch == 1 and player.cancels == 2


# -- what can be read off the loop ------------------------------------------------------------------


def test_the_status_is_numbers_and_thresholds_only_and_can_be_printed_as_it_is():
    r = run(rig(quiet(320) + [tone(1000)] * 12 + quiet(700), StopSTT(["secret words"]), stats=stats(expected=50.0, erle=21.37)))
    status = r.loop.status()
    json.dumps(status)  # plain data
    assert status["echo"]["erle_db"] == 21.37 or status["echo"]["measured"]
    assert set(status["barge_in"]["thresholds"]) == {"min_voiced_ms", "probe_ms", "margin_db", "probe_margin_db", "min_erle_db"}

    def leaves(node, path=""):
        if isinstance(node, dict):
            for key, value in node.items():
                yield from leaves(value, f"{path}/{key}")
        else:
            yield path, node

    texts = [path for path, value in leaves(status) if isinstance(value, str)]
    assert all(isinstance(v, (int, float, bool, str)) or v is None for _, v in leaves(status))
    assert texts == []  # no text at all here (with a real device the canceller's name is the one string)
    assert "secret" not in json.dumps(status)


def test_the_history_is_long_enough_for_the_lead_in_and_the_longest_decision():
    gate = BargeInGate(min_voiced_ms=1000)
    assert gate.min_voiced_frames + BARGE_LEAD_FRAMES + 2 * 2 <= HISTORY_FRAMES  # the lead-in, the sound and its gaps


def test_the_status_line_is_readable_and_still_content_free():
    r = run(rig(quiet(320) + [tone(1000)] * 12 + quiet(700), StopSTT(["secret words"]), stats=stats(expected=50.0, erle=21.4)))
    line = format_status(r.loop.status())
    assert "confirmed 1" in line and "erle 21.4 dB" in line and "needs 192 ms, 8.0 dB over the echo, erle 6.0 dB" in line
    assert "secret" not in line and "\n" not in line
    assert format_status(rig([], StopSTT([])).loop.status()).startswith("echo: none")


def test_watch_shows_the_status_while_the_loop_runs_and_stops_with_it():
    shown, naps = [], []
    r = rig([], StopSTT([]), stats=stats())
    ticks = iter([True, True, False])
    watch(r.loop, show=shown.append, interval_s=0.5, sleep=naps.append, running=lambda: next(ticks))
    assert len(shown) == 2 and naps == [0.5, 0.5] and all(line.startswith("echo:") for line in shown)


def test_a_loop_that_was_never_started_has_not_ended_and_one_whose_thread_finished_has():
    r = rig([], StopSTT([]), stats=stats())
    assert not r.loop.ended
    r.loop.start()
    r.loop.join_turns(WAIT)
    r.loop._thread.join(WAIT)
    assert r.loop.ended

"""Mutation-testing follow-up for `glide/speech/turns.py` (see docs/MUTATION_REPORT.md).

Two groups. The first drives `BargeInGate`, which is pure policy, with synthetic echo statistics chosen so that a level
sits exactly on a threshold (a margin of 20 dB is exactly a factor of 10): every comparison of the gate is then pinned
from both sides, and so are the shipped numbers, which docs/voice-echo-cancellation.md documents. The second drives the
hands-free loop over the scripted device of tests/speech/test_voice_loop.py and pins what its safety rules need beyond
`heard == []`: a half-said command is discarded so that its transcriber is released (not left waiting for audio), the
second half of a corrected utterance loses no frame, no empty chunk is sent, a probe is bounded and rate limited, and an
idle loop switches the microphone off only after the configured time. No sleeps; no sound card.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent / "speech"))  # the voice tests' scripted device lives there

from guards_voice import no_real_audio  # noqa: F401  (autouse: see tests/guards_voice.py)
from test_assistant_fakes import FakeLLM, route_json
from test_voice_loop import LOUD, QUIET, ScriptedSTT, quiet, rig, run, speech

from glide.providers.errors import ProviderError
from glide.speech import turns, vad
from glide.speech.audio import DeviceFault
from glide.speech.echo import EchoStats
from glide.speech.settings import SpeechSettings
from glide.speech.turns import BargeInGate, _Run, _Turn
from glide.speech.vad import FRAME_BYTES, FRAME_S

WAIT = 3.0

# -- the shipped policy ---------------------------------------------------------------------------------------------


def test_the_shipped_barge_in_numbers_are_the_documented_ones():
    """A change of any of these retunes a documented, measured policy: change it on purpose, with the sweeps and the docs."""
    shipped = SpeechSettings()  # the three the person can change live in the [speech] table; this is their one home
    assert (shipped.barge_min_voiced_ms, shipped.barge_margin_db, shipped.barge_min_erle_db) == (190, 8.0, 6.0)
    gate = BargeInGate()  # and an unconfigured gate takes exactly those
    assert (gate.min_voiced_frames, gate.margin_db, gate.min_erle_db) == (6, 8.0, 6.0)
    assert (turns.PROBE_FRAMES, turns.PROBE_MARGIN_LESS_DB, turns.PROBE_COOLDOWN_S) == (3, 3.0, 1.0)
    assert (turns.PROBE_QUIET_FRAMES, turns.PROBE_MAX_FRAMES, turns.HOLD_MAX_FRAMES) == (12, 94, 31)
    assert (turns.SUSPECT_RATIO, turns.SOUND_GAP_FRAMES, turns.RUN_GAP_FRAMES) == (1.4, 12, 2)
    assert (turns.RETAINED_MIN, turns.MIN_LEVEL_RMS, turns.BARGE_LEAD_FRAMES, turns.HISTORY_FRAMES) == (0.25, 60.0, 12, 64)
    assert turns.POLL_S == 0.05 and vad.START_PROBABILITY == 0.5 and FRAME_S == 0.032


# -- the gate: every comparison, from both sides ----------------------------------------------------------------------


def echo(*, expected: float = 100.0, near: float = 0.0, erle: float | None = 25.0, measured: bool = True) -> EchoStats:
    return EchoStats(near_rms=near, expected_residual_rms=expected, erle_db=erle, measured=measured)


def test_a_frame_is_voiced_at_exactly_the_start_probability_and_not_below_it():
    gate = BargeInGate()
    at = gate.feed(vad.START_PROBABILITY, 0.0, None)
    below = gate.feed(vad.START_PROBABILITY - 0.0001, 0.0, None)
    assert (at.weak, at.strong) == (True, True)  # no canceller: the voice detector alone is the evidence
    assert (below.weak, below.strong) == (False, False)


def test_the_canceller_must_have_removed_at_least_the_minimum_erle_and_exactly_the_minimum_counts():
    gate = BargeInGate(min_erle_db=6.0)
    exactly = gate.feed(1.0, 10_000.0, echo(erle=6.0))
    just_under = gate.feed(1.0, 10_000.0, echo(erle=5.99))
    unmeasured = gate.feed(1.0, 10_000.0, echo(erle=30.0, measured=False))
    assert (exactly.strong, just_under.strong, unmeasured.strong) == (True, False, False)
    assert gate.counters["waiting_for_erle"] == 2


def test_a_measured_canceller_with_no_erle_figure_counts_as_removing_nothing():
    gate = BargeInGate(min_erle_db=0.5)
    assert gate.feed(1.0, 10_000.0, echo(erle=None)).strong is False
    assert gate.counters["waiting_for_erle"] == 1


def test_the_strong_margin_is_judged_to_the_exact_level_with_a_floor_of_the_expected_residual():
    gate = BargeInGate(margin_db=20.0)  # exactly a factor of 10: with 100 of expected echo the level must reach 1000
    assert gate.feed(1.0, 1000.0, echo()).strong is True
    assert gate.feed(1.0, 999.0, echo()).strong is False


def test_the_probe_margin_is_judged_to_the_exact_level_too():
    gate = BargeInGate(margin_db=23.0)  # the probe margin is 3 dB less: exactly 20 dB, a factor of 10
    assert gate.probe_margin_db == 20.0
    assert gate.feed(1.0, 1000.0, echo()).weak is True
    assert gate.feed(1.0, 999.0, echo()).weak is False
    assert BargeInGate(margin_db=5.0).probe_margin_db == 3.0  # never below 3 dB
    assert BargeInGate().probe_margin_db == 5.0


def test_a_level_is_never_judged_against_less_than_the_floor_and_must_keep_a_quarter_of_the_microphone():
    gate = BargeInGate(margin_db=12.0)  # strong from 60 * 3.98 = 239
    assert gate.feed(1.0, 300.0, echo(expected=0.0, near=1000.0)).strong is True
    assert gate.feed(1.0, 300.0, echo(expected=0.0, near=1200.0)).strong is True  # 300 is exactly a quarter of 1200
    assert gate.feed(1.0, 300.0, echo(expected=0.0, near=1201.0)).strong is False  # echo it understood, not a voice
    assert gate.feed(1.0, 200.0, echo(expected=0.0, near=0.0)).strong is False  # below 60 * margin whatever the echo


def test_a_louder_than_expected_voiced_frame_arms_the_hold_at_exactly_the_suspect_ratio():
    floor = 100.0
    gate = BargeInGate()
    gate.feed(1.0, floor * turns.SUSPECT_RATIO, echo(expected=floor))
    assert gate.armed
    gate = BargeInGate()
    gate.feed(1.0, floor * turns.SUSPECT_RATIO - 0.001, echo(expected=floor))
    assert not gate.armed
    gate = BargeInGate()
    gate.feed(1.0, floor * 1.2, echo(expected=floor))  # 20 % over is not enough to freeze the canceller's statistics
    assert not gate.armed


def test_the_hold_lasts_for_the_maximum_number_of_frames_and_not_one_more():
    gate = BargeInGate()
    for frame in range(turns.HOLD_MAX_FRAMES):
        gate.feed(1.0, 150.0, echo(expected=100.0))
        assert gate.armed, f"released early at frame {frame + 1}"
    gate.feed(1.0, 150.0, echo(expected=100.0))
    assert not gate.armed


def weak_frame(gate: BargeInGate):
    return gate.feed(1.0, 200.0, echo(expected=100.0))  # over the 5 dB probe margin, under the 8 dB strong one


def quiet_frame(gate: BargeInGate):
    return gate.feed(0.0, 0.0, echo(expected=100.0))


def test_the_onset_is_counted_back_to_the_first_evidence_of_the_sound_through_short_dips():
    gate = BargeInGate()
    assert gate.onset_back == 0
    for _ in range(3):
        weak_frame(gate)
    assert gate.onset_back == 2  # three frames of one sound
    gate = BargeInGate()
    weak_frame(gate)
    for _ in range(6):
        quiet_frame(gate)
    weak_frame(gate)
    assert gate.onset_back == 7  # a dip of 6 frames is part of the same sound


def test_a_sound_goes_on_through_a_dip_one_frame_shorter_than_the_gap_and_a_new_one_starts_after_the_gap():
    gate = BargeInGate()
    weak_frame(gate)
    for _ in range(turns.SOUND_GAP_FRAMES - 1):
        quiet_frame(gate)
    weak_frame(gate)
    assert gate.onset_back == turns.SOUND_GAP_FRAMES  # still the first sound
    gate = BargeInGate()
    weak_frame(gate)
    for _ in range(turns.SOUND_GAP_FRAMES):
        quiet_frame(gate)
    weak_frame(gate)
    assert gate.onset_back == 0  # a new sound begins at this frame


def test_reset_forgets_the_run_the_onset_and_the_count_toward_a_confirmation():
    gate = BargeInGate()
    for _ in range(gate.min_voiced_frames - 1):
        gate.feed(1.0, 10_000.0, echo())
    assert gate.onset_back > 0
    gate.reset()
    assert gate.onset_back == 0 and not gate.armed
    assert gate.feed(1.0, 10_000.0, echo()).confirm is False  # the count starts again from this frame
    for _ in range(gate.min_voiced_frames - 2):
        assert gate.feed(1.0, 10_000.0, echo()).confirm is False
    assert gate.feed(1.0, 10_000.0, echo()).confirm is True


def test_a_sound_confirms_on_the_frame_that_makes_the_minimum_and_the_counters_tell_it():
    gate = BargeInGate()
    assert gate.counters == {"frames": 0, "candidates": 0, "rejected": 0, "confirmed": 0, "waiting_for_erle": 0}
    verdicts = [gate.feed(1.0, 10_000.0, echo()) for _ in range(gate.min_voiced_frames)]
    assert [v.confirm for v in verdicts] == [False] * (gate.min_voiced_frames - 1) + [True]
    assert gate.counters["confirmed"] == 1 and gate.counters["frames"] == gate.min_voiced_frames
    for _ in range(turns.RUN_GAP_FRAMES + 2):
        quiet_frame(gate)
    assert gate.counters["rejected"] == 0  # a confirmed sound is not a rejected candidate afterwards


def test_a_weak_sound_is_one_candidate_and_is_rejected_once_when_it_ends():
    gate = BargeInGate()
    verdicts = [weak_frame(gate) for _ in range(gate.probe_frames)]
    assert [v.probe for v in verdicts] == [False] * (gate.probe_frames - 1) + [True]
    assert gate.counters["candidates"] == 1 and gate.counters["rejected"] == 0
    weak_frame(gate)  # still the same candidate
    assert gate.counters["candidates"] == 1
    for _ in range(turns.RUN_GAP_FRAMES):
        quiet_frame(gate)
    assert gate.counters["rejected"] == 0  # a dip of this length does not end the run
    quiet_frame(gate)
    assert gate.counters["rejected"] == 1
    for _ in range(5):
        quiet_frame(gate)
    assert gate.counters["rejected"] == 1  # and it is rejected once


def test_an_evidence_run_allows_dips_and_reports_how_long_it_was_when_it_ends():
    run_ = _Run()
    for frame in range(5, 8):
        run_.feed(frame, True)
    assert (run_.count, run_.start, run_.gap, run_.ended_with) == (3, 5, 0, 0)
    for frame in range(8, 8 + turns.RUN_GAP_FRAMES):
        run_.feed(frame, False)
    assert (run_.count, run_.gap, run_.ended_with) == (3, turns.RUN_GAP_FRAMES, 0)
    run_.feed(20, True)  # evidence after a dip carries on the same run
    assert (run_.count, run_.start, run_.gap) == (4, 5, 0)
    for frame in range(21, 21 + turns.RUN_GAP_FRAMES + 1):
        run_.feed(frame, False)
    assert (run_.count, run_.gap, run_.ended_with) == (0, 0, 4)
    run_.feed(40, False)
    assert run_.ended_with == 0  # "the run that just ended" is reported for one frame only
    run_.feed(41, True)
    assert (run_.count, run_.start) == (1, 41)
    run_.clear()
    assert (run_.count, run_.gap, run_.ended_with) == (0, 0, 0)


def test_the_thresholds_are_in_whole_frames_and_never_less_than_one():
    gate = BargeInGate()
    assert gate.thresholds == {
        "min_voiced_ms": 192,
        "probe_ms": 96,
        "margin_db": 8.0,
        "probe_margin_db": 5.0,
        "min_erle_db": 6.0,
    }
    tiny = BargeInGate(min_voiced_ms=0)
    assert tiny.min_voiced_frames == 1 and tiny.probe_frames == 1
    assert tiny.thresholds["min_voiced_ms"] == 32 and tiny.thresholds["probe_ms"] == 32
    assert BargeInGate(min_voiced_ms=1000).min_voiced_frames == 32  # 1000 ms is 31.25 frames: rounded up


# -- the loop: what `heard == []` does not show ------------------------------------------------------------------------


def alive(r) -> list[bool]:
    return [t.thread.is_alive() for t in r.loop._turns]


def test_a_command_half_said_when_the_loop_stops_is_discarded_and_its_transcriber_released():
    stt = ScriptedSTT(["never"])
    r = run(rig([*speech()], stt))  # the script ends mid-utterance, which stops the loop
    assert r.heard == [] and stt.ended == []  # its audio never ended normally: nothing was submitted
    assert len(r.loop._turns) == 1 and alive(r) == [False]  # and its worker was told, so it is not left waiting for audio


def test_a_microphone_fault_releases_the_turns_transcriber_too():
    stt = ScriptedSTT(["never"])
    r = run(rig([*speech(), DeviceFault("Microphone overflow; the incomplete command was discarded.")], stt))
    assert r.heard == [] and stt.ended == [] and alive(r) == [False]
    assert r.loop.failure is not None


def test_an_utterance_that_is_too_long_is_discarded_and_its_transcriber_released():
    stt = ScriptedSTT(["never"])
    r = rig([*speech(8)], stt)
    r.loop._detector.max_frames = 3
    run(r)
    assert r.heard == [] and stt.ended == [] and r.loop._turns and not any(alive(r))
    assert any("seconds" in w for w in r.warned)


def test_a_corrected_utterance_sends_every_frame_once_the_pause_the_window_and_both_halves():
    stt = ScriptedSTT(["open safari no chrome"])
    run(rig([*quiet(320), *speech(5), *quiet(700), *speech(5), *quiet(700), *quiet(1600)], stt, merge_window_s=1.5))
    assert len(stt.streams) == 1 and stt.ended == [0]
    assert bytes(stt.audio[0]).count(LOUD) == 10  # both halves, the first frame of the second included
    # 8 of pre-roll (the first frame of speech is its last) + the other 4 + the 22 frames of the first pause + 5 more
    # + the 19 that make the silence + the 47 of the merge window: nothing dropped, nothing sent twice
    assert len(stt.audio[0]) // FRAME_BYTES == 8 + 4 + 22 + 5 + 19 + 47


def test_speech_that_resumes_inside_the_merge_window_is_not_cut_off_when_the_old_window_would_have_ended():
    stt = ScriptedSTT(["a long correction", "never"])
    run(rig([*quiet(320), *speech(5), *quiet(700), *speech(60), *quiet(700), *quiet(1600)], stt, merge_window_s=1.5))
    assert len(stt.streams) == 1 and stt.ended == [0]  # one request: the window did not end the turn in the middle of speech
    assert bytes(stt.audio[0]).count(LOUD) == 65


def test_a_turn_merged_from_several_stretches_is_capped_as_a_whole_not_per_stretch():
    # each stretch of speech is under the detector's own cap, but the turn they merge into is not: it is aborted, so a
    # hands-free loop cannot be kept recording (and later transcribed) without end by speech that keeps resuming
    stt = ScriptedSTT(["never", "never"])
    r = rig([*speech(30), *quiet(700), *speech(30), *quiet(700), *quiet(1600)], stt, merge_window_s=1.5)
    r.loop._detector.max_frames = 60
    run(r)
    assert stt.ended == [] and r.heard == [] and any("seconds" in w for w in r.warned)


def test_the_merged_turn_cap_is_exact_a_turn_of_exactly_the_cap_stands_and_one_frame_more_does_not():
    script = [*speech(30), *quiet(700), *speech(30), *quiet(700), *quiet(1600)]  # this script makes a turn of 148 frames
    ok = ScriptedSTT(["one request"])
    at = rig(script, ok, merge_window_s=1.5)
    at.loop._detector.max_frames = 148
    run(at)
    assert at.heard == ["one request"] and ok.ended == [0]
    over = ScriptedSTT(["never"])
    one_less = rig(script, over, merge_window_s=1.5)
    one_less.loop._detector.max_frames = 147
    run(one_less)
    assert over.ended == [] and one_less.heard == [] and any("seconds" in w for w in one_less.warned)


def test_an_utterance_too_long_ends_that_turn_and_what_follows_is_a_new_one_not_its_continuation():
    stt = ScriptedSTT(["never", "the rest"])
    r = rig([*speech(40), *quiet(700)], stt)
    r.loop._detector.max_frames = 30  # a turn counts every frame it holds (merged stretches too): the 31st is one too many
    run(r)
    # the first was aborted (nothing of it submitted); the frames after the cut are skipped until quiet, then no frame
    # of the old turn may ride into a new one: nothing is completed, and the person is told why
    assert stt.ended == [] and r.heard == [] and any("seconds" in w for w in r.warned)
    assert len(stt.streams) == 1
    # speech after the quiet that follows starts a NEW turn, which is transcribed whole
    stt2 = ScriptedSTT(["never", "the rest"])
    r2 = rig([*speech(40), *quiet(700), *speech(5), *quiet(700)], stt2)
    r2.loop._detector.max_frames = 30
    run(r2)
    assert len(stt2.streams) == 2 and stt2.ended == [1] and r2.heard == ["the rest"]
    assert bytes(stt2.audio[1]).count(LOUD) == 5  # only what was said after the cut, none of the aborted turn


def test_no_empty_chunk_is_ever_sent_to_the_transcriber():
    chunks = []

    class Recording(ScriptedSTT):
        def stream(self, chunks_in, *, sample_rate=16000, language=None):
            def tee():
                for chunk in chunks_in:
                    chunks.append(chunk)
                    yield chunk

            yield from super().stream(tee(), sample_rate=sample_rate, language=language)

    run(rig([*quiet(320), *speech(), *quiet(700)], Recording(["hi"])))
    assert chunks and all(chunk for chunk in chunks)


def test_what_was_heard_while_glide_spoke_is_kept_as_the_lead_in_of_the_turn_that_follows():
    mark = b"\x02\x00" * (FRAME_BYTES // 2)  # not voiced by the loop's detector: `vad` below says 1.0 only for LOUD

    def only_loud(frame: bytes) -> float:
        return 1.0 if frame == LOUD else 0.0

    stt = ScriptedSTT(["hello"])
    r = rig([*([mark] * 20), *speech(), *quiet(700)], stt, playing=True)
    r.loop._vad = only_loud
    frames = len(r.device.script)
    # Glide stops speaking after the 20 marked frames: the last of them are the pre-roll of the turn that starts.
    r.device.on_frame = lambda left: setattr(r.device, "playing", False) if left <= frames - 20 else None
    run(r)
    assert stt.audio and mark in bytes(stt.audio[0])


# -- probes: bounded and rate limited ---------------------------------------------------------------------------------


def probe_rig(clock=lambda: 0.0):
    r = rig([], ScriptedSTT([]), playing=True, clock=clock)
    started = []
    r.loop._start = started.append  # a probe's worker thread is not what is under test
    return r, started


def test_a_probe_ends_on_the_frame_that_makes_the_quiet_limit_and_not_before():
    r, _ = probe_rig()
    r.loop._probe = probe = _Turn(probe=True)
    for _ in range(turns.PROBE_QUIET_FRAMES - 1):
        r.loop._feed_probe(QUIET, False)
    assert r.loop._probe is probe
    r.loop._feed_probe(QUIET, False)
    assert r.loop._probe is None
    assert probe.audio.qsize() == turns.PROBE_QUIET_FRAMES + 1 and probe.audio.queue[-1] is None  # its transcript is finished


def test_evidence_resets_the_probes_quiet_count():
    r, _ = probe_rig()
    r.loop._probe = _Turn(probe=True)
    for _ in range(turns.PROBE_QUIET_FRAMES - 1):
        r.loop._feed_probe(QUIET, False)
    r.loop._feed_probe(LOUD, True)
    for _ in range(turns.PROBE_QUIET_FRAMES - 1):
        r.loop._feed_probe(QUIET, False)
    assert r.loop._probe is not None


def test_a_probe_never_outlives_the_maximum_number_of_frames_even_when_the_sound_goes_on():
    r, _ = probe_rig()
    r.loop._probe = _Turn(probe=True)
    for _ in range(turns.PROBE_MAX_FRAMES - 1):
        r.loop._feed_probe(LOUD, True)
    assert r.loop._probe is not None
    r.loop._feed_probe(LOUD, True)
    assert r.loop._probe is None


def test_a_probe_may_open_the_moment_the_cooldown_ends_and_not_before():
    now = [10.0]
    r, started = probe_rig(clock=lambda: now[0])
    r.loop._probe = _Turn(probe=True)
    r.loop._end_probe()  # ends one at t = 10: the next may open at 11
    assert r.loop._probe_ready_at == 10.0 + turns.PROBE_COOLDOWN_S
    now[0] = 10.0 + turns.PROBE_COOLDOWN_S - 0.001
    r.loop._open_probe()
    assert r.loop._probe is None and started == []
    now[0] = 10.0 + turns.PROBE_COOLDOWN_S
    r.loop._open_probe()
    assert r.loop._probe is not None and len(started) == 1
    assert r.loop.status()["barge_in"]["probes"] == 1


# -- idle ---------------------------------------------------------------------------------------------------------------


def idle_rig(now, **kw):
    told = []
    r = rig([], ScriptedSTT([]), idle_s=30, clock=lambda: now[0], on_idle=lambda: told.append(now[0]), **kw)
    r.loop._active_at = 100.0
    return r, told


def test_an_idle_loop_pauses_exactly_when_the_configured_time_has_passed():
    now = [129.99]
    r, told = idle_rig(now)
    r.loop._check_idle()
    assert told == [] and not r.loop._paused and r.device.calls == []
    now[0] = 130.0
    r.loop._check_idle()
    # the microphone is off before anyone is called: a callback that blocks or raises cannot leave it on
    assert told == [130.0] and r.loop._paused and r.device.calls == ["pause"]


def test_anything_in_flight_keeps_the_loop_awake_and_restarts_the_idle_count():
    now = [200.0]
    r, told = idle_rig(now, playing=True)  # Glide is speaking
    r.loop._check_idle()
    assert r.loop._active_at == 200.0
    r.device.playing = False
    now[0] = 229.9
    r.loop._check_idle()
    assert told == []  # 29.9 s since the last activity, not since the loop was made
    now[0] = 230.0
    r.loop._check_idle()
    assert told == [230.0]


def test_a_turn_still_being_transcribed_keeps_the_loop_awake():
    now = [500.0]
    r, told = idle_rig(now)
    release = threading.Event()
    turn = _Turn()
    turn.thread = threading.Thread(target=lambda: release.wait(WAIT), daemon=True)
    turn.thread.start()
    r.loop._turns = [turn]
    try:
        r.loop._check_idle()
        assert told == [] and r.loop._active_at == 500.0
    finally:
        release.set()
        turn.thread.join(WAIT)
    now[0] = 530.0
    r.loop._check_idle()
    assert told == [530.0]


def test_a_finished_turn_without_a_thread_does_not_keep_the_loop_awake():
    now = [500.0]
    r, told = idle_rig(now)
    r.loop._turns = [_Turn()]  # no worker thread was ever started for it
    r.loop._check_idle()
    assert told == [500.0]


# -- the status is numbers only --------------------------------------------------------------------------------------------


def test_the_status_rounds_the_canceller_numbers_and_names_no_underruns_a_device_does_not_report():
    r, _ = probe_rig()
    r.device.echo = EchoStats(
        erle_db=12.345, measured=True, latency_ms=3.14159, far_rms=10.4, out_rms=2.6, expected_residual_rms=7.5
    )
    status = r.loop.status()["echo"]
    assert status["erle_db"] == 12.3 and status["latency_ms"] == 3.1
    assert (status["far_rms"], status["out_rms"], status["expected_residual_rms"]) == (10, 3, 8)
    assert status["reference_underruns"] == 0


# -- the loop's own bookkeeping -------------------------------------------------------------------------------------------


class Echoing:
    """What a device with an echo canceller adds: its statistics, and the freeze the loop asks for while a voice is judged."""

    def __init__(self, r, stats):
        self.freezes = []
        r.device.echo = stats
        r.device.hold_echo_stats = self.freezes.append


def test_aborting_discards_an_open_probe_and_releases_its_transcriber():
    stt = ScriptedSTT(["stop"])
    r = rig([], stt, playing=True)
    r.loop._open_probe()
    probe = r.loop._probe
    assert probe is not None and probe.probe
    r.loop._abort()
    probe.thread.join(WAIT)
    assert r.loop._probe is None and not probe.thread.is_alive()
    assert stt.ended == []  # its audio never ended normally: a half-heard stop is not acted on


def test_when_glide_stops_speaking_the_probe_is_finished_normally_and_the_gate_and_hold_are_released():
    r, _ = probe_rig()
    echoing = Echoing(r, EchoStats(erle_db=25.0, measured=True, expected_residual_rms=100.0))
    r.loop._probe = probe = _Turn(probe=True)
    r.loop._hold(True)
    r.loop._settle_gate()
    assert r.loop._probe is None and probe.audio.queue[-1] is None  # finished, so that its transcript can decide
    assert echoing.freezes == [True, False]
    r.loop._gate.feed(1.0, 150.0, r.device.echo)
    assert r.loop._gate.armed
    r.loop._settle_gate()
    assert not r.loop._gate.armed and echoing.freezes[-1] is False


def test_each_of_the_three_things_settling_looks_for_is_enough_to_settle():
    r, _ = probe_rig()
    echoing = Echoing(r, EchoStats(erle_db=25.0, measured=True, expected_residual_rms=100.0))
    r.loop._probe = probe = _Turn(probe=True)  # only a probe is open
    r.loop._settle_gate()
    assert r.loop._probe is None and probe.audio.queue[-1] is None
    r.loop._gate.feed(1.0, 150.0, r.device.echo)  # only the gate is armed
    assert r.loop._gate.armed
    r.loop._settle_gate()
    assert not r.loop._gate.armed
    r.loop._held = True  # only the hold is on
    r.loop._settle_gate()
    assert r.loop._held is False and echoing.freezes[-1] is False


def test_a_new_probe_starts_counting_from_nothing():
    r, started = probe_rig()
    r.loop._probe_frames, r.loop._probe_quiet = 7, 5
    r.loop._open_probe()
    assert (r.loop._probe_frames, r.loop._probe_quiet) == (0, 0) and len(started) == 1


def test_starting_a_turn_forgets_the_turns_that_have_finished_and_keeps_the_ones_still_running():
    r, _ = probe_rig()
    release = threading.Event()
    r.loop._work = lambda turn: None
    done, live, never_started = _Turn(), _Turn(), _Turn()
    done.thread = threading.Thread(target=lambda: None)
    done.thread.start()
    done.thread.join()
    live.thread = threading.Thread(target=lambda: release.wait(WAIT), daemon=True)
    live.thread.start()
    r.loop._turns = [done, live, never_started]
    try:
        new = _Turn()
        del r.loop._start  # the real one, which `probe_rig` replaced with a recorder
        r.loop._start(new)
        assert r.loop._turns == [live, new]
        new.thread.join(WAIT)
    finally:
        release.set()
        live.thread.join(WAIT)


def test_a_gate_that_is_judging_nothing_is_left_alone_when_glide_is_not_speaking():
    r, _ = probe_rig()
    echoing = Echoing(r, EchoStats())
    r.loop._settle_gate()
    assert echoing.freezes == []


def test_a_frame_after_glide_stopped_speaking_releases_what_the_gate_was_holding():
    r = rig([], ScriptedSTT([]), playing=True)
    echoing = Echoing(r, EchoStats(erle_db=25.0, measured=True, expected_residual_rms=100.0))
    voiced = bytes([150, 0]) * (FRAME_BYTES // 2)  # a frame whose level is 150: over the suspect ratio of 100 of echo
    r.loop._vad = lambda frame: 1.0
    r.loop._frame(voiced)
    assert r.loop._gate.armed and echoing.freezes == [True]
    r.device.playing = False
    r.loop._frame(QUIET)
    assert not r.loop._gate.armed and echoing.freezes == [True, False]


def test_speaking_is_a_device_playing_or_an_echo_that_may_still_be_in_the_microphone():
    r = rig([], ScriptedSTT([]))
    assert r.loop._speaking() is False
    r.device.playing = True
    assert r.loop._speaking() is True
    r.device.playing = False
    r.device.echo_active = True
    assert r.loop._speaking() is True


def test_a_barge_in_cuts_the_voice_first_clears_the_gate_and_releases_the_hold_then_starts_the_turn():
    r = rig([], ScriptedSTT([]), playing=True)
    echoing = Echoing(r, EchoStats(erle_db=25.0, measured=True, expected_residual_rms=100.0))
    order = []
    r.assistant.interrupt_speech = lambda drop_pending=True: order.append(("interrupt", drop_pending))
    r.loop._begin = lambda audio: order.append(("begin", audio))
    r.loop._recent.extend([QUIET, LOUD])
    r.loop._hold(True)
    r.loop._gate.feed(1.0, 150.0, r.device.echo)
    assert r.loop._gate.armed
    r.loop._barge_in()
    assert [step[0] for step in order] == ["interrupt", "begin"] and order[0][1] is True  # no earlier turn is waiting
    assert not r.loop._gate.armed and r.loop._gate.onset_back == 0 and echoing.freezes[-1] is False
    assert r.loop._detector.active  # the frame that qualified is the first frame of the turn the detector now runs


def test_ending_a_turn_clears_the_merge_window_and_the_detector_and_closes_the_audio():
    now = [5.0]
    r = rig([], ScriptedSTT([]), clock=lambda: now[0])
    turn = r.loop._turn = _Turn()
    r.loop._merging = 5
    r.loop._detector.feed(LOUD, 1.0)
    now[0] = 9.0
    r.loop._end_turn()
    assert r.loop._turn is None and r.loop._merging == 0 and not r.loop._detector.active
    assert turn.audio.queue[-1] is None and r.loop._active_at == 9.0


def test_beginning_a_turn_notes_when_the_microphone_was_last_in_use():
    now = [5.0]
    r = rig([], ScriptedSTT([]), clock=lambda: now[0])
    r.loop._start = lambda turn: None
    now[0] = 7.5
    r.loop._begin(LOUD)
    assert r.loop._active_at == 7.5 and r.loop._turn is not None and r.loop._turn.audio.get_nowait() == LOUD


def test_aborting_clears_the_merge_window_the_detector_the_gate_and_the_hold_and_aborts_the_audio():
    r = rig([], ScriptedSTT([]), playing=True)
    echoing = Echoing(r, EchoStats(erle_db=25.0, measured=True, expected_residual_rms=100.0))
    turn = r.loop._turn = _Turn()
    r.loop._merging = 5
    r.loop._detector.feed(LOUD, 1.0)
    r.loop._hold(True)
    r.loop._gate.feed(1.0, 150.0, r.device.echo)
    r.loop._abort()
    assert r.loop._turn is None and r.loop._merging == 0 and not r.loop._detector.active
    assert not r.loop._gate.armed and echoing.freezes[-1] is False
    assert isinstance(turn.audio.queue[-1], turns.TurnAborted)


def test_pause_and_resume_commands_act_only_in_the_state_they_apply_to():
    r = rig([], ScriptedSTT([]))
    r.loop.resume()
    r.loop._apply_commands()
    assert r.device.calls == []  # not paused: nothing to resume
    r.loop.pause()
    r.loop.pause()
    r.loop._apply_commands()
    assert r.device.calls == ["pause"] and r.loop._paused  # the second pause is ignored
    r.device.loop = SimpleNamespace(stop_soon=lambda: None)  # the scripted device ends the test on a pause: not now
    r.loop.resume()
    r.loop._apply_commands()
    assert r.device.calls == ["pause", "resume"] and not r.loop._paused


def test_a_loop_that_is_not_paused_never_waits_for_a_command_and_a_paused_one_waits_for_the_first_only():
    class Commands:
        def __init__(self, items):
            self.items, self.calls = list(items), []

        def get_nowait(self):
            self.calls.append("nowait")
            if not self.items:
                raise turns.queue.Empty
            return self.items.pop(0)

        def get(self, timeout=None):
            self.calls.append(("wait", timeout))
            return self.get_nowait()

    r = rig([], ScriptedSTT([]))
    r.loop._commands = Commands([])
    r.loop._apply_commands()
    assert r.loop._commands.calls == ["nowait"]
    r.loop._paused = True
    r.loop._commands = Commands(["wake", "wake"])
    r.loop._apply_commands()
    assert r.loop._commands.calls[0] == ("wait", turns.POLL_S) and "nowait" in r.loop._commands.calls
    assert all(call == "nowait" for call in r.loop._commands.calls[2:])  # only the first read waited


def test_a_microphone_that_cannot_be_turned_on_stays_off_and_the_person_is_told():
    r = rig([], ScriptedSTT([]))
    r.loop._paused = True

    def broken():
        raise OSError("no device")

    r.device.resume_input = broken
    r.loop.resume()
    r.loop._apply_commands()
    assert r.loop._paused and r.warned == ["the microphone could not be turned on (OSError)"]


def test_turning_the_microphone_on_restarts_the_idle_count():
    now = [100.0]
    r = rig([], ScriptedSTT([]), clock=lambda: now[0])
    r.loop._paused = True
    now[0] = 250.0
    r.loop.resume()
    r.loop._apply_commands()
    assert not r.loop._paused and r.loop._active_at == 250.0


def test_start_runs_the_loop_on_a_daemon_thread_and_stop_ends_and_joins_it():
    release = threading.Event()
    r = rig([], ScriptedSTT([]))
    r.device.read = lambda timeout=None: (release.wait(0.01), None)[1]
    r.loop.start()
    thread = r.loop._thread
    try:
        assert thread.is_alive() and thread.daemon and thread.name == "glide-voice"
        r.loop.stop()
        assert not thread.is_alive()
    finally:
        r.loop._stop.set()
        release.set()


def test_stopping_from_the_loops_own_thread_does_not_try_to_join_itself():
    r = rig([], ScriptedSTT([]))
    r.loop._thread = threading.current_thread()
    r.loop.stop()  # would raise "cannot join current thread" if it tried
    assert r.loop._stop.is_set() and r.loop._commands.get_nowait() == "wake"
    r2 = rig([], ScriptedSTT([]))
    r2.loop.stop()  # never started: nothing to join
    assert r2.loop._stop.is_set()


# -- a barge-in drops pending requests only when no earlier turn is still waiting for its transcript ---------------------


def barge_in_decision(r):
    """What `_barge_in` asks `interrupt_speech` for, with the turn that follows stubbed out."""
    asked = []
    r.assistant.interrupt_speech = lambda drop_pending=True: asked.append(drop_pending)
    r.loop._begin = lambda audio: None
    r.loop._recent.append(QUIET)
    r.loop._barge_in()
    return asked


def run_turn_a(r, stt_final: bool):
    """Turn A: its audio is complete; its worker thread is running (it holds in its answer when the transcript arrived)."""
    turn = r.loop._turn = _Turn()
    turn.audio.put(LOUD)
    turn.audio.put(None)
    r.loop._turn = None
    r.loop._start(turn)
    return turn


def test_a_transcript_that_has_arrived_frees_a_barge_in_to_drop_pending_requests_even_while_its_turn_still_answers():
    gate = threading.Event()
    r = rig([], ScriptedSTT(["turn a"]), playing=True)
    r.assistant._config.fast = FakeLLM(route=route_json("answer"), deltas=["Hi."], gates={0: gate})
    turn = run_turn_a(r, True)
    try:
        assert turn.heard.wait(WAIT), "the transcript arrived but its turn was not marked as heard"
        assert turn.thread.is_alive()  # the worker is still in its answer
        assert barge_in_decision(r) == [True]
    finally:
        gate.set()
        turn.thread.join(WAIT)


def test_a_turn_still_waiting_for_its_transcript_keeps_a_barge_in_from_dropping_pending_requests():
    release = threading.Event()

    class Slow(ScriptedSTT):
        def stream(self, chunks, **kw):
            list(chunks)
            release.wait(WAIT)
            return iter(())

    r = rig([], Slow([]), playing=True)
    turn = run_turn_a(r, False)
    try:
        assert not turn.heard.is_set()
        assert barge_in_decision(r) == [False]
    finally:
        release.set()
        turn.thread.join(WAIT)


def test_a_turn_whose_transcription_failed_is_marked_heard_when_its_worker_ends():
    class Fails(ScriptedSTT):
        def stream(self, chunks, **kw):
            list(chunks)
            raise ProviderError("no socket", kind="transport")
            yield  # pragma: no cover

    r = rig([], Fails([]), playing=True)
    turn = run_turn_a(r, False)
    turn.thread.join(WAIT)
    assert not turn.thread.is_alive() and turn.heard.is_set()
    assert barge_in_decision(r) == [True]

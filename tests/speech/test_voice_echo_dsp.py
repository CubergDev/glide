"""Echo cancellation and barge-in against simulated rooms (needs numpy; the WebRTC cases also need livekit, the `aec` extra).

Run them with `uv run --python 3.13 --extra speech --extra aec pytest tests/speech/test_voice_echo_dsp.py`. Without the
extras they SKIP, as the Silero tests do: the default suite does not prove any of this, and a green default run does not
say it was proved.

The room (room.py) is synthetic: speech-like signals, an echo path made of a delay, a decaying random response and a
little loudspeaker colouring, white room noise. It shows that the signal processing and the policy do what they claim
on those signals, for the delays, levels and noise swept here. It does not show how a real microphone, speaker, room or
voice behaves: docs/live-checks/voice.md says what to check there.

Stated claims, with the measured values they were set from (the sweeps are deterministic in their seeds):

- ERLE after convergence on echo only: WebRTC >= 30 dB (measured about 40), nlms >= 12 dB (measured 15 to 28);
- no false barge-in and no false stop probe on echo only, over seeds, delays, echo levels and noise levels;
- a voice 9 dB over the echo at the microphone is detected, by both, within 0.6 s of the voice starting, the voice is
  cut within one frame of the decision, and the first 300 ms of the voice are in the turn;
- double talk does not diverge the filter: it recovers its ERLE afterwards, and the numpy filter keeps the person's voice
  (correlation above 0.95);
- the limits, said where they bite: a voice quieter than the echo is detected less often and later (the numpy filter
  needs it to be within 3 dB of the echo, WebRTC ducks a voice in double talk and needs it a little above the echo).
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from guards_voice import no_real_audio  # noqa: F401

np = pytest.importorskip("numpy")

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from room import FS, EchoPath, SimRoom, erle_db, pcm, samples, speechlike  # noqa: E402
from test_assistant_fakes import WAIT, FakeConfig, FakeLLM, FakeTTS  # noqa: E402
from test_voice_bargein import StopSTT  # noqa: E402

from glide.assistant.audio_io import rms  # noqa: E402
from glide.assistant.core import IO, Assistant  # noqa: E402
from glide.speech.echo import FrequencyDomainCanceller, WebRtcCanceller  # noqa: E402
from glide.speech.turns import VoiceLoop  # noqa: E402
from glide.speech.vad import FRAME_BYTES, FRAME_S  # noqa: E402

FRAME = FRAME_BYTES // 2
SPEAKER = 3000.0  # the level of Glide's voice as handed to the speaker (RMS, 16-bit scale)
ONSET_LEVEL = 0.4  # the person's voice counts as started when a frame reaches this share of its level


def build(kind):
    if kind == "webrtc":
        pytest.importorskip("livekit.rtc")
        return WebRtcCanceller()
    return FrequencyDomainCanceller()


def pessimistic_vad(frame: bytes) -> float:
    """Every frame that is not nearly silent counts as speech: the cleaned echo is judged by the gate alone."""
    return 1.0 if rms(frame) >= 40 else 0.0


DELAYS = {"webrtc": (30, 90, 170), "nlms": (30, 90, 150)}  # nlms covers 384 ms of delay and tail together
ERLE_MIN = {"webrtc": 30.0, "nlms": 12.0}
CANCELLERS = ["webrtc", "nlms"]


def echo_only(kind, delay, *, gain=0.7, seed=1, seconds=14.0, noise=20.0, soft=0.0):
    far = speechlike(seconds, seed, level=SPEAKER, pause=0.1)
    canceller = build(kind)
    room = SimRoom(canceller, EchoPath(delay, seed, gain=gain, soft_clip=soft), noise=noise, seed=seed)
    room.play(pcm(far), FS)
    while room.frame_no < int(seconds * FS / FRAME):
        room.read()
    return room, canceller


@pytest.mark.parametrize("kind", CANCELLERS)
@pytest.mark.parametrize("delay", [30, 90, 150])
def test_erle_after_convergence_is_above_the_stated_threshold(kind, delay):
    room, canceller = echo_only(kind, delay)
    try:
        a, b = int(6 * FS / FRAME), len(room.cleaned)
        heard = np.concatenate(room.heard[a:b])
        cleaned = np.concatenate([samples(f) for f in room.cleaned[a:b]])
        assert erle_db(heard, cleaned) >= ERLE_MIN[kind]
        assert canceller.stats.measured and canceller.stats.erle_db >= ERLE_MIN[kind] - 8  # the running figure agrees roughly
    finally:
        canceller.close()


def test_webrtc_handles_the_longest_delay_swept():
    room, canceller = echo_only("webrtc", 250)
    try:
        a = int(6 * FS / FRAME)
        assert (
            erle_db(np.concatenate(room.heard[a:]), np.concatenate([samples(f) for f in room.cleaned[a:]])) >= ERLE_MIN["webrtc"]
        )
    finally:
        canceller.close()


@pytest.mark.parametrize("kind", CANCELLERS)
@pytest.mark.parametrize("delay", [30, 90, 150])
def test_the_delay_estimate_is_the_echo_path_plus_the_frame_the_pairing_adds_to_a_frame(kind, delay):
    _, canceller = echo_only(kind, delay, seconds=10.0)
    try:
        assert canceller.stats.delay_ms == pytest.approx(delay + FRAME_S * 1000, abs=FRAME_S * 1000 + 8)
    finally:
        canceller.close()


def test_the_webrtc_latency_is_what_it_says():
    """The cleaned stream lags the microphone by `latency_samples`; with a silent reference it is the microphone, delayed."""
    canceller = build("webrtc")
    try:
        signal = speechlike(4.0, 7, level=2500, pause=0.0)
        out = np.concatenate(
            [
                samples(canceller.process(pcm(signal[i : i + FRAME]), bytes(FRAME_BYTES)))
                for i in range(0, len(signal) - FRAME, FRAME)
            ]
        )
        lags = range(0, 2 * FRAME)
        best = max(lags, key=lambda lag: np.dot(signal[: len(out) - lag], out[lag:]))
        assert best == canceller.latency_samples
        assert np.corrcoef(signal[: len(out) - best], out[best:])[0, 1] > 0.9
        assert canceller.stats.latency_ms == pytest.approx(best / FS * 1000)
    finally:
        canceller.close()


def test_the_numpy_filter_adds_no_latency():
    canceller = build("nlms")
    signal = speechlike(2.0, 7, level=2500, pause=0.0)
    out = np.concatenate(
        [samples(canceller.process(pcm(signal[i : i + FRAME]), bytes(FRAME_BYTES))) for i in range(0, len(signal) - FRAME, FRAME)]
    )
    assert np.allclose(out, np.rint(signal[: len(out)]), atol=1) and canceller.latency_samples == 0


# -- double talk ------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", CANCELLERS)
def test_double_talk_does_not_diverge_the_filter(kind):
    delay, seconds = 90, 20.0
    far = speechlike(seconds, 3, level=SPEAKER, pause=0.1)
    user = np.zeros(int(seconds * FS))
    a, b = 8 * FS, 12 * FS
    user[a:b] = speechlike(4.0, 99, level=2500, pause=0.0)
    canceller = build(kind)
    room = SimRoom(canceller, EchoPath(delay, 3, gain=0.7), user=user, noise=20.0, seed=3)
    room.play(pcm(far), FS)
    peak_before = 0.0
    try:
        while room.frame_no < int(seconds * FS / FRAME):
            room.read()
            if kind == "nlms" and room.frame_no == int(8 * FS / FRAME):
                peak_before = float(np.abs(canceller._foreground).max())

        def window(start, stop):
            lo, hi = int(start * FS / FRAME), int(stop * FS / FRAME)
            return np.concatenate(room.heard[lo:hi]), np.concatenate([samples(f) for f in room.cleaned[lo:hi]])

        before = erle_db(*window(5, 8))
        after = erle_db(*window(15, 20))
        assert after >= before - (6.0 if kind == "nlms" else 10.0) and after >= ERLE_MIN[kind]  # it recovers; it did not diverge
        _, cleaned = window(8.5, 11.5)
        lag = canceller.latency_samples
        first = int(8.5 * FS / FRAME) * FRAME
        want = user[first : first + len(cleaned)]
        corr = np.corrcoef(want[: len(want) - lag], cleaned[lag:])[0, 1]
        if kind == "nlms":
            assert corr > 0.95  # the person's voice passes through the numpy filter intact
            assert (
                float(np.abs(canceller._foreground).max()) <= 3 * peak_before
            )  # and the filter's weights stayed where they were
        else:
            assert corr > 0.2 and np.sqrt(np.mean(cleaned**2)) > 0.2 * np.sqrt(
                np.mean(want**2)
            )  # AEC3 ducks a voice in double talk
    finally:
        canceller.close()


# -- the loop over a simulated room -----------------------------------------------------------------


def scenario(kind, delay, gain, seed, *, user_level=None, user_at=7.0, noise=20.0, soft=0.0, seconds=12.0, stt=None):
    far = speechlike(seconds, seed, level=SPEAKER, pause=0.1)
    user = None
    if user_level:
        user = np.zeros(int(seconds * FS))
        segment = speechlike(3.0, seed + 1000, level=user_level, pause=0.0)
        user[int(user_at * FS) : int(user_at * FS) + len(segment)] = segment
    canceller = build(kind)
    room = SimRoom(canceller, EchoPath(delay, seed, gain=gain, soft_clip=soft), user=user, noise=noise, seed=seed)
    room.play(pcm(far), FS)
    stt = stt or StopSTT(["tell me the time"] * 3)
    io = IO(player=room, heard=lambda text: None, warn=lambda text: None)
    assistant = Assistant(FakeConfig(llm=FakeLLM(), stt=stt, tts=FakeTTS()), io=io)
    loop = VoiceLoop(assistant, room, pessimistic_vad, clock=lambda: room.now)
    room.loop_stop = loop._stop.set
    room.stop_after = int(seconds * FS / FRAME)
    decisions = []
    original = loop._barge_in
    loop._barge_in = lambda: (decisions.append(room.frame_no - 1), original())[1]
    loop.run()
    loop.join_turns(WAIT)
    status = loop.status()["barge_in"]
    assistant.close()
    canceller.close()
    return SimpleNamespace(
        room=room, stt=stt, decisions=decisions, status=status, user=user, user_level=user_level, seconds=seconds
    )


GRID = [(delay, gain, noise) for delay in (30, 90, 150) for gain in (0.3, 1.2) for noise in (10, 60)]


@pytest.mark.parametrize("kind", CANCELLERS)
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_glides_own_echo_never_interrupts_it_and_never_opens_a_probe(kind, seed):
    """Echo only, every delay, echo level and noise level of the grid: zero decisions, zero probes, by the gate alone
    (the voice detector here calls everything that is not silent speech)."""
    for delay, gain, noise in GRID:
        if kind == "webrtc" and delay == 150:
            delay = 170
        run = scenario(kind, delay, gain, seed, noise=noise)
        assert run.decisions == [], (kind, seed, delay, gain, noise)
        assert run.status["probes"] == 0 and run.status["confirmed"] == 0, (kind, seed, delay, gain, noise, run.status)
        assert run.room.cancels == [run.room.frame_no], "nothing cut the voice but the end of the run"


@pytest.mark.parametrize("kind", CANCELLERS)
def test_on_seeds_the_thresholds_were_not_tuned_on_echo_still_never_interrupts_glide(kind):
    """Held out: the margin, percentiles and run lengths were set looking at seeds 1 to 8. Seeds 9 to 16 were run once,
    afterwards. Zero interruptions in 96 runs per canceller; probes (which interrupt nothing) measured 1 for WebRTC and
    0 for the numpy filter, so the bound is two."""
    probes = 0
    for seed in range(9, 17):
        for delay, gain, noise in GRID:
            run = scenario(kind, 170 if kind == "webrtc" and delay == 150 else delay, gain, seed, noise=noise)
            assert run.decisions == [], (kind, seed, delay, gain, noise)
            probes += run.status["probes"]
    assert probes <= 2


@pytest.mark.parametrize("kind", CANCELLERS)
def test_a_loudspeaker_that_distorts_still_never_interrupts_glide(kind):
    """A soft-clipping speaker is an echo no linear canceller can remove. Nothing may interrupt (a stop probe is allowed
    at most once over the grid: measured 1 in 288 runs for WebRTC, 0 for the numpy filter)."""
    probes = 0
    for seed in (1, 2, 3):
        for delay, gain, noise in GRID[::2]:
            run = scenario(kind, 170 if kind == "webrtc" and delay == 150 else delay, gain, seed, noise=noise, soft=3.0)
            assert run.decisions == [], (kind, seed, delay, gain, noise)
            probes += run.status["probes"]
    assert probes <= 1


def voice_levels(run):
    return [rms(pcm(run.user[i * FRAME : (i + 1) * FRAME])) for i in range(len(run.user) // FRAME)]


def onset_frame(run) -> int:
    """The frame at which the voice has clearly started: a frame at 40 % of its level."""
    return next(i for i, value in enumerate(voice_levels(run)) if value >= ONSET_LEVEL * run.user_level)


def start_frame(run) -> int:
    """The first frame with any of the voice in it (5 % of its level): nothing may be decided before this."""
    return next(i for i, value in enumerate(voice_levels(run)) if value >= 0.05 * run.user_level)


CASES = [  # (kind, user level, echo gain, delays, latency budget in s): the voice is `SIR` dB over the echo at the microphone
    ("nlms", 2500, 0.3, (30, 90), 0.6),  # +9 dB
    ("webrtc", 2500, 0.3, (30, 90), 0.6),  # +9 dB
    ("nlms", 2500, 0.7, (30, 90), 0.6),  # +1.5 dB
    ("nlms", 2500, 1.2, (30, 90), 0.8),  # -3 dB
]


@pytest.mark.parametrize("kind,level,gain,delays,budget", CASES)
@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_a_voice_over_glide_is_detected_in_time_cut_at_once_and_its_first_300_ms_are_kept(
    kind, level, gain, delays, budget, seed
):
    for delay in delays:
        run = scenario(kind, delay, gain, seed, user_level=level)
        label = (kind, level, gain, delay, seed)
        onset = onset_frame(run)
        assert run.decisions, f"not detected {label}"
        decided = run.decisions[0]
        assert decided >= start_frame(run), f"decided before the voice began {label}"
        assert (decided - onset) * FRAME_S <= budget, f"too slow {label}: {(decided - onset) * FRAME_S:.2f} s"
        # the voice is cut within one frame of the decision: the speaker was sounding, and the next block is silence
        played = run.room.played
        assert any(played[decided]) and not any(played[decided + 1]), label
        # the turn that interrupted holds the first 300 ms of the voice: the cleaned frames from onset to +300 ms, whole
        first_300_ms = b"".join(run.room.cleaned[onset : onset + 10])
        turn = max(run.stt.audio)  # a probe that opened on the way, and was replaced, came first
        assert turn not in run.stt.aborted and first_300_ms in bytes(run.stt.audio[turn]), f"first words lost {label}"
        assert run.status["confirmed"] == 1


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_a_webrtc_voice_only_a_little_over_the_echo_is_detected_when_it_is_not_ducked_and_never_falsely(seed):
    """The limit, said as a test: AEC3 ducks a voice in double talk, so at +1.5 dB some are late or missed. What must
    hold is that none is detected before it starts."""
    run = scenario("webrtc", 90, 0.7, seed, user_level=2500)
    if run.decisions:
        assert run.decisions[0] >= start_frame(run)


def test_a_quiet_stop_over_glide_is_cut_more_often_through_the_probe_than_a_quiet_voice_alone():
    """The voice is 6 dB under the echo at the microphone (numpy filter): too weak to confirm by itself in most cases.
    When the transcriber hears 'stop' the weaker evidence is enough. Measured over these 8 seeds: 4 cut with a stop,
    2 without."""
    with_stop = without = 0
    for seed in range(1, 9):
        quiet_stop = scenario("nlms", 90, 0.7, seed, user_level=1000, stt=StopSTT(["stop"] * 3))
        quiet_voice = scenario("nlms", 90, 0.7, seed, user_level=1000)
        for run, name in ((quiet_stop, "stop"), (quiet_voice, "voice")):
            cuts = [frame for frame in run.room.cancels if frame < run.room.frame_no]
            if cuts:
                assert cuts[0] >= start_frame(run)  # never before the person began
                with_stop += name == "stop"
                without += name == "voice"
    assert with_stop >= 3 and with_stop > without

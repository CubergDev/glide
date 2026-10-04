"""Frozen decision traces: the same seeded inputs must give the same decisions, frame for frame, however the code is arranged.

These are characterisation tests. The inputs are random but seeded, and built to straddle every threshold the policy
has (probabilities at 0.35 and 0.5, levels a hair either side of the margins, the suspect ratio and the retained
share, ERLE either side of its floor), so a change of a comparison, an off-by-one in a run or a counter that moves is
a different trace. Each trace is reduced to a SHA-256: when one fails, `trace_*` below returns the whole list, to diff
against the version that passed. A deliberate change to the policy changes the hashes here and says so in its commit.
"""

from __future__ import annotations

import hashlib
import queue
import random
from array import array
from types import SimpleNamespace
from typing import ClassVar

import pytest
from guards_voice import no_real_audio  # noqa: F401

import glide.speech.turns as turns
from glide.assistant.audio_io import Endpointer, rms
from glide.speech.echo import EchoCanceller, EchoStats
from glide.speech.turns import BargeInGate, TurnAborted, VoiceLoop
from glide.speech.vad import FRAME_BYTES, EnergyProbability


def digest(trace) -> str:
    return hashlib.sha256(repr(trace).encode()).hexdigest()[:16]


def tone(amplitude: float) -> bytes:
    return array("h", [int(amplitude)]).tobytes() * (FRAME_BYTES // 2)


# -- the gate, frame by frame -----------------------------------------------------------------------


def gate_trace(seed: int, frames: int = 1600) -> list:
    rng = random.Random(seed)
    gate = BargeInGate(
        min_voiced_ms=rng.choice([96, 190, 320]), margin_db=rng.choice([3.0, 8.0, 12.0]), min_erle_db=rng.choice([0.0, 6.0, 12.0])
    )
    trace: list = [gate.thresholds]
    while len(trace) < frames:
        probability = rng.choice([0.0, 0.34, 0.35, 0.49, 0.5, 0.51, 1.0])
        mode = rng.choice(["none", "unmeasured", "weak_erle", "ok", "ok", "ok"])
        expected = rng.choice([0.0, 20.0, 60.0, 200.0, 800.0])
        floor = max(expected, 60.0)
        db = rng.choice([gate.margin_db, gate.probe_margin_db, 0.0])
        level = rng.choice(
            [0.0, 0.5 * floor, 1.39 * floor, 1.41 * floor, 3000.0, floor * 10 ** (db / 20) * rng.choice([0.97, 1.0, 1.03])]
        )
        near = rng.choice([0.0, level, level * 4 * rng.choice([0.99, 1.0, 1.01]), 5000.0])
        erle = gate.min_erle_db + rng.choice([-0.1, 0.0, 0.1])
        echo = None
        if mode != "none":
            echo = EchoStats(
                near_rms=near,
                expected_residual_rms=expected,
                erle_db=None if mode == "unmeasured" else (gate.min_erle_db - 1 if mode == "weak_erle" else erle),
                measured=mode != "unmeasured",
            )
        for _ in range(rng.randint(1, 18)):
            p = 0.0 if rng.random() < 0.1 else probability  # speech has dips
            if rng.random() < 0.01:
                gate.reset()
                trace.append("reset")
            v = gate.feed(p, level, echo)
            trace.append((v.confirm, v.probe, v.weak, v.strong, gate.armed, gate.onset_back))
    trace.append(dict(gate.counters))
    return trace


GATE_GOLDEN = {
    0: "30d6a122329ef979",
    1: "acbd17762415418e",
    2: "784d9b008c0f9f01",
    3: "41e3705037245a91",
    4: "7e878e715759cb2d",
    5: "c06212f99c4d4bcc",
    6: "5e84384d6a959f4b",
    7: "2af2a466776c2f55",
    8: "ea3300c4f52a1071",
    9: "ee6179d47122f4f1",
    10: "917351d32d0ff0dc",
    11: "e74be6fd583770de",
}


@pytest.mark.parametrize("seed", sorted(GATE_GOLDEN))
def test_the_gate_decides_every_frame_of_a_seeded_session_as_it_did(seed):
    assert digest(gate_trace(seed)) == GATE_GOLDEN[seed]


# -- the canceller's statistics ---------------------------------------------------------------------


class Scaler(EchoCanceller):
    name = "scaler"
    latency_samples = 64

    def __init__(self):
        super().__init__()
        self.factor = 1.0

    def _process(self, near, far):
        return array("h", [int(s * self.factor) for s in array("h", near)]).tobytes()

    def _reset_state(self): ...


def stats_trace(seed: int, frames: int = 700) -> list:
    rng = random.Random(seed)
    canceller = Scaler()
    far_history: list[float] = [0.0] * 4
    trace: list = []
    while len(trace) < frames:
        far = rng.choice([0.0, 30.0, 39.0, 40.0, 41.0, 100.0, 800.0])
        base = rng.choice([0.0, 50.0, 400.0])
        canceller.factor = rng.choice([1.0, 0.5, 0.1, 0.9])
        for _ in range(rng.randint(1, 40)):
            far_history.append(far)
            near_amp = base + 0.6 * far_history[-4]  # the echo follows the reference by three frames
            if rng.random() < 0.02:
                canceller.hold = not canceller.hold
            if rng.random() < 0.004:
                canceller.reset()
            canceller.process(tone(near_amp), tone(far))
            trace.append((repr(canceller.stats), canceller.hold))
    return trace


STATS_GOLDEN = {
    0: "ff1285e2c451fd74",
    1: "26ebcb99361f5184",
    2: "d328583aaa67ce4c",
    3: "41dc2734f2ff1cd7",
}


@pytest.mark.parametrize("seed", sorted(STATS_GOLDEN))
def test_the_cancellers_statistics_come_out_of_a_seeded_session_as_they_did(seed):
    assert digest(stats_trace(seed)) == STATS_GOLDEN[seed]


# -- the loop ---------------------------------------------------------------------------------------


class _NoThread:
    """Stands in for threading.Thread in the loop: the turn's audio queue is read at the end instead of by a worker."""

    created: ClassVar[list] = []

    def __init__(self, target=None, args=(), name=None, daemon=None):
        self.args = args

    def start(self):
        _NoThread.created.append(self)

    def is_alive(self):
        return False

    def join(self, timeout=None):
        return None


class _Assistant:
    busy = False

    def __init__(self, log):
        self.io = SimpleNamespace(heard=lambda text: None, warn=lambda message: log.append(("warn", message)))
        self._log = log

    def interrupt_speech(self, *, drop_pending=True):
        self._log.append(("interrupt", drop_pending))

    def handle_audio(self, *a, **k):
        raise AssertionError("turn threads are not run in a trace")


class _Device:
    def __init__(self, schedule, log):
        self.schedule = schedule  # one dict per frame
        self.log = log
        self.index = -1
        self.now = 1000.0
        self.loop = None
        self.hook = None

    def _at(self):
        return self.schedule[min(max(self.index, 0), len(self.schedule) - 1)]

    @property
    def playing(self):
        return self._at()["playing"]

    @property
    def echo_active(self):
        return self._at()["active"]

    @property
    def echo(self):
        return self._at()["echo"]

    def read(self, timeout=None):
        self.index += 1
        self.now += 0.032
        if self.hook:
            self.hook(self.index)
        if self.index >= len(self.schedule):
            self.loop._stop.set()
            return None
        return self.schedule[self.index]["frame"]

    def hold_echo_stats(self, hold):
        self.log.append(("hold", hold))

    def pause_input(self):
        self.log.append(("pause_input", self.index))
        self.loop.resume()  # a paused loop waits for a command: the next one is always a resume, so the trace goes on

    def resume_input(self):
        self.log.append(("resume_input", self.index))


def loop_trace(seed: int, monkeypatch, frames: int = 900) -> list:
    rng = random.Random(seed)
    log: list = []
    schedule: list = []
    while len(schedule) < frames:
        amplitude = rng.choice([0, 0, 100, 300, 420, 700, 1500])
        playing = rng.random() < 0.6
        active = playing or rng.random() < 0.3
        expected = rng.choice([50.0, 200.0])
        echo = (
            None
            if seed % 3 == 0
            else EchoStats(
                near_rms=amplitude * rng.choice([1.0, 3.0]),
                expected_residual_rms=expected,
                erle_db=rng.choice([3.0, 25.0]),
                measured=rng.random() < 0.9,
            )
        )
        for _ in range(rng.randint(1, 30)):
            schedule.append({"frame": tone(amplitude), "playing": playing, "active": active, "echo": echo})
    device = _Device(schedule, log)
    assistant = _Assistant(log)
    monkeypatch.setattr(turns.threading, "Thread", _NoThread)
    _NoThread.created = []
    loop = VoiceLoop(
        assistant,
        device,
        lambda frame: 1.0 if rms(frame) >= 30 else 0.0,
        silence_ms=rng.choice([200, 600]),
        merge_window_s=rng.choice([0.0, 0.0, 1.0]),
        idle_s=rng.choice([0.0, 5.0]),
        clock=lambda: device.now,
        barge_min_voiced_ms=rng.choice([96, 190]),
    )
    device.loop = loop
    commands = {rng.randrange(frames): rng.choice(["pause", "resume"]) for _ in range(3)}

    def hook(index):
        command = commands.get(index)
        if command:
            getattr(loop, command)()

    device.hook = hook
    loop.run()
    out: list = [log]
    for thread in _NoThread.created:
        (turn,) = thread.args
        events = []
        while True:
            try:
                item = turn.audio.get_nowait()
            except queue.Empty:
                break
            events.append(
                "end" if item is None else "abort" if isinstance(item, TurnAborted) else hashlib.sha256(item).hexdigest()[:8]
            )
        out.append((turn.probe, events))
    out.append(loop.status()["barge_in"])
    return out


LOOP_GOLDEN = {
    0: "c5a9cf095791748b",
    1: "a1090ec8e546e1b3",
    2: "97bd8c77e82a6c40",
    3: "aed6a35577f1dadd",
    4: "acb91ca7bb5dabe1",
    5: "86361a7139ad46af",
    6: "930706fc16c5d395",
    7: "677114d7e8edab0f",
    8: "9ec1253afb751241",
    9: "dac6ecf73d99ede1",
}


@pytest.mark.parametrize("seed", sorted(LOOP_GOLDEN))
def test_the_loop_makes_the_same_calls_and_turns_from_a_seeded_session(seed, monkeypatch):
    assert digest(loop_trace(seed, monkeypatch)) == LOOP_GOLDEN[seed]


# -- the two loudness-against-a-floor detectors -----------------------------------------------------


def floor_trace(seed: int, frames: int = 400) -> list:
    """`Endpointer` (push to talk) and `EnergyProbability` (hands free) keep the same running noise floor."""
    rng = random.Random(seed)
    endpointer = Endpointer(threshold=500, ratio=3.0, silence_s=0.4, no_speech_s=2.0, max_s=9.0)
    energy = EnergyProbability(threshold=500)
    trace: list = []
    level = 0.0
    for _ in range(frames):
        if rng.random() < 0.15:
            level = rng.choice([0.0, 80.0, 200.0, 499.0, 501.0, 900.0, 1499.0, 1501.0, 4000.0])
        chunk = tone(level * rng.choice([0.98, 1.0, 1.02]))
        done = endpointer.feed(chunk)
        trace.append(
            (done, endpointer.heard, endpointer.reason, round(endpointer.floor, 6), energy(chunk), round(energy.floor, 6))
        )
        if done:
            endpointer = Endpointer(threshold=500, ratio=3.0, silence_s=0.4, no_speech_s=2.0, max_s=9.0)
            energy.reset()
    return trace


FLOOR_GOLDEN = {
    0: "d901aacf65f696b7",
    1: "f79d64a380d993e9",
    2: "e0695405d47fa001",
}


@pytest.mark.parametrize("seed", sorted(FLOOR_GOLDEN))
def test_both_loudness_detectors_follow_the_room_as_they_did(seed):
    assert digest(floor_trace(seed)) == FLOOR_GOLDEN[seed]

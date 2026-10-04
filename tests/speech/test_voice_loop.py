"""The hands-free loop over a scripted device and the real Assistant (with fakes behind it): turns, merging, barge-in, idle.

The recorded failures this file pins, all offline:
- a proactive commit on silence made a double commit 154 ms apart (`commit_throttled`): one end per utterance;
- dropped speech segments: several turns, none dropped, even when the next starts before the last transcript;
- internal counters and effect counts never reach the TTS.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from guards_voice import no_real_audio  # noqa: F401

sys.path.insert(0, str(Path(__file__).parent.parent))  # the assistant tests' fakes live one directory up

from test_assistant_fakes import WAIT, FakeConfig, FakeLLM, FakePlayer, FakeTTS, route_json, wait_until

from glide.assistant.core import IO, Assistant
from glide.assistant.tasks import TaskResult
from glide.providers.base import Transcript
from glide.providers.errors import ProviderError
from glide.speech.audio import DeviceFault
from glide.speech.turns import VoiceLoop
from glide.speech.vad import FRAME_BYTES

LOUD = b"\x01\x00" * (FRAME_BYTES // 2)
QUIET = bytes(FRAME_BYTES)


def speech(n: int = 10) -> list[bytes]:
    return [LOUD] * n


def quiet(ms: int) -> list[bytes]:
    return [QUIET] * -(-ms // 32)


def vad(frame: bytes) -> float:
    return float(any(frame))


class ScriptedDevice:
    """Hands out `script` one frame per read (None is a read that timed out), then stops the loop."""

    echo = None  # no canceller: what a headset, or speaker mode without one, reports
    echo_active = False
    echo_name = None
    reference_underruns = 0

    def __init__(self, script, *, playing=False):
        self.script = list(script)
        self.playing = playing
        self.paused = False
        self.calls = []
        self.loop = None
        self.on_frame = None

    def read(self, timeout=None):
        if self.on_frame:
            self.on_frame(len(self.script))
        if not self.script:
            self.loop.stop_soon()
            return None
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    def hold_echo_stats(self, hold): ...

    def pause_input(self):
        self.paused = True
        self.calls.append("pause")
        self.loop.stop_soon()  # a paused loop only waits for a command: the test is over

    def resume_input(self):
        self.paused = False
        self.calls.append("resume")


class ScriptedSTT:
    """One transcript per stream() call, in call order. `gates[i]` holds the i-th final back until it is set."""

    def __init__(self, finals, gates=None):
        self.finals = list(finals)
        self.gates = gates or {}
        self.streams = []
        self.audio = {}  # stream index -> the audio it was given so far
        self.ended = []  # the streams whose audio ended normally, which is the one commit; an aborted one is not here
        self._lock = threading.Lock()

    def stream(self, chunks, *, sample_rate=16000, language=None):
        with self._lock:
            index = len(self.streams)
            self.streams.append(index)
        heard = self.audio[index] = bytearray()
        for chunk in chunks:
            heard.extend(chunk)
        self.ended.append(index)
        gate = self.gates.get(index)
        if gate is not None:
            gate.wait(WAIT)
        yield Transcript(self.finals[index], "en", "fake:stt", "fake", 0.0)

    def transcribe(self, audio, **kw):
        raise ProviderError("no batch in this test", kind="unsupported")


def rig(script, stt, *, playing=False, tts=None, llm=None, **loop_kw):
    llm = llm or FakeLLM()
    tts = tts or FakeTTS()
    player = FakePlayer()
    heard, warned = [], []
    io = IO(player=player, heard=heard.append, warn=warned.append)
    assistant = Assistant(FakeConfig(llm=llm, stt=stt, tts=tts), io=io)
    device = ScriptedDevice(script, playing=playing)
    loop = VoiceLoop(assistant, device, vad, **loop_kw)
    device.loop = loop
    loop.stop_soon = loop._stop.set  # the script is over: end `run()` from inside it
    return SimpleNamespace(
        loop=loop, device=device, assistant=assistant, heard=heard, warned=warned, stt=stt, tts=tts, player=player
    )


def run(r):
    r.loop.run()
    r.loop.join_turns(WAIT)
    return r


# -- turns ------------------------------------------------------------------------------------------


def test_one_utterance_is_one_turn_and_one_transcript_request():
    stt = ScriptedSTT(["what time is it"])
    r = run(rig([*quiet(300), *speech(), *quiet(700), *quiet(700)], stt))
    assert r.heard == ["what time is it"] and len(stt.streams) == 1
    assert not r.warned


def test_the_turn_audio_has_the_pre_roll_the_speech_and_the_closing_silence_once_each():
    stt = ScriptedSTT(["hi"])
    run(rig([*quiet(320), *speech(10), *quiet(700)], stt))
    frames = len(stt.audio[0]) // FRAME_BYTES
    assert frames == 8 + 10 - 1 + 19  # pre-roll (the first speech frame is its last), the rest of the speech, 608 ms of quiet


def test_a_pause_inside_an_utterance_shorter_than_the_threshold_does_not_split_it():
    stt = ScriptedSTT(["open the page and click the first link"])
    run(rig([*speech(), *quiet(500), *speech(), *quiet(700)], stt))
    assert len(stt.streams) == 1


def test_a_silence_boundary_and_a_pause_command_154_ms_apart_end_the_turn_once():
    """The recorded double commit: the end is decided in one place, so a second end 154 ms later does not exist."""
    stt = ScriptedSTT(["hello", "never"])
    r = rig([*speech(), *quiet(608), *quiet(154)], stt)
    r.device.on_frame = lambda left: r.loop.pause() if left == 5 else None  # 5 frames, 154 ms, after the silence ended it
    run(r)
    assert r.heard == ["hello"] and len(stt.streams) == 1
    assert r.device.calls == ["pause"]


def test_nothing_ends_a_turn_on_a_short_silence_before_the_configured_one():
    stt = ScriptedSTT(["one"])
    r = rig([*speech(), *quiet(576)], stt)  # 18 frames: 576 ms, one short of the 600 ms threshold
    ended_early = []
    r.device.on_frame = lambda left: ended_early.append(left) if left == 1 and r.loop._turn is None else None
    run(r)  # the turn was still open when the script ran out; stopping aborts it, silence never ended it
    assert not ended_early
    assert len(stt.streams) == 1 and stt.ended == [] and len(stt.audio[0]) // FRAME_BYTES == 10 + 18


# -- no dropped segments ----------------------------------------------------------------------------


def test_several_turns_are_all_heard_none_dropped():
    stt = ScriptedSTT(["first", "second", "third"])
    r = run(rig([*speech(), *quiet(700), *speech(), *quiet(700), *speech(), *quiet(700)], stt))
    assert sorted(r.heard) == ["first", "second", "third"] and len(stt.streams) == 3


def test_a_barge_in_while_an_earlier_turn_awaits_its_transcript_does_not_drop_it():
    """Turn B begins before turn A's final arrives, with Glide audibly speaking: both transcripts must get through."""
    gate = threading.Event()
    stt = ScriptedSTT(["turn a", "turn b"], gates={0: gate})
    r = rig([*speech(), *quiet(700), *speech(), *quiet(700)], stt, playing=True)
    r.device.on_frame = lambda left: gate.set() if left == 0 else None  # A's final is released after B has begun and ended
    run(r)
    assert sorted(r.heard) == ["turn a", "turn b"]


def test_speech_over_glide_cuts_the_voice_at_once_when_nothing_else_is_waiting():
    stt = ScriptedSTT(["stop talking"])
    r = rig([*speech(), *quiet(700)], stt, playing=True)
    epochs = []
    r.device.on_frame = lambda left: epochs.append(r.assistant._epoch)
    run(r)
    assert r.assistant._epoch > epochs[0]  # interrupt_speech ran
    assert r.heard == ["stop talking"]  # and the turn that interrupted was not dropped by it


def test_no_barge_in_is_made_when_glide_is_not_speaking():
    stt = ScriptedSTT(["hello"])
    r = rig([*speech(), *quiet(700)], stt, playing=False)
    run(r)
    assert r.assistant._epoch == 0


# -- self-correction --------------------------------------------------------------------------------


def test_a_correction_inside_the_merge_window_is_one_turn_with_the_whole_utterance():
    stt = ScriptedSTT(["open safari no chrome"])
    run(rig([*speech(), *quiet(700), *speech(), *quiet(700), *quiet(1600)], stt, merge_window_s=1.5))
    assert len(stt.streams) == 1
    assert len(stt.audio[0]) // FRAME_BYTES > 2 * 10  # both halves went to the one request


def test_a_pause_longer_than_the_merge_window_ends_the_turn():
    stt = ScriptedSTT(["first", "second"])
    run(rig([*speech(), *quiet(700), *quiet(1000), *speech(), *quiet(700), *quiet(1000)], stt, merge_window_s=0.5))
    assert len(stt.streams) == 2


# -- bounds, faults, failures -----------------------------------------------------------------------


def test_speech_that_never_stops_is_discarded_not_submitted_half_heard():
    stt = ScriptedSTT(["must not be answered"])
    r = run(rig(speech(1900), stt))
    assert r.heard == [] and not r.assistant.busy
    assert any("seconds" in w for w in r.warned)
    assert r.player.played == []


def test_bursts_inside_the_merge_window_cannot_build_a_turn_longer_than_the_cap():
    """Each burst is under the limit and the pauses between them are shorter than the merge window, so no stretch of speech
    was over the cap on its own: the turn as a whole is what is limited."""
    burst = [*speech(1000), *quiet(224)]  # 32 s of speech, then the 7 frames that close a stretch at silence_ms = 200
    stt = ScriptedSTT(["must not be answered", "the rest"])
    r = run(rig([*burst, *burst, *burst, *quiet(1700)], stt, silence_ms=200, merge_window_s=1.0))
    assert any("exceeded 60 seconds" in w for w in r.warned)
    assert 0 not in stt.ended  # the turn that grew too long was discarded, not submitted
    assert max(len(audio) for audio in stt.audio.values()) // FRAME_BYTES <= 1875 + 8  # nothing longer than the cap went out


def test_a_microphone_fault_discards_the_turn_reports_it_and_ends_the_loop():
    stt = ScriptedSTT(["never"])
    r = run(rig([*speech(), DeviceFault("Microphone overflow; the incomplete command was discarded.")], stt))
    assert r.heard == [] and r.loop.failure and "overflow" in r.warned[0]


def test_a_failing_turn_is_reported_by_type_only_and_listening_goes_on():
    class Broken(ScriptedSTT):
        def stream(self, chunks, **kw):
            list(chunks)
            raise ValueError("secret-key-1234 in a message")
            yield  # pragma: no cover

    r = run(rig([*speech(), *quiet(700)], Broken([])))
    assert r.warned == ["a voice turn failed (ValueError)"]
    assert "secret" not in " ".join(r.warned)


# -- pause, resume, idle ----------------------------------------------------------------------------


def test_pause_ends_a_turn_being_said_and_turns_the_microphone_off_resume_turns_it_on():
    stt = ScriptedSTT(["dictated"])
    r = rig([*speech(), None, None], stt)
    r.device.on_frame = lambda left: r.loop.pause() if left == 2 else None
    run(r)
    assert r.heard == ["dictated"] and r.device.calls == ["pause"] and r.loop._paused
    r.loop._stop.clear()
    r.loop.resume()
    r.loop._apply_commands()
    assert r.device.calls == ["pause", "resume"] and not r.loop._paused


def test_idle_turns_the_microphone_off_after_the_configured_time_and_tells_the_caller():
    now = [0.0]
    told = []
    stt = ScriptedSTT([])
    r = rig([None] * 8, stt, idle_s=30, clock=lambda: now[0], on_idle=lambda: told.append(now[0]))
    r.device.on_frame = lambda left: now.__setitem__(0, now[0] + 10)
    run(r)
    assert told == [30.0] and r.device.calls == ["pause"]


def test_activity_keeps_the_microphone_on():
    now = [0.0]
    r = rig([None] * 8, ScriptedSTT([]), idle_s=30, clock=lambda: now[0], playing=True)
    r.device.on_frame = lambda left: now.__setitem__(0, now[0] + 10)
    run(r)
    assert r.device.calls == []


# -- more recorded failures -------------------------------------------------------------------------


def test_a_short_command_after_a_long_idle_sends_nothing_for_the_idle_and_ends_once():
    """Replay of the recorded double commit: 20 s of idle must not reach the transcriber (it once counted toward a
    proactive flush), and the one short command is one stream with one end."""
    stt = ScriptedSTT(["open youtube and play the first video"])
    r = run(rig([*quiet(20_000), *speech(3), *quiet(700), *quiet(700)], stt))
    assert r.heard == ["open youtube and play the first video"]
    assert stt.streams == [0] and stt.ended == [0]
    assert len(stt.audio[0]) // FRAME_BYTES == 8 + 3 - 1 + 19  # the pre-roll, the speech and the closing silence: not 625 frames


def test_stopping_the_loop_discards_a_half_said_command_instead_of_submitting_it():
    stt = ScriptedSTT(["must not be answered"])
    r = run(rig([*speech(10)], stt))
    assert r.heard == [] and stt.ended == [] and not r.warned  # an abort is on purpose, so it is not an error either


def test_the_assistants_heard_hook_is_restored_when_the_loop_ends():
    stt = ScriptedSTT([])
    r = rig([], stt)
    wrapped = r.assistant.io.heard
    assert wrapped.__name__ == "on_heard"
    run(r)
    assert r.assistant.io.heard == r.heard.append


def test_a_voice_request_to_act_is_a_dry_run_unless_the_caller_said_otherwise():
    seen = []

    class Recorder:
        io = IO()
        busy = False

        def handle_audio(self, chunks, **kwargs):
            seen.append(kwargs["act"])
            list(chunks)

        def interrupt_speech(self):
            pass

        def wait_idle(self, timeout=None):
            return True

    # asking to act is not acting: with confirmation (the default, see test_voice_confirm.py) the request runs as a dry run first
    for kwargs in ({}, {"act": True, "confirm_tasks": False}, {"act": True}):
        device = ScriptedDevice([*speech(3), *quiet(700)])
        loop = VoiceLoop(Recorder(), device, vad, **kwargs)
        device.loop = loop
        loop.stop_soon = loop._stop.set
        loop.run()
        loop.join_turns(WAIT)
    assert seen == [False, True, False]


# -- internal counters never reach the TTS ----------------------------------------------------------


@pytest.mark.parametrize("outcome", ["stalled", "step limit", "provider failure", "done", "crashed", "dry run", "nothing helps"])
def test_a_task_result_is_spoken_without_its_counters(outcome):
    """`steps`, `seconds` and the run folder are on the result for the terminal; only `spoken()` goes to the TTS."""
    result = TaskResult(goal="open the page", act=True, outcome=outcome, steps=7, seconds=12.5, folder=Path("runs/run-0042"))
    r = rig([], ScriptedSTT([]))
    r.assistant._finish_task(SimpleNamespace(result=result, stop_requested=False), "en")
    assert wait_until(lambda: r.tts.calls)
    r.assistant.wait_idle(WAIT)
    said = " ".join(text for text, _ in r.tts.calls)
    assert said == result.spoken("en")
    assert not any(ch.isdigit() for ch in said) and "runs/" not in said


# -- an answer that is not audible yet --------------------------------------------------------------


def streamed_answer(gate):
    """An LLM whose answer (no ready-made reply: it is streamed) is held back until `gate` is set."""
    return FakeLLM(route=route_json("answer"), deltas=["Answer one. "], gates={0: gate})


def test_speech_over_an_answer_that_is_still_being_written_cuts_it_as_the_new_turn_starts():
    """The device is silent while the model is still writing, so the gate never sees Glide speaking: the answer used to go
    on and speak over the person's new request."""
    gate = threading.Event()
    llm = streamed_answer(gate)
    stt = ScriptedSTT(["tell me a story", "wait, never mind that story"])
    r = rig([*speech(), *quiet(700), *speech(), *quiet(700)], stt, llm=llm)
    first_b = len(speech()) + len(quiet(700))  # the index of B's first frame
    total = first_b + len(speech()) + len(quiet(700))
    epochs = {}

    def hook(left):
        index = total - left
        if index == first_b:
            assert wait_until(lambda: llm.stream_calls)  # A has heard its transcript and is being written, silently
        epochs[index] = r.assistant._epoch
        if left == 0:
            gate.set()

    r.device.on_frame = hook
    run(r)
    assert epochs[first_b + 2] > epochs[first_b]  # interrupt_speech ran when B began
    assert sorted(r.heard) == ["tell me a story", "wait, never mind that story"]  # B was not dropped by it
    assert wait_until(lambda: r.tts.calls) and r.assistant.wait_idle(WAIT)
    assert len(r.tts.calls) == 1  # only B's answer was ever spoken; A's, released late, was already cancelled


def test_a_request_still_awaiting_its_transcript_is_spared_when_an_answer_is_cut():
    gate, stt_gate = threading.Event(), threading.Event()
    llm = streamed_answer(gate)
    stt = ScriptedSTT(["first", "second", "third"], gates={1: stt_gate})
    r = rig([*speech(), *quiet(700), *speech(), *quiet(700), *speech(), *quiet(700)], stt, llm=llm)
    one, two = len(speech()) + len(quiet(700)), 2 * (len(speech()) + len(quiet(700)))
    total = 3 * one
    epochs = {}

    def hook(left):
        index = total - left
        if index == one:
            assert wait_until(lambda: llm.stream_calls)  # the first answer is being written
        if index == two:
            assert wait_until(lambda: len(stt.streams) == 2)  # the second request has reached the transcriber and waits there
        epochs[index] = r.assistant._epoch
        if left == 0:
            gate.set()
            stt_gate.set()

    r.device.on_frame = hook
    run(r)
    assert epochs[one + 2] > epochs[one]  # the first answer was cut, dropping what was pending
    assert epochs[two + 2] == epochs[two]  # the third turn cut it again but spared the second request, still being heard
    assert "second" in r.heard and "third" in r.heard


def test_nothing_is_cut_when_the_earlier_answer_is_finished_and_spoken():
    stt = ScriptedSTT(["first", "second"])
    r = rig([*speech(), *quiet(700), *speech(), *quiet(700)], stt)
    first_b = len(speech()) + len(quiet(700))
    total = 2 * first_b
    epochs = {}

    def hook(left):
        index = total - left
        if index == first_b:
            r.loop.join_turns(WAIT)
            assert r.assistant.wait_idle(WAIT)  # the first answer is over, and played
        epochs[index] = r.assistant._epoch

    r.device.on_frame = hook
    run(r)
    assert set(epochs.values()) == {0} and sorted(r.heard) == ["first", "second"]


def test_speech_that_is_queued_but_not_audible_yet_is_cut_too():
    """The answer is complete and its thread is gone, but the voice is still being made: the speaker has pending sentences
    and the device is silent."""
    release = threading.Event()
    tts = FakeTTS(release=release)
    stt = ScriptedSTT(["first", "second"])
    r = rig([*speech(), *quiet(700), *speech(), *quiet(700)], stt, tts=tts)
    first_b = len(speech()) + len(quiet(700))
    total = 2 * first_b
    epochs = {}

    def hook(left):
        index = total - left
        if index == first_b:
            r.loop.join_turns(WAIT)  # the first request is finished being answered
            assert wait_until(lambda: tts.calls)  # and its voice is being made, not yet heard
        epochs[index] = r.assistant._epoch
        if left == 0:
            release.set()

    r.device.on_frame = hook
    run(r)
    assert epochs[first_b + 2] > epochs[first_b]

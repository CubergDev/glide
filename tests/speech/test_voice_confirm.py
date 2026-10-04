"""With `act`, hands-free speech cannot start a task that clicks and types until the person says the confirm phrase.

Any audible speech reaches the loop (no wake word, no speaker check), so a request that becomes a computer task runs as a
dry run first, and only the phrase, within the time, as the whole utterance, makes the same request real. Silence, any
other sentence, a stop, being talked over, a pause or the clock running out is a no. Nothing here touches a screen: the
loop in glide/computer is replaced by a recorder of the `act` it was given.
"""

# ruff: noqa: RUF001  one sentence is checked in Mandarin, with its full-width punctuation

from __future__ import annotations

import contextlib
import threading
from types import SimpleNamespace

import pytest
from guards_voice import no_real_audio  # noqa: F401
from test_assistant_fakes import WAIT, FakeConfig, FakeLLM, FakePlayer, FakeTTS, route_json, wait_until
from test_voice_loop import ScriptedDevice, ScriptedSTT, quiet, speech, vad

from glide.assistant.core import IO, Assistant
from glide.computer import runner
from glide.computer.platform_adapter import desktop
from glide.computer.runner import RunState
from glide.speech.approval import TaskApproval
from glide.speech.turns import VoiceLoop

PHRASE = "confirm and run it"
UTTERANCE = len(speech()) + len(quiet(700))  # frames of one spoken sentence and the silence that ends it


def rig(tmp_path, monkeypatch, script, finals, *, outcome="dry run", act=True, **loop_kw):
    acts = []

    def fake_run(cfg, ctx_factory, classifier_factory=None, control=None):
        acts.append(cfg.act)
        return RunState(outcome="done" if cfg.act else outcome)

    monkeypatch.setattr(runner, "run", fake_run)
    monkeypatch.setattr(desktop, "accessibility_trusted", lambda: True)
    llm = FakeLLM(route=route_json("computer", goal="Open the tickets page", language="en"))
    stt, shown, heard, warned = ScriptedSTT(finals), [], [], []
    config = FakeConfig(llm=llm, stt=stt, tts=FakeTTS(), classifier=object(), writer=object())
    io = IO(player=FakePlayer(), show=shown.append, heard=heard.append, warn=warned.append)
    assistant = Assistant(config, io=io, runs_dir=tmp_path / "runs")
    device, now = ScriptedDevice(script), [1000.0]
    loop = VoiceLoop(assistant, device, vad, act=act, clock=lambda: now[0], **loop_kw)
    device.loop = loop
    loop.stop_soon = loop._stop.set
    return SimpleNamespace(
        loop=loop, device=device, assistant=assistant, acts=acts, llm=llm, shown=shown, warned=warned, now=now, stt=stt
    )


def at_second_sentence(r, total, action=None):
    """Before the second sentence is heard, wait for the first request to have been previewed and offered."""

    def hook(left):
        if total - left == UTTERANCE:
            assert wait_until(lambda: r.loop.awaiting_confirmation)
            r.loop.join_turns(WAIT)  # the turn that made the offer is over: a person who speaks while it is not is interrupting
            if action:
                action()

    r.device.on_frame = hook


def finish(r):
    """Run the loop over its script. The loop ends when the script does, which discards an unanswered window as stopping
    must, so before the last frame the turns heard so far are let finish (a real loop does not end in the middle of one)."""
    user = r.device.on_frame

    def hook(left):
        if user:
            user(left)
        if left == 1:
            r.loop.join_turns(WAIT)

    r.device.on_frame = hook
    r.loop.run()
    r.loop.join_turns(WAIT)
    task = r.assistant.task
    if task is not None:
        assert task.wait(WAIT)
    return r


def two_sentences():
    return [*speech(), *quiet(700), *speech(), *quiet(700)]


def lapsed(r):
    return [line for line in r.shown if line == "Not confirmed, so nothing was done."]


def test_a_spoken_request_to_act_is_previewed_and_the_phrase_makes_it_real(tmp_path, monkeypatch):
    r = rig(tmp_path, monkeypatch, two_sentences(), ["open the tickets page", PHRASE])
    at_second_sentence(r, 2 * UTTERANCE)
    finish(r)
    assert r.acts == [False, True]  # a dry run first, then the same request for real
    assert any(PHRASE in line and "within 10 seconds" in line for line in r.shown)  # the person was told what to say
    asked = [call["messages"][-1]["content"] for call in r.llm.chat_calls]
    assert asked == ["open the tickets page", "open the tickets page"]  # the phrase itself was never routed as a request
    assert not r.loop.awaiting_confirmation and r.warned == []


def test_the_phrase_must_be_the_whole_utterance_and_in_time(tmp_path, monkeypatch):
    r = rig(tmp_path, monkeypatch, two_sentences(), ["open the tickets page", "yes please confirm and run it now"])
    at_second_sentence(r, 2 * UTTERANCE)
    finish(r)
    assert True not in r.acts and len(lapsed(r)) >= 1  # the sentence was a no, and then a new request of its own


def test_another_request_in_the_window_is_a_no_and_is_itself_only_previewed(tmp_path, monkeypatch):
    r = rig(tmp_path, monkeypatch, two_sentences(), ["open the tickets page", "open another page"])
    at_second_sentence(r, 2 * UTTERANCE)
    finish(r)
    assert r.acts == [False, False] and len(lapsed(r)) >= 1


def test_a_stop_in_the_window_is_a_no_and_a_later_phrase_confirms_nothing(tmp_path, monkeypatch):
    script = [*two_sentences(), *speech(), *quiet(700)]
    r = rig(tmp_path, monkeypatch, script, ["open the tickets page", "stop", PHRASE])
    at_second_sentence(r, 3 * UTTERANCE)
    finish(r)
    assert True not in r.acts and len(lapsed(r)) >= 1


def test_silence_past_the_time_is_a_no(tmp_path, monkeypatch):
    r = rig(tmp_path, monkeypatch, [*speech(), *quiet(700), *quiet(3200)], ["open the tickets page"])

    def hook(left):
        if left == 50 and wait_until(lambda: r.loop.awaiting_confirmation):
            r.now[0] += 11  # past the 10 s window

    r.device.on_frame = hook
    finish(r)
    assert not r.loop.awaiting_confirmation and len(lapsed(r)) == 1 and r.acts == [False]


def test_speaking_over_glide_while_a_task_waits_is_a_no_so_the_phrase_after_it_confirms_nothing(tmp_path, monkeypatch):
    r = rig(tmp_path, monkeypatch, two_sentences(), ["open the tickets page", PHRASE])
    at_second_sentence(r, 2 * UTTERANCE, action=lambda: setattr(r.device, "playing", True))  # Glide is speaking again
    finish(r)
    assert True not in r.acts and len(lapsed(r)) >= 1


def test_a_pause_closes_the_window(tmp_path, monkeypatch):
    r = rig(tmp_path, monkeypatch, [*speech(), *quiet(700), None, None], ["open the tickets page"])

    def hook(left):
        if left == 2:
            assert wait_until(lambda: r.loop.awaiting_confirmation)
            r.loop.pause()

    r.device.on_frame = hook
    finish(r)
    assert not r.loop.awaiting_confirmation and len(lapsed(r)) == 1 and r.acts == [False]


def test_a_pause_closes_the_window_at_once_not_only_when_the_loop_finally_ends(tmp_path, monkeypatch):
    r = rig(tmp_path, monkeypatch, [*speech(), *quiet(700), *[None] * 400000], ["open the tickets page"])
    r.device.pause_input = lambda: setattr(r.device, "paused", True)  # unlike the scripted one, it leaves the loop running
    r.loop.start()
    try:
        assert wait_until(lambda: r.loop.awaiting_confirmation)
        r.loop.pause()
        assert wait_until(lambda: r.device.paused)
        assert not r.loop.awaiting_confirmation and len(lapsed(r)) == 1  # closed by the pause, while the loop is still alive
    finally:
        r.loop.stop()


def test_a_loop_that_ends_with_a_window_open_closes_it(tmp_path, monkeypatch):
    r = rig(tmp_path, monkeypatch, [*speech(), *quiet(700), *quiet(100)], ["open the tickets page"])
    finish(r)  # the previewed task was offered, and then the loop ended before anyone answered
    assert any(PHRASE in line for line in r.shown)
    assert not r.loop.awaiting_confirmation and len(lapsed(r)) == 1 and r.acts == [False]


def test_the_keyed_yes_makes_the_waiting_task_real_once(tmp_path, monkeypatch):
    r = rig(tmp_path, monkeypatch, [*speech(), *quiet(700), None, None, None], ["open the tickets page"])
    results = []

    def hook(left):
        if left == 3:
            assert wait_until(lambda: r.loop.awaiting_confirmation)
            results.extend([r.loop.confirm(), r.loop.confirm()])

    r.device.on_frame = hook
    finish(r)
    assert results == [True, False] and r.acts == [False, True]


def test_a_keyed_yes_with_nothing_waiting_does_nothing(tmp_path, monkeypatch):
    r = rig(tmp_path, monkeypatch, [], [])
    assert r.loop.confirm() is False and r.acts == []


def test_a_task_that_was_not_a_dry_run_that_ended_is_not_offered(tmp_path, monkeypatch):
    r = rig(tmp_path, monkeypatch, [*speech(), *quiet(700), *quiet(100)], ["open the tickets page"], outcome="stalled")
    finish(r)
    assert not r.loop.awaiting_confirmation and r.shown.count("Not confirmed, so nothing was done.") == 0
    assert not any(PHRASE in line for line in r.shown)


def test_without_act_nothing_is_offered_and_nothing_acts(tmp_path, monkeypatch):
    r = rig(tmp_path, monkeypatch, [*speech(), *quiet(700), *quiet(100)], ["open the tickets page"], act=False)
    finish(r)
    assert r.acts == [False] and not r.loop.awaiting_confirmation and not any(PHRASE in line for line in r.shown)


def test_confirmation_can_be_turned_off_on_purpose_and_then_the_request_acts_at_once(tmp_path, monkeypatch):
    r = rig(tmp_path, monkeypatch, [*speech(), *quiet(700)], ["open the tickets page"], confirm_tasks=False)
    finish(r)
    assert r.acts == [True]


def test_a_custom_phrase_replaces_the_default_one(tmp_path, monkeypatch):
    r = rig(tmp_path, monkeypatch, two_sentences(), ["open the tickets page", "go ahead glide"], confirm_phrase="go ahead glide")
    at_second_sentence(r, 2 * UTTERANCE)
    finish(r)
    assert r.acts == [False, True]


# -- the window itself ------------------------------------------------------------------------------


class Window:
    def __init__(self, timeout=10.0):
        self.now, self.shown = [100.0], []
        self.approval = TaskApproval(PHRASE, timeout, clock=lambda: self.now[0], show=self.shown.append)


@pytest.mark.parametrize("said", ["confirm and run it", "Confirm and run it!", "  CONFIRM,  and run it ", "confirm and run it."])
def test_the_phrase_is_compared_the_way_a_stop_phrase_is(said):
    w = Window()
    w.approval.open("open the page", "en")
    pending = w.approval.answer(said)
    assert pending is not None and pending.text == "open the page" and not w.approval.waiting


@pytest.mark.parametrize(
    "said", ["", "yes", "confirm", "run it", "please confirm and run it", "confirm and run it now", "confirm and run"]
)
def test_anything_else_closes_the_window_and_runs_nothing(said):
    w = Window()
    w.approval.open("open the page", "en")
    assert w.approval.answer(said) is None and not w.approval.waiting
    assert w.approval.answer(PHRASE) is None  # one answer only: the phrase after a no is not a yes


def test_the_phrase_after_the_deadline_is_a_no_and_the_deadline_itself_still_counts():
    w = Window(timeout=10.0)
    w.approval.open("x", None)
    w.now[0] += 10.0
    assert w.approval.answer(PHRASE) is not None
    w.approval.open("x", None)
    w.now[0] += 10.01
    assert w.approval.answer(PHRASE) is None and w.shown[-1] == "Not confirmed, so nothing was done."


def test_expire_closes_only_a_window_whose_time_has_run_out_and_cancel_closes_any():
    w = Window()
    w.approval.open("x", "zh")
    w.approval.expire()
    assert w.approval.waiting
    w.now[0] += 11
    w.approval.expire()
    assert not w.approval.waiting and w.shown[-1] == "没有确认，所以什么都没做。"
    w.approval.open("x", "en")
    w.approval.cancel()
    assert not w.approval.waiting
    n = len(w.shown)
    w.approval.cancel()
    w.approval.expire()
    assert len(w.shown) == n  # nothing open: nothing said


def test_the_window_tells_the_person_the_phrase_and_the_seconds_in_their_language():
    w = Window(timeout=7.4)
    w.approval.open("x", "yue")
    assert "confirm and run it" in w.shown[0] and "7秒" in w.shown[0]


def test_a_window_is_one_use_even_when_two_answers_race():
    w = Window()
    w.approval.open("x", None)
    winners, barrier = [], threading.Barrier(8)

    def answer():
        barrier.wait()
        winners.append(w.approval.answer(PHRASE))

    threads = [threading.Thread(target=answer) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(WAIT)
    assert sum(1 for p in winners if p is not None) == 1


def test_two_threads_starting_turns_at_once_both_stay_in_the_list():
    """audit2 finding 6: `_start` (loop thread) and `confirm` (UI thread) lost a turn from `_turns`."""
    loop = VoiceLoop.__new__(VoiceLoop)
    barrier = threading.Barrier(2)

    class Slow:
        """A live earlier turn whose liveness check is where the two callers overlap."""

        def is_alive(self):
            with contextlib.suppress(threading.BrokenBarrierError):  # serialised by a lock: the other caller is not here
                barrier.wait(0.3)
            return True

    earlier = SimpleNamespace(thread=Slow())
    loop._turns = [earlier]
    loop._turns_lock = threading.Lock()
    hold = threading.Event()
    loop._work = lambda turn: hold.wait(WAIT)  # a running worker, so it is not filtered out as finished
    turns = [SimpleNamespace(thread=None), SimpleNamespace(thread=None)]
    callers = [threading.Thread(target=loop._start, args=(t,)) for t in turns]
    for c in callers:
        c.start()
    for c in callers:
        c.join(WAIT)
    hold.set()
    assert all(t in loop._turns for t in turns)

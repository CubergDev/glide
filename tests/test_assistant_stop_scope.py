"""A stop reaches what was in progress before it, never a request that began after it (findings 2 and 4).

Order is decided by the order requests were made (a ticket taken when `handle_text` or `handle_audio` is called), not by
a clock, so two requests a microsecond apart are still told apart. Every test holds a request at a provider
connection or a gate and lets the other thread go through, so nothing here waits for time to pass; the only timed waits
are the bounded ones that prove a thread is held.
"""

from __future__ import annotations

import threading

from test_assistant_core import build, spoken
from test_assistant_fakes import WAIT, FakeClassifier, route_json, wait_until
from test_cancel_assistant import Hold, Model, in_thread, llm_of, loud
from test_cancel_chain import Connection

from glide.computer import runner
from glide.computer.runner import RunState
from glide.providers import interrupt
from glide.providers.base import Transcript
from glide.providers.chain import Chain, Slot
from glide.providers.stt import STT


class Mic:
    """An STT slot that serves one utterance per call, each on a connection of its own, finals in the order given."""

    name = "stt"
    model = "s"

    def __init__(self, *finals: str) -> None:
        self.finals = list(finals)
        self.connections: list[Connection] = []
        self._lock = threading.Lock()

    def stream(self, chunks, *, sample_rate=16000, language=None):
        for _ in chunks:
            pass
        with self._lock:
            connection = Connection()
            self.connections.append(connection)
            final = self.finals.pop(0)
        with interrupt.closing(connection.close, self.name):
            connection.wait()
        interrupt.check(self.name)
        yield Transcript(final, "en", self.name, self.model, 0.0)

    def transcribe(self, audio, *, language=None, prompt=None, timeout=None):
        return Transcript("", "en", self.name, self.model, 0.0)


def mic_rig(tmp_path, mic: Mic, chats, **kw):
    rig = build(tmp_path, llm=llm_of(Model(chats=chats)), stt=STT(Chain("stt", [Slot("stt", mic)])), **kw)
    rig.mic = mic
    return rig


def hear(rig):
    """Begin an utterance on the microphone and return once it is being heard, so the next one is made after it."""
    count = len(rig.mic.connections)
    done, box = in_thread(lambda: rig.assistant.handle_audio(iter(loud())))
    assert wait_until(lambda: len(rig.mic.connections) == count + 1)
    assert rig.mic.connections[count].opened.wait(WAIT)
    return done, box


# -- finding 2: a stop is for what came before it ------------------------------------------------------


def test_a_stop_word_does_not_cancel_an_utterance_that_began_after_it(tmp_path):
    """F2: the stop word's final transcript called the generic stop, which cancelled every live request, among them the
    next utterance, already being heard."""
    rig = mic_rig(tmp_path, Mic("stop", "what is two and two"), [route_json("answer", reply="Four.")])
    stop_done, stop_box = hear(rig)  # the stop word begins
    new_done, new_box = hear(rig)  # a new utterance begins before the stop word's transcript has arrived

    rig.mic.connections[0].release.set()  # the stop word is transcribed: stop

    assert stop_done.wait(WAIT) and stop_box["value"].route == "stop"
    assert not rig.mic.connections[1].closed.is_set()  # the new utterance was not cut off
    rig.mic.connections[1].release.set()
    assert new_done.wait(WAIT)
    assert new_box["value"].text == "Four." and rig.heard == ["stop", "what is two and two"]
    assert rig.warned == []


def test_a_stop_the_model_heard_does_not_cancel_an_utterance_that_began_after_the_request(tmp_path):
    """F2: the same for the stop the router detects (the fast path did not know the phrase)."""
    hold = Hold(then=route_json("stop"))
    rig = mic_rig(tmp_path, Mic("what is two and two"), [hold, route_json("answer", reply="Four.")])
    routed_done, routed = in_thread(lambda: rig.assistant.handle_text("hmm, forget the plan"))
    assert hold.connection.opened.wait(WAIT)
    new_done, new_box = hear(rig)  # the next utterance begins while the router is still thinking

    hold.connection.release.set()  # the router says: stop

    assert routed_done.wait(WAIT) and routed["value"].route == "stop"
    assert not rig.mic.connections[0].closed.is_set()
    rig.mic.connections[0].release.set()
    assert new_done.wait(WAIT) and new_box["value"].text == "Four."


def test_a_stop_word_still_cancels_an_utterance_that_began_before_it(tmp_path):
    rig = mic_rig(tmp_path, Mic("tell me everything"), [])
    old_done, old_box = hear(rig)

    assert rig.assistant.handle_text("stop").route == "stop"

    assert old_done.wait(WAIT) and rig.mic.connections[0].closed.is_set()
    assert old_box["value"].route == "none" and rig.heard == []


def test_a_stop_word_does_not_stop_a_task_that_a_later_request_started(tmp_path, monkeypatch):
    """The task belongs to the newer request, so the older stop word is not for it. A stop of its own is."""
    started, release, controls = threading.Event(), threading.Event(), []

    def fake_run(cfg, ctx_factory, classifier_factory=None, control=None):
        controls.append(control)
        started.set()
        release.wait(WAIT)
        return RunState(outcome="aborted (stopped by the user)" if control.cancelled.is_set() else "done")

    monkeypatch.setattr(runner, "run", fake_run)
    rig = mic_rig(
        tmp_path,
        Mic("stop"),
        [route_json("computer", reply="On it.", goal="Open the file")],
        writer=object(),
        classifier=FakeClassifier(None),
    )
    stop_done, _ = hear(rig)  # the stop word begins
    task = rig.assistant.handle_text("open the file", wait=False).task  # a request made after it starts a task
    assert started.wait(WAIT)

    rig.mic.connections[0].release.set()
    assert stop_done.wait(WAIT)

    assert rig.assistant.busy and not controls[0].cancelled.is_set()  # the older stop did not reach it
    assert rig.assistant.stop() is True and controls[0].cancelled.is_set()
    release.set()
    assert task.wait(WAIT)


# -- finding 4: the cut of speech is scoped the same way -----------------------------------------------


def test_a_sentence_a_newer_request_queues_while_a_stop_is_cutting_the_speech_is_not_cut(tmp_path):
    """F4: the turns were cancelled under the lock but the speech was cut after it was released, so a request that
    began in between had its first sentence cut by a stop that was not meant for it."""
    rig = build(tmp_path, llm=llm_of(Model(chats=[route_json("answer", reply="Hello."), route_json("answer", reply="Paris.")])))
    rig.assistant.handle_text("first")
    assert rig.assistant.wait_idle(WAIT)
    speaker = rig.assistant._speaker
    cutting, newer_queued, calls = threading.Event(), rig.tts.event_for("Paris."), []
    real_cancel = speaker.cancel

    def cancel_with_a_gap():
        calls.append(1)
        if len(calls) == 1:  # the stop's cut: hold it here until the newer request has said its sentence
            cutting.set()
            newer_queued.wait(0.3)
        real_cancel()

    speaker.cancel = cancel_with_a_gap
    stopper = threading.Thread(target=rig.assistant.stop)
    stopper.start()
    assert cutting.wait(WAIT)

    newer_done, newer = in_thread(lambda: rig.assistant.handle_text("second"))
    assert newer_done.wait(WAIT)
    stopper.join(WAIT)

    assert newer["value"].text == "Paris."
    assert rig.assistant.wait_idle(WAIT)
    assert spoken(rig)[-1] == "Paris." and rig.player.played, "the newer request's sentence was cut by the older stop"


def test_a_stop_from_an_older_request_does_not_cut_the_speech_of_a_newer_one(tmp_path):
    """The stop word began first, a typed request was answered and is being said, then the stop word's transcript arrives."""
    rig = mic_rig(tmp_path, Mic("stop"), [route_json("answer", reply="Paris.")])
    stop_done, _ = hear(rig)
    assert rig.assistant.handle_text("capital of France").text == "Paris."
    assert rig.assistant.wait_idle(WAIT) and rig.player.played
    cuts = rig.player.cancels

    rig.mic.connections[0].release.set()
    assert stop_done.wait(WAIT)

    assert rig.player.cancels == cuts and rig.player.played, "the older stop cut speech it was not meant for"


def test_a_stop_still_cuts_the_speech_of_a_request_that_began_before_it(tmp_path):
    rig = build(tmp_path, llm=llm_of(Model(chats=[route_json("answer", reply="Paris.")])))
    rig.assistant.handle_text("capital of France")
    assert rig.assistant.wait_idle(WAIT)
    cuts = rig.player.cancels

    rig.assistant.handle_text("stop")

    assert rig.player.cancels == cuts + 1

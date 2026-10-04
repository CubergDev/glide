"""Interrupting the assistant: the request in flight is cancelled at the provider, speech stops at once, a late answer is
never spoken, and a question a task is waiting on is not answered by the next thing the person says (D2, D3).

The models, the voice and the transcriber are real chains (providers/chain.py) over fake clients that block on an
Event, as a provider does while it is thinking. A test waits for the thing it needs to have happened (an Event, a
join) and never for time to pass; `WAIT` is only the longest it waits before deciding something is not coming.
"""

from __future__ import annotations

import threading

import pytest
from test_assistant_core import build, spoken
from test_assistant_fakes import WAIT, FakeClassifier, FakeLLM, route_json, wait_until
from test_cancel_chain import Connection

from glide.assistant.audio_io import chunked
from glide.assistant.core import Assistant
from glide.assistant.tasks import ComputerTask
from glide.computer import runner
from glide.computer.control import controlled
from glide.computer.models import Abort
from glide.computer.runner import RunState
from glide.providers import interrupt
from glide.providers.base import ChatResult, SpeechAudio, Transcript, Usage
from glide.providers.chain import Chain, Slot
from glide.providers.errors import ProviderError
from glide.providers.llm import LLM
from glide.providers.stt import STT
from glide.providers.tts import TTS

ANSWER_ROUTE = route_json("answer", reply="")


class Hold:
    """A request the provider is still thinking about: `connection` is what a cancel is supposed to close."""

    def __init__(self, then: str | None = None) -> None:
        self.connection = Connection()
        self.then = then  # what it answers if it is let go: a late answer


class Model:
    """A chat slot. Each call takes its next scripted step: a reply, or a `Hold`. A stream is a list of deltas and Holds."""

    name = "slot"
    model = "m"

    def __init__(self, chats=(), streams=()) -> None:
        self.chats = list(chats)
        self.streams = list(streams)
        self.chat_calls = 0

    def _wait(self, hold: Hold):
        with interrupt.closing(hold.connection.close, self.name):
            hold.connection.wait()
        if hold.then is None:
            raise ProviderError("slot could not be reached", kind="transport", provider=self.name)
        return hold.then

    def chat(self, messages, **kw) -> ChatResult:
        self.chat_calls += 1
        step = self.chats.pop(0)
        text = self._wait(step) if isinstance(step, Hold) else step
        return ChatResult(text, Usage(), self.name, self.model, 0.0)

    def stream(self, messages, **kw):
        for step in self.streams.pop(0):
            yield self._wait(step) if isinstance(step, Hold) else step


def llm_of(model: Model) -> LLM:
    return LLM(Chain("llm.fast", [Slot("slot", model)]))


def in_thread(call):
    done, box = threading.Event(), {}

    def target():
        try:
            box["value"] = call()
        except BaseException as exc:
            box["error"] = exc
        finally:
            done.set()

    threading.Thread(target=target, daemon=True).start()
    return done, box


# -- a request in flight ------------------------------------------------------------------------------


def test_a_newer_request_cancels_the_one_still_being_routed_and_the_old_one_says_nothing(tmp_path):
    hold = Hold(then=route_json("answer", reply="Too late."))
    model = Model(chats=[hold, route_json("answer", reply="Paris.")])
    rig = build(tmp_path, llm=llm_of(model))
    first_done, first = in_thread(lambda: rig.assistant.handle_text("first request"))
    assert hold.connection.opened.wait(WAIT)

    second = rig.assistant.handle_text("second request")

    assert first_done.wait(WAIT), "the older request is still blocked in its provider call"
    assert hold.connection.closed.is_set()  # its connection was closed, not waited out
    assert second.text == "Paris." and first["value"].text == ""
    assert rig.warned == []  # an interruption is not a failure
    rig.assistant.wait_idle(WAIT)
    assert spoken(rig) == ["Paris."] and rig.shown == ["Paris."]  # the late answer to the first was never shown or said
    assert [m["content"] for m in rig.assistant._history] == ["second request", "Paris."]


def test_a_stop_phrase_cancels_the_request_in_flight_at_the_provider(tmp_path):
    hold = Hold()
    rig = build(tmp_path, llm=llm_of(Model(chats=[hold])))
    done, _ = in_thread(lambda: rig.assistant.handle_text("tell me a long story"))
    assert hold.connection.opened.wait(WAIT)
    assert rig.assistant.handle_text("stop").route == "stop"
    assert done.wait(WAIT) and hold.connection.closed.is_set()
    assert rig.warned == [] and spoken(rig) == [] and rig.shown == []


def test_an_answer_that_arrives_after_the_interruption_is_never_spoken(tmp_path):
    hold = Hold(then=route_json("answer", reply="Far too late."))
    rig = build(tmp_path, llm=llm_of(Model(chats=[hold])))
    done, box = in_thread(lambda: rig.assistant.handle_text("hello"))
    assert hold.connection.opened.wait(WAIT)
    rig.assistant.interrupt_speech()
    hold.connection.release.set()  # the provider answers anyway, after the cancel
    assert done.wait(WAIT)
    rig.assistant.wait_idle(WAIT)
    assert spoken(rig) == [] and rig.shown == [] and rig.warned == [] and box["value"].text == ""


def test_an_answer_cut_off_mid_stream_stops_being_spoken_and_its_connection_is_closed(tmp_path):
    hold = Hold()
    model = Model(chats=[ANSWER_ROUTE], streams=[["First sentence. ", "Then", hold, " the second sentence. "]])
    rig = build(tmp_path, llm=llm_of(model))
    done, box = in_thread(lambda: rig.assistant.handle_text("explain"))
    assert hold.connection.opened.wait(WAIT)
    assert rig.tts.event_for("First sentence.").wait(WAIT)

    rig.assistant.interrupt_speech()

    assert done.wait(WAIT) and hold.connection.closed.is_set()
    rig.assistant.wait_idle(WAIT)
    assert spoken(rig) == ["First sentence."]  # nothing after the cut
    assert rig.warned == [] and box["value"].text == "First sentence."
    assert list(rig.assistant._history) == []  # an interrupted answer is not remembered


def test_without_the_interruption_the_same_answer_is_spoken_whole(tmp_path):
    """The control case: it is the cancel that stops the speech, not the shape of the script."""
    model = Model(chats=[ANSWER_ROUTE], streams=[["First sentence. ", "Second sentence. "]])
    rig = build(tmp_path, llm=llm_of(model))
    assert rig.assistant.handle_text("explain").text == "First sentence. Second sentence."
    rig.assistant.wait_idle(WAIT)
    assert spoken(rig) == ["First sentence.", "Second sentence."]


def test_closing_the_assistant_completes_while_a_provider_call_is_blocked(tmp_path):
    hold = Hold()
    rig = build(tmp_path, llm=llm_of(Model(chats=[hold])))
    request_done, _ = in_thread(lambda: rig.assistant.handle_text("hello"))
    assert hold.connection.opened.wait(WAIT)
    closed, _ = in_thread(rig.assistant.close)
    assert closed.wait(WAIT), "close() waited for the provider"
    assert request_done.wait(WAIT) and hold.connection.closed.is_set() and rig.player.closed


# -- speech in progress -------------------------------------------------------------------------------


class Voice:
    """A TTS client: the first chunk of a sentence arrives, then the connection is held, as when a sentence is long."""

    name = "voice"
    model = "v"
    sample_rate = 24000

    def __init__(self) -> None:
        self.connection = Connection()
        self.calls: list[str] = []

    def stream(self, text, *, voice=None, language=None):
        self.calls.append(text)
        with interrupt.closing(self.connection.close, self.name):
            yield b"\x01\x00\x02\x00"
            self.connection.wait()
        interrupt.check(self.name)
        yield b"\x03\x00"

    def synthesize(self, text, **kw) -> SpeechAudio:  # pragma: no cover  (the assistant only streams)
        raise NotImplementedError


def test_speech_in_progress_stops_at_once_its_connection_closes_and_queued_sentences_are_dropped(tmp_path):
    voice = Voice()
    tts = TTS(Chain("tts", [Slot("voice", voice)]))
    model = Model(chats=[route_json("answer", reply="One. Two. Three.")])
    rig = build(tmp_path, llm=llm_of(model), tts=tts)
    rig.assistant.handle_text("count")
    assert voice.connection.opened.wait(WAIT)
    assert wait_until(lambda: rig.player.played)  # the first chunk of the first sentence is playing

    rig.assistant.interrupt_speech()

    assert voice.connection.closed.is_set()  # the sentence being made was cut at the provider
    assert rig.player.cancels >= 1 and rig.player.played == []
    assert rig.assistant.wait_idle(WAIT)
    assert voice.calls == ["One."]  # "Two." and "Three." were queued and never asked for
    assert rig.warned == []  # a voice that was cut on purpose is not a speech failure


def test_a_sentence_said_after_the_cut_is_heard_in_full(tmp_path):
    voice = Voice()
    tts = TTS(Chain("tts", [Slot("voice", voice)]))
    model = Model(chats=[route_json("answer", reply="One."), route_json("answer", reply="Two.")])
    rig = build(tmp_path, llm=llm_of(model), tts=tts)
    rig.assistant.handle_text("first")
    assert voice.connection.opened.wait(WAIT)
    rig.assistant.interrupt_speech()
    voice.connection = Connection()  # a fresh connection for the next sentence
    voice.connection.release.set()
    rig.assistant.handle_text("second")
    assert rig.assistant.wait_idle(WAIT)
    assert voice.calls == ["One.", "Two."] and rig.warned == []
    assert [pcm for pcm, _ in rig.player.played] == [b"\x01\x00\x02\x00", b"\x03\x00"]  # all of it, none of the cut one


# -- a request being heard ----------------------------------------------------------------------------


class Listener:
    """An STT slot: it takes the audio, then waits for the end of the speech on a connection a cancel can close."""

    name = "stt"
    model = "s"

    def __init__(self, final: str = "") -> None:
        self.connection = Connection()
        self.final = final
        self.batch_calls: list = []

    def stream(self, chunks, *, sample_rate=16000, language=None):
        for _ in chunks:
            pass
        with interrupt.closing(self.connection.close, self.name):
            self.connection.wait()
        interrupt.check(self.name)
        yield Transcript(self.final, "en", self.name, self.model, 0.0)

    def transcribe(self, audio, *, language=None, prompt=None, timeout=None):
        self.batch_calls.append(audio)
        return Transcript("batch", "en", self.name, self.model, 0.0)


def loud() -> list[bytes]:
    from test_assistant_core import tone

    return list(chunked(tone(2000, 0.3)))


def test_a_barge_in_while_speech_is_being_transcribed_closes_the_transcriber_and_sends_nothing_else(tmp_path):
    listener = Listener(final="old request")
    stt = STT(Chain("stt", [Slot("stt", listener)]))
    rig = build(tmp_path, llm=llm_of(Model()), stt=stt)
    done, box = in_thread(lambda: rig.assistant.handle_audio(iter(loud())))
    assert listener.connection.opened.wait(WAIT)

    rig.assistant.interrupt_speech()

    assert done.wait(WAIT) and listener.connection.closed.is_set()
    assert box["value"].route == "none"
    assert listener.batch_calls == []  # the fallback request must not fire right after a stop
    assert rig.warned == [] and rig.heard == [] and spoken(rig) == []


def test_a_stop_while_speech_is_being_transcribed_ends_the_transcription(tmp_path):
    listener = Listener(final="old request")
    stt = STT(Chain("stt", [Slot("stt", listener)]))
    rig = build(tmp_path, llm=llm_of(Model()), stt=stt)
    done, box = in_thread(lambda: rig.assistant.handle_audio(iter(loud())))
    assert listener.connection.opened.wait(WAIT)
    rig.assistant.stop()
    assert done.wait(WAIT) and listener.connection.closed.is_set() and box["value"].route == "none"


def test_a_barge_in_that_keeps_pending_requests_leaves_the_one_being_heard_alone(tmp_path):
    """The voice loop passes drop_pending=False when an earlier request of the person's is still waiting for its transcript."""
    listener = Listener(final="what is two and two")
    stt = STT(Chain("stt", [Slot("stt", listener)]))
    model = Model(chats=[route_json("answer", reply="Four.")])
    rig = build(tmp_path, llm=llm_of(model), stt=stt)
    done, box = in_thread(lambda: rig.assistant.handle_audio(iter(loud())))
    assert listener.connection.opened.wait(WAIT)

    rig.assistant.interrupt_speech(drop_pending=False)

    assert not listener.connection.closed.is_set()
    listener.connection.release.set()  # the transcript arrives
    assert done.wait(WAIT)
    assert box["value"].text == "Four." and rig.heard == ["what is two and two"]


def test_a_request_typed_while_speech_is_still_being_heard_does_not_cancel_the_speech(tmp_path):
    listener = Listener(final="what is two and two")
    stt = STT(Chain("stt", [Slot("stt", listener)]))
    model = Model(chats=[route_json("answer", reply="Paris."), route_json("answer", reply="Four.")])
    rig = build(tmp_path, llm=llm_of(model), stt=stt)
    done, box = in_thread(lambda: rig.assistant.handle_audio(iter(loud())))
    assert listener.connection.opened.wait(WAIT)

    assert rig.assistant.handle_text("capital of France").text == "Paris."
    assert not listener.connection.closed.is_set()  # two requests a few seconds apart are both answered
    listener.connection.release.set()

    assert done.wait(WAIT) and box["value"].text == "Four."


def test_an_older_requests_transcript_arriving_late_does_not_cancel_a_newer_request_still_being_answered(tmp_path):
    """PR9-4175621615: requests are ordered by when they were made. The spoken one was made first, so its late
    transcript supersedes nothing that was made after it: the typed request is answered and remembered, and so is it."""
    listener = Listener(final="what is two and two")
    stt = STT(Chain("stt", [Slot("stt", listener)]))
    hold = Hold(then="Second sentence. ")
    model = Model(chats=[ANSWER_ROUTE, route_json("answer", reply="Four.")], streams=[["First sentence. ", hold]])
    rig = build(tmp_path, llm=llm_of(model), stt=stt)
    spoken_done, spoken_box = in_thread(lambda: rig.assistant.handle_audio(iter(loud())))
    assert listener.connection.opened.wait(WAIT)
    typed_done, typed_box = in_thread(lambda: rig.assistant.handle_text("capital of France"))
    assert hold.connection.opened.wait(WAIT)  # the typed request is mid-answer

    listener.connection.release.set()  # the older request's transcript arrives now
    assert spoken_done.wait(WAIT) and spoken_box["value"].text == "Four."
    hold.connection.release.set()

    assert typed_done.wait(WAIT)
    assert typed_box["value"].text == "First sentence. Second sentence."
    assert not hold.connection.closed.is_set()  # nothing closed its connection
    rig.assistant.wait_idle(WAIT)
    assert {"First sentence.", "Second sentence.", "Four."} <= set(spoken(rig))
    users = {m["content"] for m in rig.assistant._history if m["role"] == "user"}
    assert users == {"what is two and two", "capital of France"}


# -- the task ------------------------------------------------------------------------------------------


class Thinking:
    name = "slot"

    def __init__(self) -> None:
        self.connection = Connection()

    def __call__(self):
        with interrupt.closing(self.connection.close, self.name):
            self.connection.wait()
        return "late"


def test_a_stop_reaches_the_model_call_a_computer_task_is_waiting_on(tmp_path, monkeypatch):
    """Assistant.stop -> ComputerTask.stop -> the task's control -> the chain -> the connection: one path."""
    thinking, seen = Thinking(), []
    chain = Chain("llm.smart", [Slot("slot", thinking)])

    def fake_run(cfg, ctx_factory, classifier_factory=None, control=None):
        with controlled(control):  # as the real runner does
            try:
                chain.call(lambda slot: slot.client())
            except ProviderError as error:
                seen.append(error.kind)
        return RunState(outcome="aborted (stopped by the user)")

    monkeypatch.setattr(runner, "run", fake_run)
    model = Model(chats=[route_json("computer", reply="On it.", goal="Open Safari")])
    rig = build(tmp_path, llm=llm_of(model), writer=object(), classifier=FakeClassifier(None))
    reply = rig.assistant.handle_text("open safari", wait=False)
    assert thinking.connection.opened.wait(WAIT)

    assert rig.assistant.stop() is True

    assert reply.task.wait(WAIT) and thinking.connection.closed.is_set()
    assert seen == ["cancelled"] and list(chain.events) == []


# -- a question a task is waiting on --------------------------------------------------------------------


@pytest.fixture
def asking(monkeypatch):
    """The screen-driving loop replaced by one that asks the user a question and records what it got back."""
    state = type("State", (), {})()
    state.asked, state.replies, state.ask_was, state.goals = threading.Event(), [], [], []
    state.unwinding, state.release = threading.Event(), threading.Event()  # a stopped run takes its time to end
    state.release.set()

    def fake_run(cfg, ctx_factory, classifier_factory=None, control=None):
        ctx = ctx_factory(None, [])
        state.goals.append(cfg.goal)
        state.ask_was.append(ctx.ask)
        if ctx.ask is None or cfg.goal != "Open the file":
            return RunState(outcome="done")
        state.asked.set()
        try:
            state.replies.append(ctx.ask("Which one?"))
        except Abort as stopped:
            state.replies.append(stopped)
            state.unwinding.set()
            state.release.wait(WAIT)
            return RunState(outcome="aborted (stopped by the user)")
        return RunState(outcome="done")

    monkeypatch.setattr(runner, "run", fake_run)
    return state


def assistant_that_asks(tmp_path, chats, *, clarify=True, **kw):
    """An assistant built through its public `clarify` parameter, over a model that answers `chats` in turn."""
    rig = build(tmp_path, llm=llm_of(Model(chats=chats)), writer=object(), classifier=FakeClassifier(None), **kw)
    rig.assistant = Assistant(rig.config, io=rig.assistant.io, runs_dir=tmp_path / "runs", clarify=clarify)
    return rig


def start_asking_task(rig, asking):
    reply = rig.assistant.handle_text("open the file", wait=False)
    assert asking.asked.wait(WAIT) and wait_until(lambda: rig.assistant.pending_question == "Which one?")
    return reply.task


COMPUTER = route_json("computer", reply="On it.", goal="Open the file")


def test_a_new_request_is_not_taken_for_the_answer_to_a_pending_question(tmp_path, asking):
    rig = assistant_that_asks(tmp_path, [COMPUTER, route_json("answer", reply="Paris.")])
    task = start_asking_task(rig, asking)
    assert "Which one?" in rig.shown and rig.assistant.pending_question == "Which one?"

    reply = rig.assistant.handle_text("what is the capital of France")  # unrelated to the question

    assert reply.text == "Paris."  # handled as a request of its own
    assert task.wait(WAIT)
    assert len(asking.replies) == 1 and isinstance(asking.replies[0], Abort)  # the question was dropped, not answered
    assert "capital" not in str(asking.replies)  # the request's words never reached the task
    assert rig.assistant.pending_question is None and rig.assistant.answer_pending("Paris") is False


def test_a_pending_question_is_answered_only_through_answer_pending(tmp_path, asking):
    rig = assistant_that_asks(tmp_path, [COMPUTER])
    task = start_asking_task(rig, asking)
    assert rig.assistant.answer_pending("the red one") is True
    assert task.wait(WAIT)
    assert asking.replies == ["the red one"]  # exactly what was given, nothing else
    assert rig.assistant.pending_question is None and rig.assistant.answer_pending("again") is False


def test_a_stop_while_a_question_is_pending_ends_the_task_without_an_answer(tmp_path, asking):
    rig = assistant_that_asks(tmp_path, [COMPUTER])
    task = start_asking_task(rig, asking)
    assert rig.assistant.stop() is True
    assert task.wait(WAIT)
    assert len(asking.replies) == 1 and isinstance(asking.replies[0], Abort)


def test_the_question_is_spoken_and_a_voice_request_is_not_its_answer_either(tmp_path, asking):
    listener = Listener(final="what is the capital of France")
    stt = STT(Chain("stt", [Slot("stt", listener)]))
    listener.connection.release.set()
    rig = assistant_that_asks(tmp_path, [COMPUTER, route_json("answer", reply="Paris.")], stt=stt)
    task = start_asking_task(rig, asking)
    assert rig.tts.event_for("Which one?").wait(WAIT)

    reply = rig.assistant.handle_audio(iter(loud()))

    assert reply.text == "Paris." and task.wait(WAIT)
    assert len(asking.replies) == 1 and isinstance(asking.replies[0], Abort)


def test_a_correction_that_is_itself_a_task_waits_for_the_dropped_task_to_end_and_is_not_busy(tmp_path, asking, monkeypatch):
    """ "Open the file" ... "Which one?" ... "open Safari instead": the question is dropped and the new task is not busy."""
    instead = route_json("computer", reply="Switching.", goal="Open Safari")
    rig = assistant_that_asks(tmp_path, [COMPUTER, instead])
    first = start_asking_task(rig, asking)
    asking.release.clear()  # the stopped run needs a moment to end, as one that reads the screen once more does
    waited_on, wait = [], ComputerTask.wait
    monkeypatch.setattr(ComputerTask, "wait", lambda self, timeout=None: waited_on.append(self) or wait(self, timeout))

    done, box = in_thread(lambda: rig.assistant.handle_text("open Safari instead", wait=True))
    assert asking.unwinding.wait(WAIT)
    assert wait_until(lambda: first in waited_on)  # the new request is waiting for the task it dropped to end
    assert asking.goals == ["Open the file"] and not done.is_set()  # held back: the machine is not free yet
    asking.release.set()

    assert done.wait(WAIT)
    reply = box["value"]
    assert reply.route == "computer" and reply.error is None and reply.task is not first
    assert first.wait(WAIT) and isinstance(asking.replies[0], Abort) and len(asking.replies) == 1
    assert asking.goals == ["Open the file", "Open Safari"] and reply.task.result.outcome == "done"
    assert rig.warned == []  # not "a task is already running"


def test_by_default_a_task_never_asks(tmp_path, asking):
    rig = assistant_that_asks(tmp_path, [COMPUTER], clarify=False)
    reply = rig.assistant.handle_text("open the file", wait=False)
    assert reply.task.wait(WAIT)
    assert asking.ask_was == [None]


def test_the_question_wait_ends_when_the_task_is_stopped_before_it_is_asked(tmp_path):
    task = ComputerTask("g", act=False, config=object(), folder=tmp_path, on_question=lambda *_: None)
    task.stop()
    with pytest.raises(Abort):
        task._ask("Which one?")
    assert task.pending_question is None


def test_nothing_is_built_for_the_voice_without_a_player(tmp_path):
    rig = build(tmp_path, llm=FakeLLM(), speak=False)
    rig.assistant.interrupt_speech()
    rig.assistant.stop()
    assert rig.config.calls.tts == 0


def test_a_voice_that_was_cut_on_purpose_is_not_reported_but_a_voice_that_failed_is(tmp_path):
    rig = build(tmp_path, llm=FakeLLM())
    rig.assistant._speech_failed(ProviderError("voice call cancelled", kind="cancelled", provider="voice"))
    assert rig.warned == []
    rig.assistant._speech_failed(ProviderError("voice could not be reached", kind="transport", provider="voice"))
    assert rig.warned == ["speech failed: voice could not be reached"]

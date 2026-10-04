"""PointAssistant over the real Assistant and the assistant tests' fakes: a spoken question goes to the pin, never the router."""

from __future__ import annotations

import ast
import inspect
import threading
from pathlib import Path
from types import SimpleNamespace

from test_assistant_fakes import WAIT, FakeConfig, FakeLLM, FakePlayer, FakeSTT, FakeTTS, wait_until
from test_pet_core import tone
from test_point_session import Blocking, Rig

from glide.assistant.audio_io import chunked
from glide.assistant.core import IO, Assistant
from glide.assistant.point_session import PointSession
from glide.assistant.point_voice import PointAssistant


def build(stt_text="what does this mean", **options):
    llm, tts, player, stt = FakeLLM(), FakeTTS(), FakePlayer(), FakeSTT(final=stt_text)
    config = FakeConfig(llm=llm, stt=stt, tts=tts)
    heard = []
    assistant = PointAssistant(config, io=IO(player=player, heard=heard.append), **options)
    session = SimpleNamespace(asked=[], stopped=0)
    session.ask = lambda text: session.asked.append(text) or True
    session.stop = lambda: setattr(session, "stopped", session.stopped + 1)
    return SimpleNamespace(
        assistant=assistant, config=config, stt=stt, llm=llm, tts=tts, player=player, session=session, heard=heard
    )


def say_something(rig, **options):
    return rig.assistant.handle_audio(chunked(tone(2000, 0.5)), wait=False, **options)


def test_a_spoken_question_goes_to_the_session_and_never_to_the_router_or_a_task():
    rig = build()
    rig.assistant.bind(rig.session)
    reply = say_something(rig)
    assert rig.session.asked == ["what does this mean"] and reply.route == "answer" and rig.heard == ["what does this mean"]
    assert rig.llm.chat_calls == [] and rig.llm.stream_calls == [] and rig.assistant.task is None
    assert rig.config.calls.llm == 0 and rig.config.calls.classifier == 0


def test_a_stop_phrase_ends_the_answer_and_its_readout_and_asks_nothing():
    rig = build("stop")
    rig.assistant.bind(rig.session)
    assert say_something(rig).route == "stop"
    assert rig.session.stopped == 1 and rig.session.asked == [] and rig.player.cancels >= 1


def test_a_question_before_the_session_is_bound_is_dropped_not_queued():
    rig = build()
    assert say_something(rig).route == "none"
    rig.assistant.bind(rig.session)
    assert rig.session.asked == []


def test_speech_that_a_barge_in_overtook_while_it_was_heard_is_dropped():
    rig = build()
    rig.assistant.bind(rig.session)

    def overtaken():
        yield from chunked(tone(2000, 0.2))
        rig.assistant.interrupt_speech()  # the person barged in before this question was fully heard

    assert rig.assistant.handle_audio(overtaken(), wait=False).route == "none"
    assert rig.session.asked == []


def test_an_answer_is_read_aloud_a_sentence_at_a_time_and_cut_voice_keeps_a_question_being_heard():
    rig = build()
    rig.assistant.bind(rig.session)
    assert rig.assistant.say_aloud("This is error 0007. Check the connection.") is True
    assert wait_until(lambda: rig.tts.calls and rig.assistant.wait_idle(WAIT))
    assert [text for text, _ in rig.tts.calls] == ["This is error 0007.", "Check the connection."]
    cancels = rig.player.cancels

    def cut_while_heard():
        yield from chunked(tone(2000, 0.2))
        rig.assistant.cut_voice()  # the voice is cut while the person is still asking

    assert rig.assistant.handle_audio(cut_while_heard(), wait=False).route == "answer"
    assert rig.player.cancels > cancels  # the readout was cut ...
    assert rig.session.asked == ["what does this mean"]  # ... and the question being heard survived it


def test_without_a_player_saying_an_answer_is_a_no_op_and_builds_no_tts():
    config = FakeConfig(llm=FakeLLM(), tts=FakeTTS())
    assistant = PointAssistant(config, io=IO())
    assert assistant.say_aloud("Hello there.") is False
    assistant.cut_voice()
    assert config.calls.tts == 0


def test_a_stop_phrase_from_the_speech_settings_ends_the_answer_like_the_built_in_ones():
    rig = build("hold on please", extra_stop_phrases=("Hold on, please!",))
    rig.assistant.bind(rig.session)
    assert say_something(rig).route == "stop"
    assert rig.session.stopped == 1 and rig.session.asked == [] and rig.player.cancels >= 1
    plain = build("hold on please")  # without the configured phrase the same words are a question about the pin
    plain.assistant.bind(plain.session)
    assert say_something(plain).route == "answer"
    assert plain.session.stopped == 0 and plain.session.asked == ["hold on please"]


def test_a_stop_is_for_the_question_being_heard_before_it_and_not_for_the_one_asked_after():
    rig = build()
    rig.assistant.bind(rig.session)
    heard, release = threading.Event(), threading.Event()

    def slow_question():
        yield from chunked(tone(2000, 0.2))
        heard.set()
        assert release.wait(WAIT)

    first = []
    worker = threading.Thread(target=lambda: first.append(rig.assistant.handle_audio(slow_question(), wait=False)))
    worker.start()
    assert heard.wait(WAIT)
    rig.stt.final = "stop"  # the next utterance is the stop word
    assert say_something(rig).route == "stop"
    release.set()
    worker.join(WAIT)
    assert first[0].route == "none" and rig.session.asked == []  # the question made before the stop is never asked
    rig.stt.final = "and the second one"
    assert say_something(rig).route == "answer" and rig.session.asked == ["and the second one"]


def test_a_stop_that_overtakes_the_question_while_the_session_takes_it_ends_what_it_started():
    rig = build()
    rig.assistant.bind(rig.session)

    def ask_then_stop_arrives(text):
        rig.session.asked.append(text)
        rig.assistant.stop()  # a stop came for this question while it was being handed over
        return True

    rig.session.ask = ask_then_stop_arrives
    assert say_something(rig).route == "none"
    assert rig.session.stopped == 1  # the answer that was just started is ended again, never shown or spoken


def test_an_answer_that_arrives_after_a_spoken_stop_is_neither_shown_nor_read_aloud():
    blocking = Blocking()
    pin = Rig(blocking)
    rig = build()
    events = []
    session = PointSession(
        pin.selection,
        blocking,
        lambda kind, **data: events.append(kind),
        speak=rig.assistant.say_aloud,
        cancel_speech=rig.assistant.cut_voice,
        clock=pin.clock,
    )
    rig.assistant.bind(session)
    try:
        assert say_something(rig).route == "answer"
        assert blocking.entered.wait(WAIT)
        rig.stt.final = "stop"
        assert say_something(rig).route == "stop"
        blocking.release.set()  # the provider answers, late
        wait_until(lambda: False, timeout=0.3)
        assert "answer" not in events and "stopped" in events
        assert rig.tts.calls == [] and rig.player.played == []
    finally:
        session.close()
        pin.session.close()


def test_point_voice_uses_only_the_public_surface_of_the_assistant():
    """What a change to core.py must keep: the options and methods point voice calls, and nothing private of the base."""
    assert {"responder", "on_stop", "extra_stop_phrases"} <= set(inspect.signature(Assistant).parameters)
    assert list(inspect.signature(Assistant.say_aloud).parameters) == ["self", "text", "language"]
    assert inspect.signature(Assistant.say_aloud).parameters["language"].kind is inspect.Parameter.KEYWORD_ONLY
    assert list(inspect.signature(Assistant.cut_voice).parameters) == ["self"]
    own = {name for name in vars(PointAssistant) if name.startswith("_")}
    tree = ast.parse((Path(inspect.getsourcefile(PointAssistant))).read_text())
    private = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr.startswith("_") and not node.attr.startswith("__")
    }
    own_state = {"_session", "_bound"}
    assert private - own - own_state == set(), "point_voice.py reaches into a private member of Assistant"

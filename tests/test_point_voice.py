"""PointAssistant over the real Assistant and the assistant tests' fakes: a spoken question goes to the pin, never the router."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

from test_assistant_fakes import WAIT, FakeConfig, FakeLLM, FakePlayer, FakeSTT, FakeTTS, wait_until
from test_pet_core import tone

from glide.assistant.audio_io import chunked
from glide.assistant.core import IO, Assistant
from glide.assistant.point_voice import PointAssistant


def build(stt_text="what does this mean", **options):
    llm, tts, player = FakeLLM(), FakeTTS(), FakePlayer()
    config = FakeConfig(llm=llm, stt=FakeSTT(final=stt_text), tts=tts)
    heard = []
    assistant = PointAssistant(config, io=IO(player=player, heard=heard.append), **options)
    session = SimpleNamespace(asked=[], stopped=0)
    session.ask = lambda text: session.asked.append(text) or True
    session.stop = lambda: setattr(session, "stopped", session.stopped + 1)
    return SimpleNamespace(assistant=assistant, config=config, llm=llm, tts=tts, player=player, session=session, heard=heard)


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
    rig.assistant.say("This is error 0007. Check the connection.")
    assert wait_until(lambda: rig.tts.calls and rig.assistant.wait_idle(WAIT))
    assert [text for text, _ in rig.tts.calls] == ["This is error 0007.", "Check the connection."]
    epoch = rig.assistant._epoch
    rig.assistant.cut_voice()
    assert rig.assistant._epoch == epoch  # unlike interrupt_speech, a request still being heard survives
    assert rig.player.cancels >= 1


def test_without_a_player_saying_an_answer_is_a_no_op_and_builds_no_tts():
    config = FakeConfig(llm=FakeLLM(), tts=FakeTTS())
    assistant = PointAssistant(config, io=IO())
    assistant.say("Hello there.")
    assistant.cut_voice()
    assert config.calls.tts == 0


def test_the_four_assistant_members_it_relies_on_still_exist_with_the_shapes_it_calls():
    """If core.py changes one of these, this fails here, not in front of a person."""
    parameters = inspect.signature(Assistant._handle).parameters
    assert list(parameters) == ["self", "text", "act", "wait", "hint_language", "epoch"]
    assert list(inspect.signature(Assistant._begin).parameters) == ["self", "epoch"]
    assert list(inspect.signature(Assistant._silence).parameters) == ["self"]
    assert list(inspect.signature(Assistant._speaker_or_none).parameters) == ["self"]

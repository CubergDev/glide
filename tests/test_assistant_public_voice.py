"""The public ways a front end uses the voice and the request path without the router: `say_aloud`, `cut_voice`, `responder`.

Point-to-ask (glide/assistant/point_voice.py) is the first user. Each test is over the assistant's fakes, offline.
"""

from __future__ import annotations

from test_assistant_core import build, spoken
from test_assistant_fakes import WAIT, wait_until

from glide.assistant.core import Assistant, Reply


def test_say_aloud_queues_one_sentence_at_a_time_in_order_and_says_whether_it_queued_anything(tmp_path):
    rig = build(tmp_path)
    assert rig.assistant.say_aloud("This is error 0007. Check the connection.") is True
    assert wait_until(lambda: len(spoken(rig)) == 2 and rig.assistant.wait_idle(WAIT))
    assert spoken(rig) == ["This is error 0007.", "Check the connection."]
    assert rig.assistant.say_aloud("   ") is False and rig.assistant.say_aloud("") is False


def test_say_aloud_passes_a_given_language_to_every_sentence_and_detects_one_otherwise(tmp_path):
    rig = build(tmp_path)
    rig.assistant.say_aloud("Hello there. How are you.", language="fr")
    assert wait_until(lambda: len(rig.tts.calls) == 2)
    assert [language for _, language in rig.tts.calls] == ["fr", "fr"]


def test_say_aloud_without_a_player_does_nothing_and_builds_no_tts(tmp_path):
    rig = build(tmp_path, speak=False)
    assert rig.assistant.say_aloud("Hello there.") is False
    rig.assistant.cut_voice()
    assert rig.config.calls.tts == 0 and rig.tts.calls == []


def test_cut_voice_silences_the_readout_and_nothing_else(tmp_path):
    rig = build(tmp_path)
    rig.assistant.say_aloud("One sentence.")
    assert wait_until(lambda: rig.tts.calls)
    cancels = rig.player.cancels
    rig.assistant.cut_voice()
    assert rig.player.cancels > cancels
    assert rig.assistant.say_aloud("Said after the cut.") is True  # the voice is usable at once; nothing was stopped
    assert wait_until(lambda: "Said after the cut." in spoken(rig))
    reply = rig.assistant.handle_text("what time is it", wait=False)  # no stop or barge-in was recorded: a request is answered
    assert reply.route == "answer" and reply.text


def test_cut_voice_does_not_cancel_a_request_that_is_being_answered(tmp_path):
    rig = build(tmp_path)
    seen = []

    def responder(text, language):
        rig.assistant.cut_voice()  # called from inside the request, as an answer's superseding does
        seen.append(text)
        return Reply("answer")

    assistant = Assistant(rig.config, io=rig.assistant.io, runs_dir=tmp_path / "runs", responder=responder)
    assert assistant.handle_text("what is this").route == "answer" and seen == ["what is this"]


def make(tmp_path, **options):
    rig = build(tmp_path)
    calls = {"asked": [], "stopped": 0}

    def responder(text, language):
        calls["asked"].append((text, language))
        return Reply("answer", language=language)

    def on_stop():
        calls["stopped"] += 1

    rig.assistant = Assistant(
        rig.config, io=rig.assistant.io, runs_dir=tmp_path / "runs", responder=responder, on_stop=on_stop, **options
    )
    return rig, calls


def test_a_responder_gets_the_request_and_the_router_and_the_model_are_never_called(tmp_path):
    rig, calls = make(tmp_path)
    reply = rig.assistant.handle_text("what does this mean", hint_language="en")
    assert reply.route == "answer" and calls["asked"] == [("what does this mean", "en")] and "total_s" in reply.timings
    assert rig.llm.chat_calls == [] and rig.llm.stream_calls == [] and rig.assistant.task is None
    assert rig.config.calls.llm == 0


def test_a_stop_phrase_never_reaches_the_responder_and_calls_on_stop(tmp_path):
    rig, calls = make(tmp_path, extra_stop_phrases=("hold on please",))
    assert rig.assistant.handle_text("stop").route == "stop"
    assert rig.assistant.handle_text("Hold on, please!").route == "stop"  # the configured phrase, normalised like an utterance
    assert calls["asked"] == [] and calls["stopped"] == 2


def test_a_request_that_a_stop_overtakes_while_the_responder_has_it_is_dropped_and_on_stop_ends_its_work(tmp_path):
    rig = build(tmp_path)
    stopped = []

    def responder(text, language):
        rig.assistant.stop()  # a stop for this request arrives while it is being handed over
        return Reply("answer")

    rig.assistant = Assistant(
        rig.config, io=rig.assistant.io, runs_dir=tmp_path / "runs", responder=responder, on_stop=lambda: stopped.append(1)
    )
    assert rig.assistant.handle_text("what is this").route == "none" and stopped == [1]


def test_a_request_made_after_a_stop_reaches_the_responder(tmp_path):
    rig, calls = make(tmp_path)
    rig.assistant.stop()
    assert rig.assistant.handle_text("and the second one").route == "answer"
    assert calls["asked"] == [("and the second one", None)]

"""Clarify has a channel: ask(question) -> reply or None. With no reply nothing is done and the user is told what is needed."""

from __future__ import annotations

import pytest
from routing_fakes import FakeClassifier, FakeLLM, Pick, everything_shown, fast_json

from glide.routing import Context, Router, RoutingSettings, resolve
from glide.routing.decision import WHY_CLARIFY_BUDGET, WHY_CLARIFY_STOP, WHY_CLARIFY_UNANSWERED


class Channel:
    """A front end's `Clarifier`: answers from a script, records the questions."""

    def __init__(self, *replies):
        self.replies, self.questions = list(replies), []

    def ask(self, question):
        self.questions.append(question)
        return self.replies.pop(0) if self.replies else None


def asks_then_acts(state):
    """Unsure until the user has answered a question: then it is an instruction like any other."""
    answered = any(t["role"] == "assistant" for t in state["conversation"])
    return Pick("execute", 0.95) if answered else Pick("clarify", 0.9)


def make(classifier, llm=None, **settings):
    return Router(classifier, llm, settings=RoutingSettings(**settings))


def test_a_clear_request_never_asks():
    channel = Channel("never used")
    result = resolve(make(FakeClassifier(Pick("answer", 0.95))), "hello", None, channel)
    assert (result.decision.route, result.asked, result.said, result.unresolved) == ("answer", 0, "", False)
    assert channel.questions == []


def test_no_channel_means_no_action_and_a_sentence_that_says_what_is_needed():
    result = resolve(make(FakeClassifier(Pick("clarify", 0.9))), "do the thing", None, None)
    assert result.unresolved and not result.decision.acts and result.asked == 0
    assert result.decision.why_code == WHY_CLARIFY_UNANSWERED and result.decision.tier == "router"
    assert result.said.startswith("I did not do anything") and "Could you say a little more?" in result.said


def test_the_sentence_is_in_the_users_language():
    classifier = FakeClassifier(Pick("clarify", 0.9))
    assert "我乜都未做" in resolve(make(classifier), "做嗰樣嘢", None, None, language="yue").said
    assert "我什么都没有做" in resolve(make(classifier), "做那件事", Context(language="zh"), None).said


def test_a_channel_that_returns_none_is_no_action_and_the_question_is_said():
    channel = Channel()  # no reply: a timeout, a stop, or a new request arrived instead
    result = resolve(make(FakeClassifier(Pick("clarify", 0.9))), "do the thing", None, channel)
    assert channel.questions != [] and result.asked == 1 and result.unresolved
    assert result.decision.why_code == WHY_CLARIFY_UNANSWERED and channel.questions[0] in result.said


@pytest.mark.parametrize("blank", ["", "   ", None])
def test_a_blank_reply_is_no_reply(blank):
    result = resolve(make(FakeClassifier(Pick("clarify", 0.9))), "do the thing", None, Channel(blank))
    assert result.unresolved and result.decision.why_code == WHY_CLARIFY_UNANSWERED


def test_the_fast_models_question_is_the_one_asked():
    llm = FakeLLM(fast_json("clarify", "high", question="Which contact do you mean?"))
    channel = Channel()
    resolve(make(FakeClassifier(Pick("clarify", 0.4)), llm), "tell him I am late", None, channel)
    assert channel.questions == ["Which contact do you mean?"]


def test_an_answer_is_added_to_the_conversation_and_the_request_is_routed_again():
    classifier = FakeClassifier(asks_then_acts)
    channel = Channel("Kelvin Chan")
    result = resolve(make(classifier), "tell him I am late", None, channel)
    assert result.decision.route == "execute" and result.asked == 1 and not result.unresolved
    second = classifier.calls[1]["state"]["conversation"]
    assert [t["role"] for t in second] == ["assistant", "user"] and second[1]["text"] == "Kelvin Chan"
    assert "tell him I am late" in result.decision.goal and "Clarifications:" in result.decision.goal
    assert "User: Kelvin Chan" in result.decision.goal


def test_questions_are_limited_by_the_budget_and_the_user_is_told():
    classifier = FakeClassifier(Pick("clarify", 0.9))  # never satisfied
    channel = Channel("a thing", "another thing")
    result = resolve(make(classifier, max_clarifications=1), "do it", None, channel)
    assert len(channel.questions) == 1 and result.unresolved and result.decision.why_code == WHY_CLARIFY_BUDGET
    assert result.said
    zero = resolve(make(classifier, max_clarifications=0), "do it", None, Channel("x"))
    assert zero.asked == 0 and zero.decision.why_code == WHY_CLARIFY_BUDGET


def test_a_reply_that_is_a_stop_is_a_stop_not_an_answer():
    result = resolve(make(FakeClassifier(Pick("clarify", 0.9))), "do the thing", None, Channel("never mind"))
    assert result.decision.route == "stop" and result.decision.why_code == WHY_CLARIFY_STOP and not result.unresolved


def test_a_configured_stop_phrase_stops_a_clarification_too():
    router = Router(FakeClassifier(Pick("clarify", 0.9)), settings=RoutingSettings(), stop_phrases=["hold on please"])
    assert resolve(router, "do the thing", None, Channel("hold on please")).decision.route == "stop"


def test_the_on_question_hook_sees_each_question_before_it_is_asked():
    seen = []
    resolve(make(FakeClassifier(Pick("clarify", 0.9))), "x y", None, Channel("a"), on_question=seen.append)
    assert len(seen) == 1


def test_a_cancelled_reroute_is_not_turned_into_an_action():
    calls = {"n": 0}

    def script(state):
        calls["n"] += 1
        from routing_fakes import failing

        return Pick("clarify", 0.9) if calls["n"] == 1 else failing("cancelled")

    result = resolve(make(FakeClassifier(script)), "do it", None, Channel("Kelvin"))
    assert result.decision.cancelled and not result.decision.acts


def test_a_clarifying_reply_does_not_carry_text_from_the_screen():
    canary = "CANARY-77"
    from glide.routing import Span

    ctx = Context(data=(Span(canary, "page"),))
    classifier = FakeClassifier(asks_then_acts)
    resolve(make(classifier), "do the thing", ctx, Channel("Kelvin"))
    assert canary not in everything_shown(classifier)

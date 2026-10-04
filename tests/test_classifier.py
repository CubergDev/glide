"""The classifier adapters on their own: the LLM-backed one, the TypeSafe-protocol one, and the chain.

No network and no key from the environment. The LLM is a script of replies, the TypeSafe server is an
`httpx2.MockTransport` (the SDK speaks httpx2, not httpx), and the chain runs on fakes. The loop itself
is exercised in test_classifier_scenarios.py.
"""

from __future__ import annotations

import json
import logging
import threading
import traceback
from types import SimpleNamespace

import httpx2
import pytest
from typesafe_sdk import Choice, ChoiceAnswer, Noul, NoulAnswer, Score

from glide.computer.calls import Calls, MeteredClassifier
from glide.providers.base import ChatResult, ProviderSpec, Usage
from glide.providers.chain import ChainPolicy
from glide.providers.classifier import (
    ChainedClassifier,
    ClassifierReply,
    LLMClassifier,
    TypeSafeClassifier,
    build_client,
    typesafe_error,
)
from glide.providers.errors import AllProvidersFailed, ProviderError

KEY = "sk-test-0123456789abcdef"
SPEC = ProviderSpec("jev", "typesafe", "https://jev.example.test", "JEV_API_KEY")


# ---------------------------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------------------------


class ScriptedLLM:
    """Answers `chat` from a script: text to return, (text, finish_reason), or an exception to raise.

    The last step repeats once the script runs out. Every call is kept for the test to read.
    """

    name = "scripted"
    model = "m-1"

    def __init__(self, *script):
        self.script = list(script)
        self.calls: list[dict] = []
        self.closed = False

    def chat(self, messages, *, max_tokens=512, temperature=0.0, schema=None, logprobs=False, timeout=None):
        self.calls.append(
            {
                "messages": list(messages),
                "max_tokens": max_tokens,
                "temperature": temperature,
                "schema": schema,
                "timeout": timeout,
            }
        )
        step = self.script[min(len(self.calls), len(self.script)) - 1]
        if isinstance(step, BaseException):
            raise step
        text, finish = step if isinstance(step, tuple) else (step, "stop")
        return ChatResult(
            text=text, usage=Usage(100, 10), provider="scripted:m-1", model="m-1", latency_s=0.0, finish_reason=finish
        )

    def close(self):
        self.closed = True


def top(*pairs: tuple[str, float]) -> dict:
    return {"top": [{"option": option, "p": p} for option, p in pairs]}


def reply(kind=None, ok=0.8, **extra) -> str:
    kind = kind or top(("b", 0.6), ("a", 0.3), ("c", 0.1))
    return json.dumps({"kind": kind, "ok": ok, **extra})


def questions() -> dict:
    return {
        "kind": Choice(instructions="Which move?", criteria={"a": "do a", "b": "do b", "c": None, "d": "split\nover lines"}),
        "ok": Noul(instructions="Did it work?", criteria={"true": "yes it did", "false": "no it did not"}),
    }


STATE = {"goal": "buy a ticket", "items": [{"i": 0, "text": "Home"}]}


def classify(llm, **kw):
    return LLMClassifier(llm).system_one(state=STATE, questions=questions(), **kw)


# ---------------------------------------------------------------------------------------------
# LLMClassifier: what it asks
# ---------------------------------------------------------------------------------------------


def test_one_call_answers_every_question_with_the_sdks_own_answer_types():
    llm = ScriptedLLM(reply())
    result = classify(llm)

    assert len(llm.calls) == 1
    assert isinstance(result, ClassifierReply)
    assert isinstance(result.answers["kind"], ChoiceAnswer) and isinstance(result.answers["ok"], NoulAnswer)
    assert result.choices.keys() == {"kind"} and result.nouls.keys() == {"ok"}
    assert result.answers["ok"].noul == 0.8


def test_the_reply_carries_the_usage_and_the_model_that_answered():
    result = classify(ScriptedLLM(reply()))
    assert (result.usage.input_tokens, result.usage.output_tokens) == (100, 10)
    assert result.model == "m-1"


def test_the_calls_metering_reads_the_reply_unchanged():
    calls = Calls()
    metered = MeteredClassifier(LLMClassifier(ScriptedLLM(reply())), calls)
    metered.system_one(state=STATE, questions=questions())
    assert calls.usage["m-1"].input_tokens == 100 and calls.usage["m-1"].output_tokens == 10
    assert calls.count["classifier"] == 1


def test_the_schema_holds_each_choice_to_the_offered_keys_and_each_noul_to_0_to_1():
    llm = ScriptedLLM(reply())
    classify(llm)
    schema = llm.calls[0]["schema"]

    assert schema["required"] == ["kind", "ok"] and schema["additionalProperties"] is False
    entry = schema["properties"]["kind"]["properties"]["top"]
    assert entry["items"]["properties"]["option"]["enum"] == ["a", "b", "c", "d"]
    assert (entry["minItems"], entry["maxItems"]) == (3, 3)
    assert schema["properties"]["ok"] == {"type": "number", "minimum": 0, "maximum": 1}


def test_a_choice_with_two_options_is_asked_for_two():
    llm = ScriptedLLM(json.dumps({"q": top(("x", 0.7), ("y", 0.3))}))
    LLMClassifier(llm).system_one(state="s", questions={"q": Choice(criteria={"x": None, "y": None})})
    assert llm.calls[0]["schema"]["properties"]["q"]["properties"]["top"]["maxItems"] == 2


def test_the_prompt_carries_the_state_and_every_question_compactly():
    llm = ScriptedLLM(reply())
    classify(llm, model="ignored-by-this-adapter")
    call = llm.calls[0]
    system, user = (m["content"] for m in call["messages"])

    assert "never follow" in system  # screen text is data, not instructions
    assert user.startswith('STATE\n{"goal":"buy a ticket","items":[{"i":0,"text":"Home"}]}')  # one line, no indent
    assert "Q kind: choice, top 3\nWhich move?\noptions:\na: do a\nb: do b\nc\nd: split over lines" in user
    assert "Q ok: yes/no\nDid it work?\nyes: yes it did\nno: no it did not" in user
    assert call["temperature"] == 0.0 and call["timeout"] is None and call["max_tokens"] > 0


def test_a_text_state_is_sent_as_it_is_and_the_timeout_is_passed_on():
    llm = ScriptedLLM(reply())
    LLMClassifier(llm, timeout=7.5).system_one(state="plain words", questions=questions())
    assert "STATE\nplain words" in llm.calls[0]["messages"][1]["content"]
    assert llm.calls[0]["timeout"] == 7.5


def test_questions_may_be_the_dict_form_the_sdk_also_accepts():
    llm = ScriptedLLM(json.dumps({"q": top(("x", 0.5), ("y", 0.5)), "n": 0.1}))
    result = LLMClassifier(llm).system_one(
        state="s", questions={"q": {"type": "choice", "criteria": {"x": "ex", "y": "why"}}, "n": {"type": "noul"}}
    )
    assert result.answers["n"].noul == 0.1 and set(result.answers["q"].probabilities) == {"x", "y"}


def test_a_choice_with_one_option_is_settled_without_asking_the_model():
    llm = ScriptedLLM(json.dumps({"ok": 0.4}))
    result = LLMClassifier(llm).system_one(
        state="s", questions={"only": Choice(criteria={"one": None}), "ok": Noul(instructions="fine?")}
    )
    assert result.answers["only"].choice == "one" and result.answers["only"].confidence == 1.0
    assert "only" not in llm.calls[0]["schema"]["properties"]

    alone = ScriptedLLM()
    result = LLMClassifier(alone).system_one(state="s", questions={"only": Choice(criteria={"one": None})})
    assert alone.calls == [] and result.answers["only"].choice == "one" and result.usage.input_tokens == 0


@pytest.mark.parametrize(
    ("questions_", "kind"),
    [
        ({}, "bad_request"),
        ({"q": Choice(criteria={})}, "bad_request"),
        ({"q": "what?"}, "bad_request"),
        ({"q": Score(criteria=["bad", "good"])}, "unsupported"),  # a chain falls through to a slot that answers scores
        ({"q": {"type": "score", "criteria": ["a"]}}, "unsupported"),
    ],
)
def test_a_question_it_cannot_ask_is_refused_before_any_call(questions_, kind):
    llm = ScriptedLLM(reply())
    with pytest.raises(ProviderError) as caught:
        LLMClassifier(llm).system_one(state="s", questions=questions_)
    assert caught.value.kind == kind and llm.calls == []


# ---------------------------------------------------------------------------------------------
# LLMClassifier: confidence
# ---------------------------------------------------------------------------------------------


def test_confidence_is_the_top_probability_and_the_distribution_names_every_option():
    answer = classify(ScriptedLLM(reply())).answers["kind"]

    assert answer.choice == "b" and answer.confidence == pytest.approx(0.6)
    assert answer.probabilities == pytest.approx({"a": 0.3, "b": 0.6, "c": 0.1, "d": 0.0})
    assert sum(answer.probabilities.values()) == pytest.approx(1.0)


def test_the_choice_is_the_most_probable_option_even_when_the_model_listed_another_first():
    answer = classify(ScriptedLLM(reply(top(("a", 0.2), ("c", 0.7), ("b", 0.1))))).answers["kind"]
    assert answer.choice == "c" and answer.confidence == pytest.approx(0.7)


def test_a_split_distribution_gives_a_confidence_under_the_floor_so_the_loop_would_stop():
    answer = classify(ScriptedLLM(reply(top(("a", 0.35), ("b", 0.33), ("c", 0.32))))).answers["kind"]
    assert answer.choice == "a" and answer.confidence < 0.4  # the runner's min_confidence


def test_the_lowest_confidence_the_scheme_expresses_is_a_third():
    answer = classify(ScriptedLLM(reply(top(("a", 1), ("b", 1), ("c", 1))))).answers["kind"]  # equal weights
    assert answer.confidence == pytest.approx(1 / 3) and answer.choice == "a"  # a tie goes to the one listed first


def test_a_list_that_sums_to_more_than_one_is_scaled_down_to_sum_to_one():
    answer = classify(ScriptedLLM(reply(top(("a", 0.9), ("b", 0.9), ("c", 0.2))))).answers["kind"]
    assert answer.confidence == pytest.approx(0.45) and sum(answer.probabilities.values()) == pytest.approx(1.0)


def test_a_list_that_sums_to_less_than_one_is_never_scaled_up_so_held_back_doubt_stays_doubt():
    """`[0.3, 0.05, 0.05]` among many options is a model 30% sure. Dividing by the sum would make it 75% sure
    and the loop would click, which is the failure the confidence design exists to prevent."""
    answer = classify(ScriptedLLM(reply(top(("a", 0.3), ("b", 0.05), ("c", 0.05))))).answers["kind"]
    assert answer.confidence == pytest.approx(0.3) and answer.confidence < 0.4
    assert answer.probabilities == pytest.approx({"a": 0.3, "b": 0.05, "c": 0.05, "d": 0.0})  # the rest is unassigned

    answer = classify(ScriptedLLM(reply(top(("a", 0.5), ("b", 0.25), ("c", 0.0))))).answers["kind"]
    assert answer.confidence == pytest.approx(0.5)


def test_a_model_that_is_sure_is_believed():
    answer = classify(ScriptedLLM(reply(top(("a", 1.0), ("b", 0.0), ("c", 0.0))))).answers["kind"]
    assert answer.confidence == 1.0 and answer.probabilities["a"] == 1.0


# ---------------------------------------------------------------------------------------------
# LLMClassifier: validation and the one retry
# ---------------------------------------------------------------------------------------------

INVALID = {
    "unknown key": reply(top(("zzz", 0.6), ("a", 0.3), ("c", 0.1))),
    "missing question": json.dumps({"kind": top(("a", 0.6), ("b", 0.3), ("c", 0.1))}),
    "a lone option": reply(top(("a", 0.3))),  # would normalise up to 1.0 and switch the 0.4 floor off
    "too many options": reply(top(("a", 0.4), ("b", 0.3), ("c", 0.2), ("d", 0.1))),
    "a repeated option": reply(top(("a", 0.5), ("a", 0.3), ("c", 0.2))),
    "negative p": reply(top(("a", 0.9), ("b", 0.3), ("c", -0.2))),
    "nan p": '{"kind": {"top": [{"option": "a", "p": NaN}, {"option": "b", "p": 0.3}, {"option": "c", "p": 0.1}]}, "ok": 0.5}',
    "p as text": reply(top(("a", "0.6"), ("b", 0.3), ("c", 0.1))),
    "p as bool": reply(top(("a", True), ("b", 0.3), ("c", 0.1))),
    "all zero": reply(top(("a", 0), ("b", 0), ("c", 0))),
    "no top list": reply({"choice": "a"}),
    "entry not an object": reply({"top": ["a", "b", "c"]}),
    "noul as percent": reply(ok=80),
    "p as percentages": reply(top(("a", 30), ("b", 5), ("c", 5))),  # 30 would read as certain once scaled
    "p just over one": reply(top(("a", 1.5), ("b", 0.3), ("c", 0.1))),
    "noul as bool": reply(ok=True),
    "noul negative": reply(ok=-0.1),
    "not json": "I think it is a.",
    "a json array": "[1, 2, 3]",
    "empty": "",
}


@pytest.mark.parametrize("bad", INVALID.values(), ids=INVALID.keys())
def test_an_unusable_answer_is_retried_once_with_the_error_and_a_good_second_answer_is_used(bad):
    llm = ScriptedLLM(bad, reply())
    result = classify(llm)

    assert len(llm.calls) == 2
    retry = llm.calls[1]["messages"]
    assert retry[:2] == llm.calls[0]["messages"]  # the same question, then the bad reply and what was wrong with it
    assert retry[2]["role"] == "assistant" and retry[3]["role"] == "user" and "not usable" in retry[3]["content"]
    assert result.answers["kind"].choice == "b"
    assert (result.usage.input_tokens, result.usage.output_tokens) == (200, 20)  # both calls are paid for


@pytest.mark.parametrize("bad", INVALID.values(), ids=INVALID.keys())
def test_two_unusable_answers_raise_a_content_error_a_chain_can_fail_over_on(bad):
    llm = ScriptedLLM(bad)
    with pytest.raises(ProviderError) as caught:
        classify(llm)

    assert caught.value.kind == "content" and len(llm.calls) == 2
    assert caught.value.provider == "scripted:m-1"  # names the provider that answered, for the log line


def test_the_error_says_what_was_wrong_without_repeating_the_whole_reply():
    with pytest.raises(ProviderError) as caught:
        classify(ScriptedLLM(reply(top(("x" * 500, 0.6), ("a", 0.3), ("c", 0.1)))))
    assert "not one of the options" in str(caught.value) and len(str(caught.value)) < 300


def test_json_inside_a_code_fence_is_read_without_a_retry_and_a_sentence_round_it_is_not():
    for wrapped in (f"```json\n{reply()}\n```", f"  {reply()}\n"):
        llm = ScriptedLLM(wrapped)
        assert classify(llm).answers["kind"].choice == "b" and len(llm.calls) == 1
    # r2 seams 9: an object the screen text got into a sentence is not the answer; the one retry asks again
    llm = ScriptedLLM(f"Here you go: {reply()} Hope that helps.", reply())
    assert classify(llm).answers["kind"].choice == "b" and len(llm.calls) == 2


def test_an_index_the_model_wrote_as_a_number_is_the_key_it_stands_for():
    llm = ScriptedLLM(json.dumps({"item": top((3, 0.6), ("1", 0.3), (2, 0.1))}))
    result = LLMClassifier(llm).system_one(state="s", questions={"item": Choice(criteria={str(i): None for i in range(5)})})
    assert result.answers["item"].choice == "3" and result.answers["item"].probabilities["1"] == pytest.approx(0.3)


def test_a_reply_cut_off_by_the_token_limit_is_retried_with_more_room():
    llm = ScriptedLLM(('{"kind": {"top": [{"option": "a", "p"', "length"), reply())
    classify(llm)
    assert llm.calls[1]["max_tokens"] == 2 * llm.calls[0]["max_tokens"]


def test_a_provider_error_from_the_llm_is_not_retried_and_a_bug_is_not_hidden():
    limited = ScriptedLLM(ProviderError("slow down", kind="rate_limit", retry_after=3.0))
    with pytest.raises(ProviderError) as caught:
        classify(limited)
    assert caught.value.kind == "rate_limit" and caught.value.retry_after == 3.0 and len(limited.calls) == 1

    with pytest.raises(KeyError):
        classify(ScriptedLLM(KeyError("bug")))


# ---------------------------------------------------------------------------------------------
# LLMClassifier: lifecycle
# ---------------------------------------------------------------------------------------------


def test_as_a_context_manager_it_leaves_a_shared_llm_open_unless_told_to_close_it():
    shared = ScriptedLLM(reply())
    with LLMClassifier(shared) as classifier:
        classifier.system_one(state=STATE, questions=questions())
    assert shared.closed is False

    owned = ScriptedLLM(reply())
    with LLMClassifier(owned, close_llm=True):
        pass
    assert owned.closed is True


def test_it_holds_no_state_between_calls_so_two_can_run_at_once():
    llm = ScriptedLLM(reply())
    classifier = LLMClassifier(llm)
    results = []
    threads = [
        threading.Thread(target=lambda: results.append(classifier.system_one(state=STATE, questions=questions())))
        for _ in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(results) == 8 and all(r.answers["kind"].choice == "b" for r in results)


# ---------------------------------------------------------------------------------------------
# The TypeSafe-protocol client
# ---------------------------------------------------------------------------------------------

WIRE = {
    "model": "jev-latest",
    "usage": {"input_tokens": 41, "output_tokens": 3},
    "answers": {
        "kind": {"type": "choice", "choice": "b", "confidence": 0.8, "probabilities": {"a": 0.2, "b": 0.8}},
        "ok": {"type": "noul", "noul": 0.9},
    },
}


class Server:
    """An `httpx2.MockTransport` that records every request and answers from `respond(request)`."""

    def __init__(self, respond=None):
        self.requests: list[httpx2.Request] = []
        self.respond = respond or (lambda request: httpx2.Response(200, json=WIRE))
        self.transport = httpx2.MockTransport(self.handle)

    def handle(self, request):
        self.requests.append(request)
        result = self.respond(request)
        if isinstance(result, BaseException):
            raise result
        return result


def typesafe(server: Server, spec: ProviderSpec = SPEC, model: str = "jev-latest", **options) -> TypeSafeClassifier:
    return build_client(spec, model, KEY, options, transport=server.transport)


def ask(client, **kw):
    return client.system_one(state=STATE, questions=questions(), **kw)


def test_the_typesafe_client_posts_to_the_specs_server_and_returns_the_sdks_answers():
    server = Server()
    client = typesafe(server)
    reply_ = ask(client)

    (request,) = server.requests
    assert request.method == "POST" and str(request.url) == "https://jev.example.test/v1/systemone"
    body = json.loads(request.content)
    assert body["model"] == "jev-latest" and body["state"] == STATE and set(body["questions"]) == {"kind", "ok"}
    assert isinstance(reply_.answers["kind"], ChoiceAnswer) and reply_.answers["kind"].confidence == 0.8
    assert reply_.answers["ok"].noul == 0.9 and reply_.model == "jev-latest" and reply_.usage.input_tokens == 41
    assert client.name == "jev:jev-latest" and client.model == "jev-latest"


def test_any_systemone_server_is_the_same_client_with_another_base_url():
    laya = ProviderSpec("laya", "typesafe", "https://laya.example.test/", "LAYA_API_KEY")
    server = Server()
    client = typesafe(server, laya, "laya-fast")
    ask(client)
    assert str(server.requests[0].url) == "https://laya.example.test/v1/systemone"
    assert json.loads(server.requests[0].content)["model"] == "laya-fast" and client.name == "laya:laya-fast"


def test_a_per_call_model_overrides_the_slots_and_other_options_the_sdk_lacks_are_dropped():
    server = Server()
    client = typesafe(server)
    ask(client, model="jev-other", something_unknown=1)
    assert json.loads(server.requests[0].content)["model"] == "jev-other"


def test_the_default_server_is_typesafes_when_the_spec_names_none():
    server = Server()
    typesafe(server, ProviderSpec("jev", "typesafe", "", "JEV_API_KEY")).system_one(state=STATE, questions=questions())
    assert server.requests[0].url.host == "api.typesafe.ai"


def test_nothing_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "env-key-must-not-be-used")
    monkeypatch.setenv("TYPESAFE_BASE_URL", "https://env.example.test")
    monkeypatch.setenv("TYPESAFE_DEFAULT_MODEL", "env-model")
    server = Server()
    ask(typesafe(server))
    request = server.requests[0]
    assert request.url.host == "jev.example.test" and request.headers["authorization"] == f"Bearer {KEY}"
    assert json.loads(request.content)["model"] == "jev-latest"

    for missing in ("", "   ", None):  # a missing key is an auth failure, never the env var's key
        with pytest.raises(ProviderError) as caught:
            build_client(SPEC, "jev-latest", missing, transport=server.transport)
        assert caught.value.kind == "auth" and "env-key" not in str(caught.value)
    assert len(server.requests) == 1


def test_only_a_typesafe_spec_and_sane_options_are_accepted():
    with pytest.raises(ValueError, match="openai_compat"):
        build_client(ProviderSpec("x", "openai_compat"), "m", KEY)
    for bad in ({"max_retries": -1}, {"max_retries": "2"}, {"timeout": "fast"}, {"headers": "nope"}):
        with pytest.raises(ValueError):
            build_client(SPEC, "m", KEY, bad)


def test_options_from_the_spec_and_the_caller_are_merged():
    server = Server(lambda request: httpx2.Response(500, json={"error": "down"}))
    spec = ProviderSpec("jev", "typesafe", "https://jev.example.test", "K", options={"max_retries": 0})
    with pytest.raises(ProviderError):
        ask(build_client(spec, "m", KEY, {"headers": {"X-Team": "glide"}}, transport=server.transport))
    assert server.requests[0].headers["x-team"] == "glide"


@pytest.mark.parametrize(
    ("status", "kind"),
    [
        (400, "bad_request"),
        (401, "auth"),
        (403, "auth"),
        (404, "unsupported"),
        (422, "bad_request"),
        (429, "rate_limit"),
        (500, "server"),
        (502, "server"),
        (503, "server"),
        (504, "timeout"),
    ],
)
def test_each_http_failure_becomes_the_provider_error_kind_a_chain_acts_on(status, kind):
    server = Server(lambda request: httpx2.Response(status, json={"error": "nope"}))
    with pytest.raises(ProviderError) as caught:
        ask(typesafe(server))
    assert caught.value.kind == kind and caught.value.status == status and caught.value.provider == "jev:jev-latest"
    assert len(server.requests) == 1  # no retry inside the SDK: the chain is what tries the next provider


def test_a_rate_limit_carries_its_retry_after_in_seconds():
    server = Server(lambda request: httpx2.Response(429, headers={"retry-after": "7"}, json={"error": "slow down"}))
    with pytest.raises(ProviderError) as caught:
        ask(typesafe(server, max_retries=2))  # a rate limit is the chain's cooldown to honour, never the SDK's to sleep on
    assert caught.value.kind == "rate_limit" and caught.value.retry_after == 7.0 and len(server.requests) == 1


def test_a_reply_that_does_not_fit_the_schema_is_a_content_error():
    for response in (httpx2.Response(200, json={"model": "m"}), httpx2.Response(200, text="not json at all")):
        with pytest.raises(ProviderError) as caught:
            ask(typesafe(Server(lambda request, r=response: r)))
        assert caught.value.kind == "content"


def test_timeouts_and_dropped_connections_are_told_apart():
    for exc, kind in ((httpx2.ReadTimeout("slow"), "timeout"), (httpx2.ConnectError("refused"), "transport")):
        with pytest.raises(ProviderError) as caught:
            ask(typesafe(Server(lambda request, e=exc: e)))
        assert caught.value.kind == kind and "refused" not in str(caught.value)


def test_retries_are_off_by_default_and_a_transient_failure_is_retried_when_asked(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda seconds: None)
    answers = iter([httpx2.Response(503, json={"error": "busy"}), httpx2.Response(200, json=WIRE)])
    server = Server(lambda request: next(answers))
    assert ask(typesafe(server, max_retries=1)).answers["ok"].noul == 0.9
    assert len(server.requests) == 2


def test_the_sdk_error_it_does_not_know_is_a_bad_request_and_a_provider_error_passes_through():
    assert typesafe_error(ValueError("x")).kind == "bad_request"
    err = ProviderError("already ours", kind="server")
    assert typesafe_error(err) is err


# --- the key ---------------------------------------------------------------------------------


def hostile(request):
    """A server that echoes the key it was sent back in its error, as a careless one might."""
    return httpx2.Response(500, json={"error": f"bad token {request.headers['authorization']} / {KEY}"})


def test_the_key_is_in_the_authorization_header_and_nowhere_else(caplog):
    caplog.set_level(logging.DEBUG)
    server = Server()
    client = typesafe(server)
    answer = ask(client)

    request = server.requests[0]
    assert request.headers["authorization"] == f"Bearer {KEY}"
    assert KEY not in str(request.url) and KEY not in request.content.decode()
    assert not any(KEY in value for name, value in request.headers.items() if name.lower() != "authorization")
    assert KEY not in repr(client) and KEY not in repr(answer) and KEY not in str(client.name)
    assert KEY not in caplog.text and "typesafe_sdk" in caplog.text  # the SDK did log, and redacted its header


def test_a_server_that_echoes_the_key_does_not_get_it_into_an_error():
    client = typesafe(Server(hostile))
    with pytest.raises(ProviderError) as caught:
        ask(client)

    error = caught.value
    rendered = "\n".join([str(error), repr(error), repr(error.args), "".join(traceback.format_exception(error))])
    assert KEY not in rendered and "[redacted]" in str(error)


def test_a_key_is_never_in_what_a_failed_build_says():
    with pytest.raises(ValueError) as caught:
        build_client(SPEC, "m", KEY, {"timeout": "fast"})
    assert KEY not in str(caught.value)
    with pytest.raises(ProviderError) as caught:
        build_client(SPEC, "m", KEY, {"timeout": -1})  # the SDK refuses this one itself
    assert caught.value.kind == "bad_request" and KEY not in str(caught.value)


# ---------------------------------------------------------------------------------------------
# ChainedClassifier
# ---------------------------------------------------------------------------------------------


class Slot:
    """A classifier client from a script: a reply to return or an exception to raise, in turn."""

    def __init__(self, name, *script):
        self.name, self.model = name, "m"
        self.script = list(script) or [self.answer(name)]
        self.calls: list[dict] = []
        self.closed = False

    @staticmethod
    def answer(who: str) -> SimpleNamespace:
        return SimpleNamespace(
            answers={"kind": ChoiceAnswer(choice=who, confidence=0.9, probabilities={who: 0.9})}, usage=None, model=who
        )

    def system_one(self, state, questions, *, model=None, **kw):
        self.calls.append({"state": state, "model": model, **kw})
        step = self.script[min(len(self.calls), len(self.script)) - 1]
        if isinstance(step, BaseException):
            raise step
        return step

    def close(self):
        self.closed = True


def err(kind="server", **kw) -> ProviderError:
    return ProviderError(f"{kind} failure", kind=kind, **kw)


def chain_of(*slots, policy=None, **kw) -> ChainedClassifier:
    return ChainedClassifier.from_clients(slots, policy, **kw)


def pick(classifier) -> str:
    return classifier.system_one(state="s", questions={"kind": Choice(criteria={"a": None, "b": None})}).model


def test_the_first_classifier_answers_and_the_chain_remembers_which():
    a, b = Slot("a"), Slot("b")
    classifier = chain_of(a, b)
    assert pick(classifier) == "a" and classifier.last_slot == "a" and b.calls == []


def test_a_failing_classifier_hands_over_to_the_next_and_says_so():
    seen = []
    classifier = chain_of(Slot("a", err("rate_limit")), Slot("b"), on_event=seen.append)
    assert pick(classifier) == "b" and classifier.last_slot == "b"
    assert [(e.role, e.from_slot, e.to_slot, e.kind) for e in classifier.chain.events] == [("classifier", "a", "b", "rate_limit")]
    assert seen == list(classifier.chain.events)


def test_every_classifier_failing_raises_all_providers_failed_naming_each():
    classifier = chain_of(Slot("a", err("timeout")), Slot("b", err("auth")))
    with pytest.raises(AllProvidersFailed) as caught:
        pick(classifier)
    assert [(name, e.kind) for name, e in caught.value.errors] == [("a", "timeout"), ("b", "auth")]


def test_our_own_bad_request_is_not_tried_on_the_next_classifier():
    b = Slot("b")
    with pytest.raises(ProviderError) as caught:
        pick(chain_of(Slot("a", err("bad_request")), b))
    assert caught.value.kind == "bad_request" and b.calls == []


def test_an_llm_that_cannot_give_a_usable_answer_fails_over_to_the_next_slot():
    garbage = LLMClassifier(ScriptedLLM("no idea"), name="llm:scripted")
    classifier = chain_of(garbage, Slot("typesafe"))
    assert pick(classifier) == "typesafe"
    assert classifier.chain.events[-1].kind == "content" and classifier.chain.events[-1].from_slot == "llm:scripted"


def test_an_llm_chain_that_ran_out_of_providers_does_not_end_the_classifier_chain():
    """`exhausted` is not a failover kind. Passed up as it is, it would stop the chain before the slot after it."""
    inner = AllProvidersFailed("llm", [("x", err("timeout")), ("y", err("server"))])
    exhausted = LLMClassifier(ScriptedLLM(inner), name="llm:facade")
    b = Slot("b")
    classifier = chain_of(exhausted, b)

    assert pick(classifier) == "b" and len(b.calls) == 1
    event = classifier.chain.events[-1]
    assert event.from_slot == "llm:facade" and event.kind == "server"  # what the inner chain last failed on


def test_a_bare_sdk_client_in_a_slot_is_still_a_provider_failure_not_a_bug():
    class Raw(Slot):
        def system_one(self, state, questions, *, model=None, **kw):
            from typesafe_sdk import TypeSafeAPITimeoutError

            raise TypeSafeAPITimeoutError(10.0)

    assert pick(chain_of(Raw("raw"), Slot("b"))) == "b"


def test_a_bug_in_a_slot_propagates():
    with pytest.raises(KeyError):
        pick(chain_of(Slot("a", KeyError("bug")), Slot("b")))


def test_a_slow_classifier_is_raced_against_the_next_because_a_decision_is_safe_to_run_twice():
    release = threading.Event()

    class Slow(Slot):
        def system_one(self, state, questions, *, model=None, **kw):
            release.wait(5)
            return self.answer("slow")

    classifier = chain_of(Slow("slow"), Slot("fast"), policy=ChainPolicy(hedge_after_s=0.05))
    try:
        assert pick(classifier) == "fast"
    finally:
        release.set()
    assert [(e.from_slot, e.to_slot, e.kind) for e in classifier.chain.events] == [("slow", "fast", "slow")]


def test_a_model_is_passed_to_the_slots_only_when_the_caller_gave_one():
    slot = Slot("a")
    classifier = chain_of(slot)
    pick(classifier)
    classifier.system_one(state="s", questions={}, model="jev-other", timeout=3)
    assert slot.calls == [{"state": "s", "model": None}, {"state": "s", "model": "jev-other", "timeout": 3}]


def test_closing_closes_every_slot_even_when_one_refuses_and_the_context_manager_does_it():
    class Stubborn(Slot):
        def close(self):
            super().close()
            raise OSError("would not close")

    a, b, c = Slot("a"), Stubborn("b"), Slot("c")
    with pytest.raises(OSError), chain_of(a, b, c):
        pass
    assert (a.closed, b.closed, c.closed) == (True, True, True)

    d = Slot("d")
    with chain_of(d) as classifier:
        assert classifier.system_one(state="s", questions={}).model == "d"
    assert d.closed


def test_a_slot_with_only_a_context_manager_is_closed_through_it():
    class OnlyExit:
        name, model = "x", "m"
        exited = False

        def system_one(self, state, questions, **kw):
            return Slot.answer("x")

        def __exit__(self, *exc):
            self.exited = True

    slot = OnlyExit()
    chain_of(slot).close()
    assert slot.exited


def test_two_slots_with_one_name_are_refused():
    with pytest.raises(ValueError):
        chain_of(Slot("a"), Slot("a"))


def test_the_chain_composes_with_the_metered_client_unchanged():
    calls = Calls()
    classifier = MeteredClassifier(chain_of(LLMClassifier(ScriptedLLM("bad", reply()))), calls)
    classifier.system_one(state=STATE, questions=questions())
    assert calls.usage["m-1"].input_tokens == 200  # the retry's tokens are counted too

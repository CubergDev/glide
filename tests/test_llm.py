"""The OpenAI-compatible chat adapter and the LLM facade.

Offline: requests go to an httpx.MockTransport. Only the last section talks real HTTP, to a server on 127.0.0.1
that the test starts itself, to check what a mock cannot: connection reuse, chunked streaming, real timeouts.
"""

from __future__ import annotations

import contextlib
import gc
import json
import logging
import socket
import threading
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import combinations, pairwise

import httpx
import pytest

from glide.providers import llm as llm_module
from glide.providers.base import ProviderSpec
from glide.providers.chain import Chain, ChainPolicy, Slot
from glide.providers.errors import AllProvidersFailed, ProviderError
from glide.providers.llm import LLM, OpenAICompatLLM, build_client

KEY = "sk-test-SECRET-0123456789abcdef"
BASE = "https://llm.example.test/v1"
MSG = [{"role": "user", "content": "hi"}]
SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}, "reason": {"type": "string"}},
    "required": ["ok", "reason"],
    "additionalProperties": False,
}
JSON_REPLY = '{"ok": true, "reason": "fine"}'


# -- fakes ----------------------------------------------------------------------------------------


class Recorder:
    """The transport's handler: keeps every request and its parsed body, and answers with `reply(request, body, n)`."""

    def __init__(self, reply):
        self.reply = reply
        self.requests: list[httpx.Request] = []
        self.bodies: list[dict] = []
        self._lock = threading.Lock()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        with self._lock:
            self.requests.append(request)
            self.bodies.append(body)
            n = len(self.requests)
        return self.reply(request, body, n)

    def values(self, field: str) -> list:
        return [b.get(field) for b in self.bodies]

    def formats(self) -> list:
        return [(b.get("response_format") or {}).get("type") for b in self.bodies]


def scripted(*steps):
    """Answer each request with the next step, and the last one for ever after. A step is an exception to raise or a
    function that makes a fresh response."""

    def reply(request, body, n):
        step = steps[min(n - 1, len(steps) - 1)]
        if isinstance(step, BaseException):
            raise step
        return step()

    return reply


def completion(content="hello", *, finish="stop", usage=None, **message) -> dict:
    out = {"choices": [{"index": 0, "message": {"role": "assistant", "content": content, **message}, "finish_reason": finish}]}
    if usage is not None:
        out["usage"] = usage
    return out


def ok(content="hello", **kw):
    return lambda: httpx.Response(200, json=completion(content, **kw))


def status(code: int, body="boom", headers=None):
    def make():
        if isinstance(body, str):
            return httpx.Response(code, text=body, headers=headers)
        return httpx.Response(code, json=body, headers=headers)

    return make


def make(*steps, reply=None, name="t:m", **kw) -> tuple[OpenAICompatLLM, Recorder]:
    recorder = Recorder(reply or scripted(*steps))
    kw.setdefault("api_key", KEY)
    client = OpenAICompatLLM(name, "m", BASE, transport=httpx.MockTransport(recorder), **kw)
    return client, recorder


class Pieces(httpx.SyncByteStream):
    """A response body that arrives in pieces, optionally taking time, and notes whether it was closed."""

    def __init__(self, pieces, *, clock=None, seconds=0.0, then: BaseException | None = None):
        self.pieces = list(pieces)
        self.clock, self.seconds, self.then = clock, seconds, then
        self.closed = False

    def __iter__(self):
        for piece in self.pieces:
            if self.clock is not None:
                self.clock.advance(self.seconds[0] if isinstance(self.seconds, list) else self.seconds)
                if isinstance(self.seconds, list):
                    self.seconds = self.seconds[1:] or [0.0]
            yield piece
        if self.then is not None:
            raise self.then

    def close(self):
        self.closed = True


def event(data) -> bytes:
    text = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    return f"data: {text}\n\n".encode()


def delta(content=None, *, finish=None, **extra) -> dict:
    d = dict(extra)
    if content is not None:
        d["content"] = content
    return {"choices": [{"index": 0, "delta": d, "finish_reason": finish}]}


def sse(*events, tail=b"data: [DONE]\n\n", pieces=None, **body_kw):
    """A streaming reply. `events` are dicts (sent as `data: json`) or raw bytes; `pieces` re-cuts the body."""
    raw = b"".join(e if isinstance(e, bytes) else event(e) for e in events) + tail
    cut = pieces(raw) if pieces else [raw]

    def make_response():
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=Pieces(cut, **body_kw))

    return make_response


def texts(*contents, finish="stop"):
    """The events of a reply that says `contents` piece by piece."""
    return [delta(role="assistant"), *(delta(c) for c in contents), delta(finish=finish)]


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def error_of(call) -> ProviderError:
    with pytest.raises(ProviderError) as caught:
        call()
    return caught.value


# -- a plain chat ---------------------------------------------------------------------------------


def test_chat_posts_to_chat_completions_with_the_key_in_a_header_and_returns_the_reply():
    c, rec = make(
        ok(
            "Hi there",
            usage={"prompt_tokens": 12, "completion_tokens": 3},
            model="routed/other",
        )
    )
    result = c.chat(MSG, max_tokens=64, temperature=0.2)

    (request,) = rec.requests
    assert request.method == "POST" and str(request.url) == f"{BASE}/chat/completions"
    assert request.headers["authorization"] == f"Bearer {KEY}"
    assert KEY not in str(request.url)
    assert rec.bodies[0] == {"model": "m", "messages": MSG, "max_tokens": 64, "temperature": 0.2, "stream": False}
    assert (result.text, result.provider, result.model, result.finish_reason) == ("Hi there", "t:m", "m", "stop")
    assert (result.usage.input_tokens, result.usage.output_tokens) == (12, 3)
    assert result.ttft_s is None and result.logprobs is None and result.latency_s >= 0


@pytest.mark.parametrize(
    "base",
    [BASE, BASE + "/", BASE + "/chat/completions", "  " + BASE + "/ "],
)
def test_the_base_url_may_be_given_with_a_slash_or_the_full_path(base):
    rec = Recorder(scripted(ok()))
    OpenAICompatLLM("t:m", "m", base, KEY, transport=httpx.MockTransport(rec)).chat(MSG)
    assert str(rec.requests[0].url) == f"{BASE}/chat/completions"


def test_no_key_means_no_authorization_header():
    c, rec = make(ok(), api_key=None)
    c.chat(MSG)
    assert "authorization" not in rec.requests[0].headers


def test_a_malformed_setup_is_a_value_error_that_does_not_echo_the_key():
    for base in ("", "llm.example.test/v1", "ftp://x/v1"):
        with pytest.raises(ValueError, match="base_url"):
            OpenAICompatLLM("t:m", "m", base, KEY)
    with pytest.raises(ValueError, match="token_param"):
        OpenAICompatLLM("t:m", "m", BASE, KEY, {"token_param": "max_new_tokens"})
    with pytest.raises(ValueError) as caught:
        OpenAICompatLLM("t:m", "m", BASE, "sk-ünïcode-key")
    assert "sk-" not in str(caught.value)


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        # OpenAI and OpenRouter: prompt_tokens counts the cached ones too, so they are taken out of it.
        ({"prompt_tokens": 1200, "completion_tokens": 30, "prompt_tokens_details": {"cached_tokens": 1000}}, (200, 1000, 30)),
        # DeepSeek names the cached part itself.
        (
            {"prompt_tokens": 900, "completion_tokens": 5, "prompt_cache_hit_tokens": 700, "prompt_cache_miss_tokens": 200},
            (200, 700, 5),
        ),
        ({"prompt_tokens": 50, "completion_tokens": 2}, (50, 0, 2)),
        ({"prompt_tokens": 10, "completion_tokens": None, "prompt_tokens_details": {"cached_tokens": None}}, (10, 0, 0)),
        ({"prompt_tokens": 10, "prompt_tokens_details": {"cached_tokens": 99}}, (0, 99, 0)),  # never negative
        (None, (0, 0, 0)),
        ("junk", (0, 0, 0)),
    ],
)
def test_usage_counts_cached_tokens_apart(usage, expected):
    result = make(ok(usage=usage))[0].chat(MSG)
    assert (result.usage.input_tokens, result.usage.cached_input_tokens, result.usage.output_tokens) == expected


def test_logprobs_are_asked_for_and_returned_only_when_requested():
    entries = [{"token": "Hi", "logprob": -0.1, "top_logprobs": []}]
    reply = completion()
    reply["choices"][0]["logprobs"] = {"content": entries}
    c, rec = make(lambda: httpx.Response(200, json=reply))
    assert c.chat(MSG, logprobs=True).logprobs == entries
    assert rec.bodies[0]["logprobs"] is True
    assert c.chat(MSG).logprobs is None and "logprobs" not in rec.bodies[1]
    reply["choices"][0]["logprobs"] = None
    assert c.chat(MSG, logprobs=True).logprobs is None  # asked for, not given


def test_a_reply_cut_off_by_the_limit_still_returns_its_text_with_the_reason():
    result = make(ok("half a sent", finish="length"))[0].chat(MSG)
    assert (result.text, result.finish_reason) == ("half a sent", "length")


def test_content_given_as_parts_is_joined_and_other_parts_are_ignored():
    parts = [{"type": "text", "text": "Hel"}, {"type": "reasoning", "text": "secret"}, {"type": "text", "text": "lo"}]
    assert make(ok(parts))[0].chat(MSG).text == "Hello"


# -- thinking is never returned -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("<think>let me see</think>The answer.", "The answer."),
        ("<think>let me see</think>\n\n  The answer.\n", "The answer."),
        ("<THINK>x</Think>Shouting.", "Shouting."),
        ("Before <think>multi\nline\nthought</think> after", "Before after"),
        ("<think>a</think>One<think>b</think> two", "One two"),
        ("Cut off <think>never closes, the budget ran out", "Cut off"),
        ("1 < 2 and 3 > 2", "1 < 2 and 3 > 2"),
        ("use <thing> here", "use <thing> here"),
    ],
)
def test_an_inline_think_block_is_cut_out_of_a_chat_reply(content, expected):
    assert make(ok(content))[0].chat(MSG).text == expected


def test_reasoning_fields_are_never_returned_or_streamed():
    c, _ = make(
        ok("Answer", reasoning_content="SECRET one", reasoning="SECRET two", reasoning_details=[{"text": "SECRET three"}])
    )
    result = c.chat(MSG)
    assert result.text == "Answer" and "SECRET" not in repr(result)

    events = [
        delta(role="assistant", reasoning_content="SECRET a"),
        delta(reasoning="SECRET b"),
        delta(reasoning_details=[{"type": "reasoning.text", "text": "SECRET c"}]),
        delta("An"),
        delta("swer", reasoning_content="SECRET d"),
        delta(finish="stop"),
    ]
    c, _ = make(sse(*events))
    assert list(c.stream(MSG)) == ["An", "swer"]


def test_a_reply_with_reasoning_and_no_answer_is_an_error_and_never_hands_the_reasoning_back():
    for reply in (
        ok("", finish="length", reasoning_content="SECRET one"),
        ok(None, reasoning="SECRET two"),
        ok("", reasoning_details=[{"type": "reasoning.text", "text": "SECRET three"}]),
    ):
        error = error_of(lambda reply=reply: make(reply)[0].chat(MSG))
        assert error.kind == "content" and "SECRET" not in str(error)
    only_reasoning = [delta(role="assistant", reasoning_content="SECRET a"), delta(reasoning="SECRET b"), delta(finish="stop")]
    seen = []
    with pytest.raises(ProviderError) as caught:
        seen.extend(make(sse(*only_reasoning))[0].stream(MSG))
    assert seen == [] and caught.value.kind == "content" and "SECRET" not in str(caught.value)


def test_a_reply_that_is_only_thinking_is_a_content_error():
    error = error_of(lambda: make(ok("<think>and then I ran out</think>"))[0].chat(MSG))
    assert error.kind == "content"


# -- what the request asks for --------------------------------------------------------------------


def test_the_schema_is_asked_for_strictly_and_spelled_out_in_the_system_prompt():
    c, rec = make(ok(JSON_REPLY))
    messages = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "go"}]
    assert c.chat(messages, schema=SCHEMA).text == JSON_REPLY

    body = rec.bodies[0]
    assert body["response_format"] == {"type": "json_schema", "json_schema": {"name": "reply", "strict": True, "schema": SCHEMA}}
    system = body["messages"][0]["content"]
    assert system.startswith("Be brief.") and "single JSON object" in system and json.dumps(SCHEMA) in system
    assert messages[0]["content"] == "Be brief."  # the caller's own messages are untouched


def test_the_schema_goes_in_whichever_form_the_system_message_takes_and_is_added_when_there_is_none():
    c, rec = make(ok(JSON_REPLY))
    parts = [{"type": "text", "text": "Be brief."}]
    originals = [
        [{"role": "system", "content": parts}, {"role": "user", "content": "go"}],
        [{"role": "user", "content": "go"}],
    ]
    snapshots = json.dumps(originals)
    for messages in originals:
        c.chat(messages, schema=SCHEMA)
    assert json.dumps(originals) == snapshots  # not mutated, so a chain can hand them to the next provider
    first, second = rec.bodies[0]["messages"], rec.bodies[1]["messages"]
    assert (
        first[0]["content"][0] == {"type": "text", "text": "Be brief."} and json.dumps(SCHEMA) in first[0]["content"][1]["text"]
    )
    assert [m["role"] for m in second] == ["system", "user"] and json.dumps(SCHEMA) in second[0]["content"]


def test_a_plain_chat_sends_no_response_format_and_adds_nothing_to_the_prompt():
    c, rec = make(ok())
    c.chat(MSG)
    assert "response_format" not in rec.bodies[0] and rec.bodies[0]["messages"] == MSG
    c.chat(MSG, schema=None)
    assert "response_format" not in rec.bodies[1]


def test_content_parts_with_images_are_passed_through_unchanged():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "what?"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
            ],
        }
    ]
    c, rec = make(ok())
    c.chat(messages)
    assert rec.bodies[0]["messages"] == messages


def test_reasoning_effort_is_sent_only_when_it_is_set():
    c, rec = make(ok(), options={"reasoning_effort": "none"})
    c.chat(MSG)
    assert rec.values("reasoning_effort") == ["none"]
    c, rec = make(ok())
    c.chat(MSG)
    assert "reasoning_effort" not in rec.bodies[0]


def test_extra_body_and_headers_are_merged_and_the_fields_the_client_owns_win():
    options = {
        "extra_body": {
            "provider": {"order": ["x"]},
            "model": "WRONG",
            "stream": True,
            "messages": [],
            "thinking": {"type": "disabled"},
        },
        "extra_headers": {"X-Title": "glide", "HTTP-Referer": "https://example.test"},
    }
    c, rec = make(ok(), options=options)
    c.chat(MSG)
    body = rec.bodies[0]
    assert body["provider"] == {"order": ["x"]} and body["thinking"] == {"type": "disabled"}
    assert (body["model"], body["stream"], body["messages"]) == ("m", False, MSG)
    assert rec.requests[0].headers["x-title"] == "glide" and rec.requests[0].headers["authorization"] == f"Bearer {KEY}"
    options["extra_body"]["provider"]["order"].append("later")  # the client took a copy
    c.chat(MSG)
    assert rec.bodies[1]["provider"] == {"order": ["x"]}


def test_extra_headers_may_replace_authorization():
    c, rec = make(ok(), options={"extra_headers": {"Authorization": "Token other"}})
    c.chat(MSG)
    assert rec.requests[0].headers["authorization"] == "Token other"


def test_options_it_does_not_know_are_ignored():
    c, rec = make(ok(), options={"models": ["a", "b"], "timeout": 3})
    assert c.chat(MSG).text == "hello" and len(rec.requests) == 1


# -- learning what the endpoint refuses -----------------------------------------------------------


def refusing(message: str, when):
    """A reply function that answers 400 `message` to a request for which `when(body)` is true."""

    def reply(request, body, n):
        if when(body):
            return httpx.Response(400, json={"error": {"message": message, "type": "invalid_request_error"}})
        return httpx.Response(200, json=completion(JSON_REPLY))

    return reply


def test_a_refused_response_format_steps_down_in_order_and_stays_down():
    c, rec = make(
        reply=refusing("Invalid parameter: response_format is not supported", lambda b: "response_format" in b),
    )
    assert c.chat(MSG, schema=SCHEMA).text == JSON_REPLY
    assert rec.formats() == ["json_schema", "json_object", None]
    assert c.chat(MSG, schema=SCHEMA).text == JSON_REPLY
    assert rec.formats() == ["json_schema", "json_object", None, None]  # the second call starts where the first ended
    assert c.settings["response_format"] == "none"
    for body in rec.bodies:  # and in every case the schema is in the prompt as well
        assert json.dumps(SCHEMA) in body["messages"][0]["content"]


def test_one_step_down_is_enough_when_json_object_is_accepted():
    c, rec = make(
        reply=refusing(
            "response_format json_schema is unavailable", lambda b: (b.get("response_format") or {}).get("type") == "json_schema"
        )
    )
    c.chat(MSG, schema=SCHEMA)
    c.chat(MSG, schema=SCHEMA)
    assert rec.formats() == ["json_schema", "json_object", "json_object"]


@pytest.mark.parametrize("word", ["response_format", "json_schema", "json_object", "RESPONSE_FORMAT"])
def test_any_of_the_format_words_triggers_the_step_down(word):
    c, rec = make(reply=refusing(f"bad {word}", lambda b: (b.get("response_format") or {}).get("type") == "json_schema"))
    c.chat(MSG, schema=SCHEMA)
    assert rec.formats() == ["json_schema", "json_object"]


def test_a_400_that_is_about_something_else_is_one_request_and_a_bad_request():
    c, rec = make(status(400, {"error": {"message": "messages: field required"}}))
    error = error_of(lambda: c.chat(MSG, schema=SCHEMA))
    assert error.kind == "bad_request" and error.status == 400 and len(rec.requests) == 1
    assert c.settings["response_format"] == "json_schema"  # nothing was learned


def test_a_format_refusal_without_a_schema_is_not_ours_to_fix():
    c, rec = make(status(400, "response_format is broken"))
    assert error_of(lambda: c.chat(MSG)).kind == "bad_request" and len(rec.requests) == 1


def test_a_server_that_refuses_every_format_ends_in_an_error_after_a_bounded_number_of_requests():
    c, rec = make(
        status(400, "unknown parameter response_format temperature reasoning_effort max_completion_tokens"),
        options={"reasoning_effort": "low"},
    )
    assert error_of(lambda: c.chat(MSG, schema=SCHEMA)).kind == "bad_request"
    assert len(rec.requests) == 6  # the first, then one for each of the five things it could be asked without


OPENAI_MAX_TOKENS = "Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead."


def test_a_400_that_names_max_completion_tokens_switches_the_parameter_once():
    c, rec = make(reply=refusing(OPENAI_MAX_TOKENS, lambda b: "max_tokens" in b))
    c.chat(MSG, max_tokens=77)
    c.chat(MSG, max_tokens=77)
    assert [("max_tokens" in b, b.get("max_completion_tokens")) for b in rec.bodies] == [(True, None), (False, 77), (False, 77)]
    assert c.settings["token_param"] == "max_completion_tokens"


def test_the_token_parameter_switch_works_the_other_way_and_never_flips_back():
    message = "Unsupported parameter: 'max_completion_tokens'. Use 'max_tokens' instead."
    c, rec = make(
        reply=refusing(message, lambda b: "max_completion_tokens" in b), options={"token_param": "max_completion_tokens"}
    )
    c.chat(MSG)
    assert [("max_tokens" in b, "max_completion_tokens" in b) for b in rec.bodies] == [(False, True), (True, False)]

    # A server that names both parameters whichever it was sent cannot make it ping-pong.
    c, rec = make(status(400, "max_tokens or max_completion_tokens is wrong"))
    assert error_of(lambda: c.chat(MSG)).kind == "bad_request"
    assert len(rec.requests) == 2


def test_streams_learn_the_token_parameter_too():
    def reply(request, body, n):
        if "max_tokens" in body:
            return httpx.Response(400, json={"error": {"message": OPENAI_MAX_TOKENS}})
        return sse(*texts("ok"))()

    c, rec = make(reply=reply)
    assert list(c.stream(MSG, max_tokens=9)) == ["ok"]
    assert "max_completion_tokens" in rec.bodies[1] and rec.bodies[1]["stream"] is True


def test_a_refused_reasoning_effort_is_dropped_for_good_and_says_so():
    c, rec = make(
        reply=refusing("Unrecognized request argument supplied: reasoning_effort", lambda b: "reasoning_effort" in b),
        options={"reasoning_effort": "none"},
    )
    assert c.settings["reasoning_effort"] == "none"
    c.chat(MSG)
    c.chat(MSG)
    assert ["reasoning_effort" in b for b in rec.bodies] == [True, False, False]
    assert c.settings["reasoning_effort"] is None


def test_a_reasoning_model_that_only_takes_its_default_temperature_is_asked_without_it():
    message = "Unsupported value: 'temperature' does not support 0.0 with this model. Only the default (1) value is supported."
    c, rec = make(reply=refusing(message, lambda b: "temperature" in b))
    c.chat(MSG)
    c.chat(MSG)
    assert ["temperature" in b for b in rec.bodies] == [True, False, False]


def test_every_refusal_can_happen_on_one_call_and_each_is_learned():
    def reply(request, body, n):
        said = []
        if "max_tokens" in body:
            said.append("use max_completion_tokens")
        if "reasoning_effort" in body:
            said.append("reasoning_effort unsupported")
        if "temperature" in body:
            said.append("temperature unsupported")
        if (body.get("response_format") or {}).get("type") == "json_schema":
            said.append("json_schema unsupported")
        if said:
            return httpx.Response(400, json={"error": {"message": "; ".join(said[:1])}})
        return httpx.Response(200, json=completion(JSON_REPLY))

    c, rec = make(reply=reply, options={"reasoning_effort": "low"})
    assert c.chat(MSG, schema=SCHEMA).text == JSON_REPLY
    assert len(rec.requests) == 5
    assert c.settings == {
        "response_format": "json_object",
        "token_param": "max_completion_tokens",
        "reasoning_effort": None,
        "temperature": False,
    }


def test_422_is_read_like_400():
    c, rec = make(
        reply=lambda request, body, n: (
            httpx.Response(422, json={"detail": [{"msg": "response_format not allowed"}]})
            if (body.get("response_format") or {}).get("type") == "json_schema"
            else httpx.Response(200, json=completion(JSON_REPLY))
        )
    )
    assert c.chat(MSG, schema=SCHEMA).text == JSON_REPLY and rec.formats() == ["json_schema", "json_object"]


def test_two_calls_refused_at_the_same_moment_step_down_once_between_them():
    barrier = threading.Barrier(2)

    def reply(request, body, n):
        if (body.get("response_format") or {}).get("type") == "json_schema":
            if n <= 2:
                barrier.wait(5)  # both requests are in flight before either learns anything
            return httpx.Response(400, json={"error": {"message": "response_format unsupported"}})
        return httpx.Response(200, json=completion(JSON_REPLY))

    c, rec = make(reply=reply)
    results = []
    threads = [threading.Thread(target=lambda: results.append(c.chat(MSG, schema=SCHEMA).text)) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert results == [JSON_REPLY, JSON_REPLY]
    assert c.settings["response_format"] == "json_object"  # not "none": the second 400 did not step down again
    assert rec.formats().count("json_object") == 2 and rec.formats().count(None) == 0


def test_calls_refused_for_different_reasons_at_the_same_moment_each_teach_the_client_something():
    barrier = threading.Barrier(2)

    def reply(request, body, n):
        refused = None
        if n <= 2:
            barrier.wait(5)  # both requests are in flight before either learns anything
            refused = "temperature is unsupported" if n == 1 else "response_format is unsupported"
        elif "temperature" in body:
            refused = "temperature is unsupported"
        elif (body.get("response_format") or {}).get("type") == "json_schema":
            refused = "response_format is unsupported"
        if refused:
            return httpx.Response(400, json={"error": {"message": refused}})
        return httpx.Response(200, json=completion(JSON_REPLY))

    c, _ = make(reply=reply)
    results = []
    threads = [threading.Thread(target=lambda: results.append(c.chat(MSG, schema=SCHEMA).text)) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    assert results == [JSON_REPLY, JSON_REPLY]
    assert c.settings["temperature"] is False and c.settings["response_format"] == "json_object"


# -- replies that cannot be used ------------------------------------------------------------------


def test_an_empty_reply_that_hit_the_limit_says_the_budget_went_on_thinking():
    error = error_of(lambda: make(ok("", finish="length"))[0].chat(MSG, max_tokens=50))
    assert error.kind == "content" and "thinking" in str(error) and "token" in str(error)
    assert error.provider == "t:m"


@pytest.mark.parametrize("content", ["", None, "   \n"])
def test_any_empty_reply_is_a_content_error(content):
    error = error_of(lambda: make(ok(content))[0].chat(MSG))
    assert error.kind == "content" and "no text" in str(error)


def test_a_refusal_is_a_content_error_that_says_so():
    error = error_of(lambda: make(ok(None, refusal="I can't help with that."))[0].chat(MSG, schema=SCHEMA))
    assert error.kind == "content" and "refused" in str(error)


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        (JSON_REPLY, JSON_REPLY),
        (f"```json\n{JSON_REPLY}\n```", JSON_REPLY),
        (f"Sure, here you go: {JSON_REPLY} Hope that helps!", JSON_REPLY),
        (
            '{"nested": {"ok": false, "reason": "inner"}, "ok": true, "reason": "outer"}',
            '{"nested": {"ok": false, "reason": "inner"}, "ok": true, "reason": "outer"}',
        ),
        ('{"extra": 1, "ok": false, "reason": ""}', '{"extra": 1, "ok": false, "reason": ""}'),
    ],
)
def test_with_a_schema_the_json_is_found_in_the_reply(reply, expected):
    assert make(ok(reply))[0].chat(MSG, schema=SCHEMA).text == expected


@pytest.mark.parametrize(
    "reply",
    [
        "I would say yes.",
        '{"ok": true, "reason": "cut off',  # cut off before it closes
        '{"ok": true}',  # a required field is missing
        '["ok", "reason"]',  # not an object
        '{"outer": {"ok": true, "reason": "x"}}',  # the fitting object is only nested in a wrong one
    ],
)
def test_with_a_schema_a_reply_that_is_not_that_json_is_a_content_error(reply):
    error = error_of(lambda: make(ok(reply))[0].chat(MSG, schema=SCHEMA))
    assert error.kind == "content" and error.provider == "t:m"


def test_a_schema_for_an_array_wants_an_array():
    schema = {"type": "array", "items": {"type": "integer"}}
    assert make(ok("The numbers: [1, 2, 3]"))[0].chat(MSG, schema=schema).text == "[1, 2, 3]"
    assert error_of(lambda: make(ok('{"numbers": 1}'))[0].chat(MSG, schema=schema)).kind == "content"


def test_a_truncated_json_reply_says_it_hit_the_token_limit():
    error = error_of(lambda: make(ok('{"ok": tr', finish="length"))[0].chat(MSG, schema=SCHEMA))
    assert error.kind == "content" and "token limit" in str(error)


def test_without_a_schema_the_text_is_not_looked_at_for_json():
    assert make(ok("just words"))[0].chat(MSG).text == "just words"


@pytest.mark.parametrize(
    "response",
    [
        lambda: httpx.Response(200, text="<html>captive portal</html>"),
        lambda: httpx.Response(200, text=""),
        lambda: httpx.Response(200, json=["not", "an", "object"]),
        lambda: httpx.Response(200, json={"choices": []}),
        lambda: httpx.Response(200, json={"choices": [{"finish_reason": "stop"}]}),
        lambda: httpx.Response(200, json={"id": "x"}),
        lambda: httpx.Response(200, content=b"\xff\xfe\x00bad"),
    ],
)
def test_a_reply_that_is_not_a_chat_completion_is_a_content_error(response):
    assert error_of(lambda: make(response)[0].chat(MSG)).kind == "content"


@pytest.mark.parametrize(
    ("body", "kind", "retry"),
    [
        ({"error": {"code": 429, "message": "slow down"}}, "rate_limit", None),
        ({"error": {"code": 502, "message": "upstream died"}}, "server", None),
        ({"error": {"code": 401, "message": "no"}}, "auth", None),
        ({"error": {"code": "weird", "message": "?"}}, "server", None),
        ({"error": "plain string"}, "server", None),
    ],
)
def test_a_200_with_an_error_in_the_body_is_mapped_like_its_status(body, kind, retry):
    error = error_of(lambda: make(lambda: httpx.Response(200, json=body))[0].chat(MSG))
    assert error.kind == kind


def test_an_error_inside_the_choice_is_an_error():
    reply = completion("", finish="error")
    reply["choices"][0]["error"] = {"code": 503, "message": "provider unavailable"}
    assert error_of(lambda: make(lambda: httpx.Response(200, json=reply))[0].chat(MSG)).kind == "server"
    reply = completion("", finish="error")
    assert error_of(lambda: make(lambda: httpx.Response(200, json=reply))[0].chat(MSG)).kind == "server"


# -- errors ---------------------------------------------------------------------------------------

STATUS_KINDS = {
    400: "bad_request",
    401: "auth",
    402: "auth",  # OpenRouter's out of credits: this key is spent, another provider may answer
    403: "auth",
    404: "unsupported",
    408: "timeout",
    422: "bad_request",
    429: "rate_limit",
    500: "server",
    502: "server",
    503: "server",
    504: "timeout",
}


@pytest.mark.parametrize(("code", "kind"), list(STATUS_KINDS.items()))
def test_status_maps_to_the_error_kind_for_chat_and_for_stream(code, kind):
    c, rec = make(status(code, "nope"))
    error = error_of(lambda: c.chat(MSG))
    assert (error.kind, error.status, error.provider) == (kind, code, "t:m") and str(code) in str(error)
    error = error_of(lambda: list(c.stream(MSG)))
    assert (error.kind, error.status) == (kind, code)
    assert len(rec.requests) == 2  # nothing was retried: there was nothing to learn from it


def test_a_stream_that_is_refused_has_its_error_body_read_into_the_message():
    c, _ = make(status(503, '{"error": {"message": "model is overloaded, try later"}}'))
    assert "model is overloaded" in str(error_of(lambda: list(c.stream(MSG))))


def test_a_spent_quota_reported_as_a_429_is_treated_like_a_spent_key_not_a_rate_limit():
    body = {"error": {"message": "You exceeded your current quota", "type": "insufficient_quota", "code": "insufficient_quota"}}
    error = error_of(lambda: make(status(429, body))[0].chat(MSG))
    assert error.kind == "auth" and error.status == 429
    assert error_of(lambda: make(status(429, "slow down"))[0].chat(MSG)).kind == "rate_limit"


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({"retry-after": "7"}, 7.0),
        ({"retry-after": "2.5"}, 2.5),
        ({"retry-after-ms": "1500"}, 1.5),
        ({"retry-after-ms": "1500", "retry-after": "9"}, 1.5),
        ({"retry-after": "-3"}, 0.0),
        ({"retry-after": "soon"}, None),
        ({"retry-after": "nan"}, None),
        ({}, None),
    ],
)
def test_retry_after_comes_through_on_a_rate_limit(headers, expected):
    error = error_of(lambda: make(status(429, "slow down", headers))[0].chat(MSG))
    assert error.kind == "rate_limit" and error.retry_after == expected


def test_retry_after_may_be_a_date():
    when = format_datetime(datetime.now(UTC) + timedelta(seconds=90), usegmt=True)
    error = error_of(lambda: make(status(429, "x", {"retry-after": when}))[0].chat(MSG))
    assert 80 <= error.retry_after <= 91
    error = error_of(lambda: list(make(status(503, "x", {"retry-after": when}))[0].stream(MSG)))
    assert 80 <= error.retry_after <= 91


TRANSPORT_FAILURES = [
    (httpx.ConnectError("refused"), "transport"),
    (httpx.ConnectTimeout("slow"), "timeout"),
    (httpx.ReadTimeout("slow"), "timeout"),
    (httpx.WriteTimeout("slow"), "timeout"),
    (httpx.PoolTimeout("busy"), "timeout"),
    (httpx.ReadError("reset"), "transport"),
    (httpx.WriteError("reset"), "transport"),
    (httpx.RemoteProtocolError("bad framing"), "transport"),
    (httpx.LocalProtocolError("bad header"), "transport"),
    (httpx.ProxyError("proxy"), "transport"),
    (httpx.UnsupportedProtocol("scheme"), "transport"),
    (httpx.DecodingError("gzip"), "transport"),
    (httpx.TooManyRedirects("loop"), "transport"),
    (httpx.InvalidURL("bad"), "transport"),
]


@pytest.mark.parametrize(
    ("failure", "kind"), TRANSPORT_FAILURES, ids=lambda v: type(v).__name__ if isinstance(v, Exception) else v
)
def test_no_raw_httpx_exception_escapes_from_chat_or_stream(failure, kind):
    c, _ = make(failure)
    for call in (lambda: c.chat(MSG), lambda: list(c.stream(MSG))):
        error = error_of(call)
        assert type(error) is ProviderError and error.kind == kind and error.provider == "t:m"


def test_a_connection_that_dies_mid_stream_is_a_provider_error_after_the_text_that_came():
    c, _ = make(sse(*texts("one ", "two")[:2], tail=b"", then=httpx.ReadError("reset")))
    seen = []
    with pytest.raises(ProviderError) as caught:
        for piece in c.stream(MSG):
            seen.append(piece)
    assert seen == ["one "] and caught.value.kind == "transport"


def test_a_body_that_dies_mid_chat_is_a_provider_error():
    def response():
        return httpx.Response(200, stream=Pieces([b'{"choices": [{"mess'], then=httpx.ReadError("reset")))

    assert error_of(lambda: make(response)[0].chat(MSG)).kind == "transport"


def test_timeouts_are_5_seconds_to_connect_and_the_timeout_argument_to_read():
    c, rec = make(ok())
    c.chat(MSG)
    c.chat(MSG, timeout=8)
    c.chat(MSG, timeout=2)
    first, second, third = (r.extensions["timeout"] for r in rec.requests)
    assert (first["connect"], first["read"]) == (5.0, 30.0)
    assert (second["connect"], second["read"]) == (5.0, 8.0)
    assert (third["connect"], third["read"]) == (2.0, 2.0)  # never longer than the whole budget

    s, srec = make(sse(*texts("x")))
    list(s.stream(MSG, timeout=3))
    assert (srec.requests[0].extensions["timeout"]["connect"], srec.requests[0].extensions["timeout"]["read"]) == (3.0, 3.0)


def test_a_server_that_dribbles_a_body_forever_is_cut_off_at_the_timeout(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(llm_module, "_clock", clock)
    pieces = [b'{"choices": [{"message": {"content": "' + b"x" * 3 + b'"}}]}'] + [b" "] * 50
    c, _ = make(lambda: httpx.Response(200, stream=Pieces(pieces, clock=clock, seconds=10)))
    error = error_of(lambda: c.chat(MSG, timeout=25))
    assert error.kind == "timeout"


def test_keepalive_comments_do_not_hold_the_first_token_off_past_the_timeout(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(llm_module, "_clock", clock)

    def separate(raw):  # every event in a piece of its own, each taking ten seconds to arrive
        return [piece + b"\n\n" for piece in raw.split(b"\n\n") if piece]

    c, _ = make(sse(*([b": OPENROUTER PROCESSING\n\n"] * 20), *texts("late"), pieces=separate, clock=clock, seconds=10))
    error = error_of(lambda: list(c.stream(MSG, timeout=25)))
    assert error.kind == "timeout"


def test_a_connection_the_server_closed_while_it_idled_is_retried_once_on_a_fresh_one():
    c, rec = make(httpx.RemoteProtocolError("Server disconnected without sending a response."), ok("second try"))
    assert c.chat(MSG).text == "second try" and len(rec.requests) == 2

    c, rec = make(httpx.ReadError("reset"), sse(*texts("fresh")))
    assert "".join(c.stream(MSG)) == "fresh" and len(rec.requests) == 2

    c, rec = make(httpx.RemoteProtocolError("still hanging up"))  # a server that always does is an error, after one retry
    assert error_of(lambda: c.chat(MSG)).kind == "transport" and len(rec.requests) == 2


def test_a_reset_after_a_long_wait_is_not_retried_because_that_would_break_the_timeout(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(llm_module, "_clock", clock)

    def reply(request, body, n):
        clock.advance(10)  # the server sat on the request, then dropped it
        raise httpx.ReadError("reset")

    c, rec = make(reply=reply)
    assert error_of(lambda: c.chat(MSG, timeout=30)).kind == "transport"
    assert len(rec.requests) == 1


def test_a_quick_reset_is_not_retried_either_once_the_whole_budget_is_spent(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(llm_module, "_clock", clock)

    def reply(request, body, n):
        clock.advance(4)  # quick enough to look stale, but the 3 s budget is gone
        raise httpx.RemoteProtocolError("Server disconnected without sending a response.")

    c, rec = make(reply=reply)
    assert error_of(lambda: c.chat(MSG, timeout=3)).kind == "transport"
    assert len(rec.requests) == 1


def test_a_refused_connection_or_a_timeout_is_not_retried():
    for failure in (httpx.ConnectError("refused"), httpx.ConnectTimeout("slow"), httpx.ReadTimeout("slow")):
        c, rec = make(failure)
        error_of(lambda c=c: c.chat(MSG))
        assert len(rec.requests) == 1  # a dead host would otherwise cost two connect timeouts before the chain moved on


def test_a_redirect_is_an_error_and_is_not_followed_with_the_key():
    c, rec = make(status(307, "", {"location": "https://elsewhere.example.test/v1/chat/completions"}))
    assert error_of(lambda: c.chat(MSG)).kind == "bad_request"
    assert len(rec.requests) == 1


# -- the key never leaks --------------------------------------------------------------------------

BLOB = "A" * 200


def echoing(status_code: int, where: str = "text"):
    def make_response():
        body = f'{{"error": {{"message": "Incorrect API key provided: {KEY}. Payload {BLOB}"}}}}'
        return httpx.Response(status_code, text=body)

    return make_response


LEAKY_CASES = {
    "401 echoing the key": (echoing(401), "chat"),
    "500 echoing the key": (echoing(500), "chat"),
    "400 echoing the key": (echoing(400), "chat"),
    "429 echoing the key": (echoing(429), "chat"),
    "200 html echoing the key": (lambda: httpx.Response(200, text=f"<html>{KEY}</html>"), "chat"),
    "200 error body echoing the key": (
        lambda: httpx.Response(200, json={"error": {"code": 500, "message": f"bad {KEY}"}}),
        "chat",
    ),
    "200 error without a code": (lambda: httpx.Response(200, json={"error": {"message": f"bad {KEY}"}}), "chat"),
    "timeout": (httpx.ReadTimeout(f"timed out talking with {KEY}"), "chat"),
    "connect failure": (httpx.ConnectError(f"cannot connect {KEY}"), "chat"),
    "stream 401 echoing the key": (echoing(401), "stream"),
    "stream 503 echoing the key": (echoing(503), "stream"),
    "stream error event echoing the key": (
        sse({"error": {"code": 500, "message": f"oops {KEY}"}}),
        "stream",
    ),
    "stream event that is not JSON": (sse(f"{KEY} not json".encode().join([b"data: ", b"\n\n"])), "stream"),
}


@pytest.mark.parametrize("case", list(LEAKY_CASES))
def test_the_key_is_in_no_error_message_repr_or_url(case, caplog):
    caplog.set_level(logging.DEBUG)
    step, how = LEAKY_CASES[case]
    c, rec = make(step)
    error = error_of(lambda: c.chat(MSG) if how == "chat" else list(c.stream(MSG)))

    assert type(error) is ProviderError
    for text in (str(error), repr(error), repr(vars(error)), repr(c), repr(vars(c)), repr(Slot("a", c)), caplog.text):
        assert KEY not in text
    assert BLOB not in str(error)  # nor does the start of an echoed payload
    assert all(KEY not in str(r.url) for r in rec.requests)
    assert rec.requests[0].headers["authorization"] == f"Bearer {KEY}"  # it is sent, in the header and only there


def test_the_key_is_not_in_a_switch_event_after_a_failover(caplog):
    caplog.set_level(logging.DEBUG)
    seen = []
    a, _ = make(echoing(401), name="a:m")
    b, _ = make(ok("from b"), name="b:m")
    llm = LLM(Chain("llm", [Slot("a:m", a), Slot("b:m", b)], on_event=seen.append))
    assert llm.chat(MSG).text == "from b"
    (event,) = llm.chain.events
    assert event.kind == "auth" and KEY not in repr(event) and KEY not in repr(seen) and KEY not in caplog.text
    assert KEY not in repr(llm.chain.status())


def test_a_key_with_spaces_around_it_is_trimmed_and_scrubbed():
    c, rec = make(status(401, "bad key sk-padded-key"), api_key="  sk-padded-key\n")
    error = error_of(lambda: c.chat(MSG))
    assert rec.requests[0].headers["authorization"] == "Bearer sk-padded-key" and "sk-padded-key" not in str(error)


def test_the_repr_names_the_slot_and_host_and_nothing_else():
    c = OpenAICompatLLM(
        "or:m", "m", "https://user:pw@host.example.test:8443/v1?key=QUERYKEY", KEY, {"extra_headers": {"X-Secret": "HDR"}}
    )
    text = repr(c)
    assert text == "OpenAICompatLLM(name='or:m', base_url='https://host.example.test:8443/v1')"
    assert "QUERYKEY" not in text and "pw" not in text and "HDR" not in text


# -- streaming ------------------------------------------------------------------------------------


def test_a_stream_yields_only_text_and_skips_everything_else():
    events = [
        delta(role="assistant", content=""),  # a role-only first delta, as OpenAI sends
        b": keepalive\n\n",
        b": OPENROUTER PROCESSING\n\n",
        delta(role="assistant"),
        delta("Hel"),
        delta(""),
        delta("lo", logprobs=None),
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {"choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 7, "prompt_tokens_details": {"cached_tokens": 60}}},
    ]
    c, rec = make(sse(*events))
    pieces = list(c.stream(MSG, max_tokens=33, temperature=0.4))
    assert pieces == ["Hel", "lo"] and "" not in pieces

    body = rec.bodies[0]
    assert body["stream"] is True and body["stream_options"] == {"include_usage": True}
    assert (body["max_tokens"], body["temperature"]) == (33, 0.4)
    assert rec.requests[0].headers["accept"] == "text/event-stream"
    assert (c.last_usage.input_tokens, c.last_usage.cached_input_tokens, c.last_usage.output_tokens) == (40, 60, 7)
    assert c.last_finish_reason == "stop"


def test_time_to_first_token_is_measured_from_the_request_to_the_first_visible_text(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(llm_module, "_clock", clock)
    pieces = [
        event(delta(role="assistant")),
        b": keepalive\n\n",
        event(delta("Hel")),
        event(delta("lo")),
        event(delta(finish="stop")),
        b"data: [DONE]\n\n",
    ]
    c, _ = make(
        lambda: httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=Pieces(pieces, clock=clock, seconds=[0.1, 0.1, 0.3, 0.2, 0.1, 0.1]),
        )
    )
    assert c.last_ttft_s is None
    assert "".join(c.stream(MSG)) == "Hello"
    assert c.last_ttft_s == pytest.approx(0.5)  # not when the role-only delta came, nor the first byte

    make_chat = make(ok())[0]
    make_chat.chat(MSG)
    assert make_chat.last_ttft_s is None


def test_ttft_after_a_think_block_is_the_first_visible_token(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(llm_module, "_clock", clock)
    pieces = [
        event(delta("<think>hm")),
        event(delta("m</think>")),
        event(delta("Yes")),
        event(delta(finish="stop")),
        b"data: [DONE]\n\n",
    ]
    c, _ = make(
        lambda: httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=Pieces(pieces, clock=clock, seconds=1.0)
        )
    )
    assert list(c.stream(MSG)) == ["Yes"] and c.last_ttft_s == pytest.approx(3.0)


THINK_TEXT = "a<think>x</think>b"


def cut(text: str, cuts) -> list[str]:
    bounds = [0, *cuts, len(text)]
    return [text[i:j] for i, j in pairwise(bounds)]


def streamed(c: OpenAICompatLLM) -> str:
    return "".join(c.stream(MSG))


def test_a_think_block_is_cut_whichever_way_the_chunks_split_the_tags():
    current: list[list[str]] = []

    def reply(request, body, n):
        return sse(*texts(*current[0]))()

    c, _ = make(reply=reply)
    count = 0
    for k in (0, 1, 2, 3):
        for cuts in combinations(range(1, len(THINK_TEXT)), k):
            current[:] = [cut(THINK_TEXT, cuts)]
            assert streamed(c) == "ab", current[0]
            count += 1
    assert count > 800


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("a < b", "a < b"),
        ("x <", "x <"),  # a held-back "<" at the very end was text
        ("use <thing> ok", "use <thing> ok"),
        ("<THINK>x</THINK>  y", "y"),
        ("hello <think>never closes", "hello "),
        ("<think>x</think>\n\nAnswer", "Answer"),
        ("1</think>2", "1</think>2"),  # a closing tag with nothing open is plain text
    ],
)
def test_think_filter_leaves_other_text_alone_whatever_the_split(text, expected):
    current: list[list[str]] = []
    c, _ = make(reply=lambda request, body, n: sse(*texts(*current[0]))())
    for k in (0, 1, 2):
        for cuts in combinations(range(1, len(text)), k):
            current[:] = [cut(text, cuts)]
            try:
                got = streamed(c)
            except ProviderError as e:  # a reply that was only thinking has nothing in it
                assert e.kind == "content" and expected == ""
                continue
            assert got == expected, current[0]


def test_a_stream_of_only_thinking_is_a_content_error():
    c, _ = make(sse(*texts("<think>", "all of it", "</think>")))
    assert error_of(lambda: list(c.stream(MSG))).kind == "content"


@pytest.mark.parametrize(
    "pieces",
    [
        lambda raw: [raw],
        lambda raw: [raw[i : i + 7] for i in range(0, len(raw), 7)],
        lambda raw: [bytes([b]) for b in raw],  # also splits the multi-byte characters
    ],
    ids=["whole", "sevens", "bytes"],
)
def test_events_are_reassembled_however_the_network_cuts_the_body(pieces):
    c, _ = make(sse(*texts("héllo ", "wörld €", " ✓"), pieces=pieces))
    assert "".join(c.stream(MSG)) == "héllo wörld € ✓"


# U+2028, U+0085 and U+2029 end a line for str.splitlines, and so for httpx's iter_lines, but not for SSE.
SEPARATORS = "a" + chr(0x2028) + "b" + chr(0x85) + "c" + chr(0x2029) + "d"
BOM = chr(0xFEFF)


def raw_stream(body: bytes):
    return lambda: httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)


def chunk_line(content=None, **kw) -> str:
    return json.dumps(delta(content, **kw), ensure_ascii=False)


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        pytest.param(
            f"data:{chunk_line('A')}\r\n\r\ndata:{chunk_line('B')}\r\n\r\ndata:[DONE]\r\n\r\n",
            "AB",
            id="crlf-and-no-space-after-the-colon",
        ),
        pytest.param(f"data: {chunk_line('A')}\r\rdata: {chunk_line('B')}\r\rdata: [DONE]\r\r", "AB", id="lone-cr-line-ends"),
        pytest.param(f"data: {chunk_line('A')}\n\ndata: {chunk_line(finish='stop')}", "A", id="no-done-and-no-final-blank-line"),
        pytest.param(f"data: {chunk_line('A')}\n\ndata: {chunk_line('B')}\n\ndata: [DONE]", "AB", id="done-without-a-newline"),
        pytest.param(
            'data: {"choices":\ndata: [{"delta": {"content": "M"}}]}\n\ndata: [DONE]\n\n', "M", id="an-event-over-two-data-lines"
        ),
        pytest.param(
            f": hi\nretry: 3000\nid: 7\nevent: message\ndata: {chunk_line('A')}\n\n\n\n: bye\ndata:\n\ndata: [DONE]\n\n",
            "A",
            id="other-fields-and-empty-data",
        ),
        pytest.param(
            f"data: {chunk_line(SEPARATORS)}\n\ndata: [DONE]\n\n", SEPARATORS, id="unicode-line-separators-inside-the-text"
        ),
        pytest.param(f"{BOM}data: {chunk_line('A')}\n\ndata: [DONE]\n\n", "A", id="byte-order-mark"),
    ],
)
def test_sse_framing_edge_cases(body, expected):
    c, _ = make(raw_stream(body.encode()))
    assert "".join(c.stream(MSG)) == expected


def test_the_stream_stops_reading_at_done():
    c, _ = make(sse(*texts("x"), tail=b"data: [DONE]\n\ndata: " + b"{not json at all"))
    assert list(c.stream(MSG)) == ["x"]


def test_a_stream_event_that_is_not_json_is_a_content_error():
    c, _ = make(raw_stream(b"data: <html>\n\n"))
    assert error_of(lambda: list(c.stream(MSG))).kind == "content"


def test_a_stream_that_ends_without_finishing_is_a_transport_error():
    c, _ = make(sse(delta("half"), tail=b""))
    seen = []
    with pytest.raises(ProviderError) as caught:
        for piece in c.stream(MSG):
            seen.append(piece)
    assert seen == ["half"] and caught.value.kind == "transport"


def test_an_empty_stream_and_a_stream_cut_off_by_the_limit_are_content_errors():
    error = error_of(lambda: list(make(sse(delta(role="assistant"), delta(finish="stop")))[0].stream(MSG)))
    assert error.kind == "content" and "no text" in str(error)
    error = error_of(lambda: list(make(sse(delta(role="assistant"), delta(finish="length")))[0].stream(MSG)))
    assert error.kind == "content" and "thinking" in str(error)


def test_a_stream_that_hit_the_limit_after_some_text_is_not_an_error():
    c, _ = make(sse(delta("some"), delta(finish="length")))
    assert list(c.stream(MSG)) == ["some"] and c.last_finish_reason == "length"


@pytest.mark.parametrize(
    ("events", "kind"),
    [
        # OpenRouter reports a failure after the stream began as an event, with HTTP 200 already sent.
        (
            [
                {
                    "error": {"code": 502, "message": "Provider disconnected"},
                    "choices": [{"delta": {"content": ""}, "finish_reason": "error"}],
                }
            ],
            "server",
        ),
        ([{"error": {"code": 429, "message": "rate limited upstream"}}], "rate_limit"),
        ([{"choices": [{"delta": {"content": ""}, "finish_reason": "error"}]}], "server"),
        ([b'event: error\ndata: {"message": "overloaded"}\n\n'], "server"),
    ],
)
def test_an_error_event_in_the_stream_is_a_provider_error(events, kind):
    c, _ = make(sse(*events))
    assert error_of(lambda: list(c.stream(MSG))).kind == kind


def test_an_error_event_after_some_text_keeps_the_text_that_came():
    c, _ = make(sse(delta("fine so far"), {"error": {"code": 502, "message": "gone"}}))
    seen = []
    with pytest.raises(ProviderError) as caught:
        for piece in c.stream(MSG):
            seen.append(piece)
    assert seen == ["fine so far"] and caught.value.kind == "server"


def test_a_server_that_ignores_stream_and_sends_one_json_reply_is_still_read():
    c, _ = make(ok("<think>x</think>Whole reply", usage={"prompt_tokens": 4, "completion_tokens": 2}))
    assert list(c.stream(MSG)) == ["Whole reply"] and c.last_usage.output_tokens == 2
    c, _ = make(lambda: httpx.Response(200, json={"error": {"code": 429, "message": "slow"}}))
    assert error_of(lambda: list(c.stream(MSG))).kind == "rate_limit"


def test_closing_a_stream_early_closes_the_connection():
    body = Pieces([event(delta("one")), event(delta("two")), event(delta("three")), b"data: [DONE]\n\n"])
    c, _ = make(lambda: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=body))
    stream = c.stream(MSG)
    assert next(stream) == "one"
    stream.close()  # as a caller does when the user interrupts the spoken answer
    assert body.closed


def test_a_finished_stream_and_a_refused_one_release_their_connection():
    body = Pieces([event(d) for d in texts("x")] + [b"data: [DONE]\n\n"])
    c, _ = make(lambda: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=body))
    list(c.stream(MSG))
    assert body.closed
    refused = Pieces([b'{"error": "no"}'])
    c, _ = make(lambda: httpx.Response(500, stream=refused))
    error_of(lambda: list(c.stream(MSG)))
    assert refused.closed


def test_the_stream_request_is_sent_on_the_first_next_and_errors_come_from_it():
    c, rec = make(status(503, "down"))
    stream = c.stream(MSG)
    assert rec.requests == []
    assert error_of(lambda: next(stream)).kind == "server"


def test_ttft_and_usage_are_reset_at_the_start_of_each_stream():
    c, _ = make(sse(*texts("x")), status(500))
    list(c.stream(MSG))
    assert c.last_ttft_s is not None
    error_of(lambda: list(c.stream(MSG)))
    assert c.last_ttft_s is None and c.last_usage is None


# -- the client itself ----------------------------------------------------------------------------


def test_one_http_client_serves_every_call_so_the_connection_is_reused(monkeypatch):
    made, options = [], {}
    real = httpx.Client

    def counting(**kw):
        client = real(**kw)
        made.append(client)
        options.update(kw)
        return client

    monkeypatch.setattr(llm_module.httpx, "Client", counting)
    c, _ = make(reply=lambda request, body, n: sse(*texts("x"))() if body["stream"] else ok()())
    c.chat(MSG)
    list(c.stream(MSG))
    c.chat(MSG)
    assert len(made) == 1 and not made[0].is_closed
    assert options["limits"].keepalive_expiry > 5.0  # httpx's own 5 s would close the connection between spoken turns
    c.close()
    assert made[0].is_closed


def test_it_is_a_context_manager_that_closes():
    with make(ok())[0] as c:
        assert c.chat(MSG).text == "hello"
    assert c._client.is_closed


def test_several_threads_may_share_one_client():
    c, rec = make(ok("same"))
    results = []
    threads = [threading.Thread(target=lambda: results.append(c.chat(MSG).text)) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert results == ["same"] * 8 and len(rec.requests) == 8


# -- build_client ---------------------------------------------------------------------------------


def test_build_client_names_the_slot_and_layers_the_slot_options_over_the_providers(monkeypatch):
    rec = Recorder(scripted(ok("built")))
    real = httpx.Client
    monkeypatch.setattr(llm_module.httpx, "Client", lambda **kw: real(**{**kw, "transport": httpx.MockTransport(rec)}))
    spec = ProviderSpec(
        name="openrouter",
        kind="openai_compat",
        base_url=BASE,
        api_key_env="OPENROUTER_API_KEY",
        options={
            "extra_body": {"provider": {"sort": "latency"}},
            "reasoning_effort": "high",
            "token_param": "max_completion_tokens",
        },
    )
    c = build_client(spec, "deepseek/deepseek-v4.1-flash", KEY, {"reasoning_effort": "none"})

    assert isinstance(c, OpenAICompatLLM)
    assert (c.name, c.model) == ("openrouter:deepseek/deepseek-v4.1-flash", "deepseek/deepseek-v4.1-flash")
    result = c.chat(MSG, max_tokens=10)
    body = rec.bodies[0]
    assert body["model"] == "deepseek/deepseek-v4.1-flash" and body["provider"] == {"sort": "latency"}
    assert body["reasoning_effort"] == "none" and body["max_completion_tokens"] == 10 and "max_tokens" not in body
    assert result.provider == "openrouter:deepseek/deepseek-v4.1-flash" and spec.options["reasoning_effort"] == "high"


def test_build_client_without_a_key_or_options_and_with_a_missing_url():
    spec = ProviderSpec(name="local", kind="openai_compat", base_url="http://127.0.0.1:1234/v1")
    c = build_client(spec, "qwen", None, {})
    assert c.name == "local:qwen" and c.settings["token_param"] == "max_tokens"
    with pytest.raises(ValueError, match="base_url"):
        build_client(ProviderSpec(name="x", kind="openai_compat"), "m", KEY, {})


# -- the facade -----------------------------------------------------------------------------------


def llm_of(*clients: OpenAICompatLLM, policy=None, on_event=None) -> LLM:
    return LLM(Chain("llm", [Slot(c.name, c) for c in clients], policy, on_event=on_event))


def test_the_facade_fails_over_from_a_to_b_says_so_and_names_the_slot_that_answered():
    seen = []
    a, _ = make(status(503, "overloaded"), name="a:model-a")
    b, rec_b = make(ok("from b", usage={"prompt_tokens": 5, "completion_tokens": 1}), name="b:model-b")
    llm = llm_of(a, b, on_event=seen.append)

    result = llm.chat(MSG, max_tokens=20, schema=None)

    assert (result.text, result.provider) == ("from b", "b:model-b")
    assert result.model == "m" and result.usage.input_tokens == 5
    assert [(e.role, e.from_slot, e.to_slot, e.kind) for e in llm.chain.events] == [("llm", "a:model-a", "b:model-b", "server")]
    assert seen == list(llm.chain.events) and "503" in seen[0].reason
    assert rec_b.bodies[0]["max_tokens"] == 20  # the keyword arguments reach the client


def test_the_facade_exposes_its_chain():
    a, _ = make(ok(), name="a")
    llm = llm_of(a)
    assert isinstance(llm.chain, Chain) and llm.chain.names == ["a"]
    llm.chain.pin("a")
    assert llm.chat(MSG).provider == "a"


def test_the_slot_name_is_what_comes_back_even_if_the_client_was_named_otherwise():
    a, _ = make(ok(), name="client-name")
    llm = LLM(Chain("llm", [Slot("slot-name", a)]))
    assert llm.chat(MSG).provider == "slot-name"


def test_a_bad_request_is_raised_and_not_tried_on_the_next_provider():
    a, _ = make(status(400, "messages: field required"), name="a")
    b, rec_b = make(ok(), name="b")
    with pytest.raises(ProviderError) as caught:
        llm_of(a, b).chat(MSG)
    assert caught.value.kind == "bad_request" and rec_b.requests == []


def test_a_content_error_fails_over_too():
    a, _ = make(ok("", finish="length"), name="a")
    b, _ = make(ok("answer"), name="b")
    llm = llm_of(a, b)
    assert llm.chat(MSG).text == "answer"
    assert llm.chain.events[0].kind == "content"


def test_every_provider_failing_is_one_error_with_each_cause():
    a, _ = make(status(401, "no"), name="a")
    b, _ = make(httpx.ConnectError("x"), name="b")
    with pytest.raises(AllProvidersFailed) as caught:
        llm_of(a, b).chat(MSG)
    assert [(n, e.kind) for n, e in caught.value.errors] == [("a", "auth"), ("b", "transport")]


def test_a_json_chat_through_the_facade_fails_over_when_the_first_model_talks_instead():
    a, _ = make(ok("Well, I think so."), name="a")
    b, _ = make(ok(JSON_REPLY), name="b")
    result = llm_of(a, b).chat(MSG, schema=SCHEMA)
    assert (result.provider, result.text) == ("b", JSON_REPLY)


def test_a_slow_first_provider_is_raced_against_the_next_and_the_first_answer_wins():
    release = threading.Event()

    def slow(request, body, n):
        release.wait(5)
        return httpx.Response(200, json=completion("slow"))

    a, _ = make(reply=slow, name="a")
    b, _ = make(ok("fast"), name="b")
    llm = llm_of(a, b, policy=ChainPolicy(hedge_after_s=0.05))
    try:
        result = llm.chat(MSG)  # hedge=None: race because the policy says to
    finally:
        release.set()
    assert (result.text, result.provider) == ("fast", "b")
    assert [(e.from_slot, e.to_slot, e.kind) for e in llm.chain.events] == [("a", "b", "slow")]


def test_a_hedged_chat_with_a_failing_first_provider_just_fails_over():
    a, _ = make(status(500, "x"), name="a")
    b, _ = make(ok("B"), name="b")
    assert llm_of(a, b, policy=ChainPolicy(hedge_after_s=5)).chat(MSG, hedge=True).provider == "b"


def test_hedge_false_never_races_and_hedge_none_does_nothing_without_a_policy():
    def slowish(request, body, n):
        threading.Event().wait(0.2)
        return httpx.Response(200, json=completion("slow"))

    for policy, hedge in ((ChainPolicy(hedge_after_s=0.05), False), (None, None), (None, True)):
        a, _ = make(reply=slowish, name="a")
        b, rec_b = make(ok("fast"), name="b")
        llm = llm_of(a, b, policy=policy)
        assert llm.chat(MSG, hedge=hedge).text == "slow"
        assert rec_b.requests == [] and not llm.chain.events


def test_a_streaming_facade_fails_over_before_the_first_token_and_not_after():
    seen = []
    a, _ = make(status(503, "down"), name="a")
    b, _ = make(sse(*texts("x", "y")), name="b")
    llm = llm_of(a, b, on_event=seen.append)
    assert list(llm.stream(MSG, max_tokens=5)) == ["x", "y"]
    assert [(e.from_slot, e.to_slot, e.kind) for e in seen] == [("a", "b", "server")]

    broken, _ = make(sse(delta("x"), {"error": {"code": 502, "message": "gone"}}), name="a")
    good, _ = make(sse(*texts("y")), name="b")
    got = []
    with pytest.raises(ProviderError) as caught:
        for piece in llm_of(broken, good).stream(MSG):
            got.append(piece)
    assert got == ["x"] and caught.value.kind == "stream"


def test_a_role_only_first_delta_does_not_commit_the_chain_to_a_provider_that_then_fails():
    a, _ = make(sse(delta(role="assistant", content=""), {"error": {"code": 503, "message": "gone"}}), name="a")
    b, _ = make(sse(*texts("ok")), name="b")
    llm = llm_of(a, b)
    assert list(llm.stream(MSG)) == ["ok"] and llm.chain.events[0].from_slot == "a"


def _two_token_stream(held: list | None = None):
    """A facade stream over two tokens. Generators handed to `held` stay referenced, so only an explicit close ends them."""
    body = Pieces([event(delta("one")), event(delta("two")), b"data: [DONE]\n\n"])
    a, _ = make(lambda: httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=body), name="a")
    if held is not None:
        original = a.stream
        a.stream = lambda *args, **kwargs: held.append(original(*args, **kwargs)) or held[-1]
    return body, llm_of(a).stream(MSG)


def test_dropping_a_facade_stream_early_closes_the_connection():
    """The explicit path: whoever stops reading closes the stream, and the chain closes the client's own stream with it,
    though something else still holds that one, so the connection is not left to a collector."""
    held: list = []
    body, stream = _two_token_stream(held)
    with contextlib.closing(stream):
        assert next(stream) == "one"
        assert not body.closed  # still open while it is being read
    assert body.closed


def test_a_collected_facade_stream_closes_the_connection():
    """The best-effort path, for a caller that forgets to close: collection closes the generator. The collection is
    forced here, so the test does not depend on when an interpreter happens to run it."""
    body, stream = _two_token_stream()
    assert next(stream) == "one"
    del stream
    gc.collect()
    assert body.closed


# -- real HTTP, on loopback -----------------------------------------------------------------------


class Loopback:
    """An HTTP/1.1 server on 127.0.0.1 that hands each POST (handler, parsed body) to `handle` and counts the TCP
    connections it was given."""

    def __init__(self, handle):
        outer = self
        self.connections = 0

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"  # keep-alive

            def setup(self):
                outer.connections += 1
                super().setup()

            def do_POST(self):
                handle(self, json.loads(self.rfile.read(int(self.headers["Content-Length"]))))

            def log_message(self, format, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


def send_json(h, payload, code=200):
    data = json.dumps(payload).encode()
    h.send_response(code)
    h.send_header("Content-Type", "application/json")
    h.send_header("Content-Length", str(len(data)))
    h.end_headers()
    h.wfile.write(data)


def start_sse(h):
    h.send_response(200)
    h.send_header("Content-Type", "text/event-stream")
    h.send_header("Transfer-Encoding", "chunked")
    h.end_headers()


def send_chunk(h, data: bytes):
    h.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
    h.wfile.flush()


def end_chunks(h):
    h.wfile.write(b"0\r\n\r\n")
    h.wfile.flush()


@pytest.fixture
def loopback():
    started: list[Loopback] = []

    def start(handle) -> Loopback:
        started.append(Loopback(handle))
        return started[-1]

    yield start
    for server in started:
        server.close()


def answer_or_stream(h, body):
    if not body["stream"]:
        return send_json(h, completion("hi", usage={"prompt_tokens": 3, "completion_tokens": 1}))
    start_sse(h)
    for e in texts("a", "b"):
        send_chunk(h, event(e))
    send_chunk(h, b"data: [DONE]\n\n")
    end_chunks(h)


def test_over_real_http_one_connection_serves_every_call_chats_and_streams_alike(loopback):
    server = loopback(answer_or_stream)
    with OpenAICompatLLM("t:m", "m", server.url, KEY) as c:
        assert c.chat(MSG).text == "hi"
        assert "".join(c.stream(MSG)) == "ab"
        assert c.chat(MSG).text == "hi"
        assert "".join(c.stream(MSG)) == "ab"
    assert server.connections == 1  # the handshake is paid once; a stream read to [DONE] returns its connection too


def test_over_real_http_the_first_token_arrives_while_the_server_is_still_talking(loopback):
    release = threading.Event()

    def handle(h, body):
        start_sse(h)
        send_chunk(h, event(delta(role="assistant")))
        send_chunk(h, event(delta("Hel")))
        release.wait(5)  # the rest of the reply is held back until the client has shown it the first piece
        send_chunk(h, event(delta("lo")))
        send_chunk(h, event(delta(finish="stop")))
        send_chunk(h, b"data: [DONE]\n\n")
        end_chunks(h)

    server = loopback(handle)
    with OpenAICompatLLM("t:m", "m", server.url, KEY) as c:
        stream = c.stream(MSG, timeout=5)
        try:
            assert next(stream) == "Hel" and not release.is_set()
            assert c.last_ttft_s is not None and c.last_ttft_s < 4
        finally:
            release.set()
        assert list(stream) == ["lo"]


def test_over_real_http_a_server_that_goes_quiet_is_a_timeout_error(loopback):
    gone = threading.Event()

    def handle(h, body):
        gone.wait(5)

    server = loopback(handle)
    try:
        with OpenAICompatLLM("t:m", "m", server.url, KEY) as c:
            assert error_of(lambda: c.chat(MSG, timeout=0.2)).kind == "timeout"
            assert error_of(lambda: list(c.stream(MSG, timeout=0.2))).kind == "timeout"
    finally:
        gone.set()


def test_over_real_http_a_closed_port_is_a_transport_error():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]  # nothing listens here once the probe closes
    with OpenAICompatLLM("t:m", "m", f"http://127.0.0.1:{port}/v1", KEY) as c:
        assert error_of(lambda: c.chat(MSG)).kind == "transport"
        assert error_of(lambda: list(c.stream(MSG))).kind == "transport"


def test_over_real_http_a_server_that_hangs_up_mid_stream_is_a_transport_error_after_its_text(loopback):
    def handle(h, body):
        start_sse(h)
        send_chunk(h, event(delta("one ")))
        h.close_connection = True
        h.connection.shutdown(socket.SHUT_RDWR)  # no closing chunk: the body ends short

    server = loopback(handle)
    seen = []
    with OpenAICompatLLM("t:m", "m", server.url, KEY) as c, pytest.raises(ProviderError) as caught:
        for piece in c.stream(MSG):
            seen.append(piece)
    assert seen == ["one "] and caught.value.kind == "transport"


def test_over_real_http_the_error_status_and_body_come_through(loopback, caplog):
    caplog.set_level(logging.DEBUG)  # httpx and httpcore log at DEBUG, here with a real connection under them

    def handle(h, body):
        send_json(h, {"error": {"message": f"Incorrect API key provided: {h.headers['Authorization'][7:]}"}}, 401)

    server = loopback(handle)
    with OpenAICompatLLM("t:m", "m", server.url, KEY) as c:
        for call in (lambda: c.chat(MSG), lambda: list(c.stream(MSG))):
            error = error_of(call)
            assert error.kind == "auth" and error.status == 401 and "Incorrect API key" in str(error) and KEY not in str(error)
    assert KEY not in caplog.text and "Authorization" not in caplog.text

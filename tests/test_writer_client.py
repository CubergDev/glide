"""ChainWriter: the writer's one call, `messages.create`, answered by LLM facades. No network, no keys."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import httpx
import pytest

from glide.computer.calls import Calls, MeteredWriter
from glide.computer.writer import (
    Writer,
    WriterError,
    compose_answer,
    compose_text,
    compose_url,
    provider,
    spells_out_schema,
)
from glide.providers.base import ChatResult, Usage
from glide.providers.chain import Chain, ChainPolicy, Slot
from glide.providers.errors import AllProvidersFailed, ProviderError
from glide.providers.writer_client import ChainWriter

URL = '{"ok": true, "url": "https://example.com", "reason": ""}'
ANSWER = '{"achieved": true, "answer": "Sep 19 in Miami.", "focus": "", "question": ""}'
SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": False}
OUTPUT_CONFIG = {"format": {"type": "json_schema", "schema": SCHEMA}}
SECRET = "sk-secret-123"


class FakeFacade:
    """Stands in for an LLM facade: records each `chat` and answers with one text, or raises."""

    def __init__(self, name: str, reply: str = URL, usage: Usage | None = None, error: BaseException | None = None):
        self.name = name
        self.reply = reply
        self.usage = usage or Usage()
        self.error = error
        self.api_key = SECRET  # a facade may well hold a key; nothing of ChainWriter may show it
        self.calls: list[dict] = []

    def chat(self, messages, *, max_tokens=512, temperature=0.0, schema=None, timeout=None):
        self.calls.append(
            {
                "messages": list(messages),
                "max_tokens": max_tokens,
                "temperature": temperature,
                "schema": schema,
                "timeout": timeout,
            }
        )
        if self.error is not None:
            raise self.error
        return ChatResult(
            text=self.reply,
            usage=self.usage,
            provider=f"{self.name}:model-x",
            model="model-x",
            latency_s=0.0,
            finish_reason="stop",
        )


def pair(**kw) -> tuple[ChainWriter, FakeFacade, FakeFacade]:
    fast, smart = FakeFacade("fast", **kw), FakeFacade("smart", **kw)
    return ChainWriter(fast, smart), fast, smart


@pytest.fixture(autouse=True)
def models(clean_env):
    """Distinct ids for the two jobs, and none of the shell's writer settings."""
    for name in ("CLICKER_WRITER_REASONING", "CLICKER_ANSWER_REASONING"):
        clean_env.delenv(name, raising=False)
    clean_env.setenv("CLICKER_WRITER_MODEL", "writer-model")
    clean_env.setenv("CLICKER_ANSWER_MODEL", "answer-model")
    return clean_env


def create(writer: ChainWriter, **overrides):
    request = {
        "model": "writer-model",
        "max_tokens": 200,
        "system": "Be brief.",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
        **overrides,
    }
    return writer.messages.create(**request)


# -- the shape, both ways ------------------------------------------------------------------------------


def test_text_and_an_image_go_to_the_facade_as_chat_messages_and_the_text_comes_back_as_a_block():
    writer, fast, _ = pair(reply="hi there")
    reply = create(
        writer,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}},
                    {"type": "text", "text": "what is this?"},
                ],
            }
        ],
        output_config=OUTPUT_CONFIG,
    )
    (call,) = fast.calls
    assert call["messages"] == [
        {"role": "system", "content": "Be brief."},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                {"type": "text", "text": "what is this?"},
            ],
        },
    ]
    assert (call["max_tokens"], call["schema"]) == (200, SCHEMA)
    assert [(b.type, b.text) for b in reply.content] == [("text", "hi there")]
    assert reply.model == "model-x" and reply.provider == "fast:model-x"


def test_a_string_turn_stays_a_string_and_an_image_may_be_a_url():
    writer, fast, _ = pair()
    create(
        writer,
        system=[{"type": "text", "text": "One."}, {"type": "text", "text": "Two."}],
        messages=[
            {"role": "user", "content": "plain"},
            {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
            {"role": "user", "content": [{"type": "image", "source": {"type": "url", "url": "https://example.com/a.png"}}]},
        ],
    )
    system, user, assistant, image = fast.calls[0]["messages"]
    assert system == {"role": "system", "content": "One.\n\nTwo."}
    assert user == {"role": "user", "content": "plain"}
    assert assistant == {"role": "assistant", "content": [{"type": "text", "text": "ok"}]}
    assert image["content"] == [{"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}]


def test_no_schema_is_sent_unless_the_request_asks_for_a_json_schema():
    writer, fast, _ = pair()
    create(writer)
    create(writer, output_config={"format": {"type": "text"}})
    assert [c["schema"] for c in fast.calls] == [None, None]


def test_an_empty_reply_is_an_empty_text_block_and_the_writer_then_refuses_the_step():
    writer, _, _ = pair(reply="")
    reply = create(writer)
    assert [(b.type, b.text) for b in reply.content] == [("text", "")]
    with pytest.raises(WriterError, match="without usable JSON"):
        compose_url(writer, "open example", [])


def test_thinking_and_reasoning_are_accepted_and_the_timeout_is_passed_on():
    fast, smart = FakeFacade("fast"), FakeFacade("smart")
    writer = ChainWriter(fast, smart, timeout=12.5)
    create(writer, thinking={"type": "disabled"}, reasoning="low")
    assert fast.calls[0]["timeout"] == 12.5 and fast.calls[0]["temperature"] == 0.0
    assert "thinking" not in fast.calls[0] and "reasoning" not in fast.calls[0]


def test_a_request_the_chat_api_has_no_form_for_is_a_bad_request_and_reaches_no_facade():
    writer, fast, smart = pair()
    bad = {"role": "user", "content": [{"type": "tool_use", "id": "t", "name": "n", "input": {}}]}
    with pytest.raises(ProviderError) as caught:
        create(writer, messages=[bad])
    assert caught.value.kind == "bad_request" and "tool_use" in str(caught.value)
    half_image = {"role": "user", "content": [{"type": "image", "source": {"media_type": "image/png"}}]}
    with pytest.raises(ProviderError, match="image block"):
        create(writer, messages=[half_image])
    assert fast.calls == smart.calls == []


def test_a_keyword_the_writer_does_not_send_is_refused_not_dropped():
    writer, _, _ = pair()
    with pytest.raises(TypeError):
        create(writer, top_k=3)


# -- routing -------------------------------------------------------------------------------------------


def test_the_answer_model_goes_to_smart_and_any_other_model_to_fast():
    writer, fast, smart = pair()
    create(writer, model="answer-model")
    create(writer, model="writer-model")
    create(writer, model="some-other-model")
    assert (len(fast.calls), len(smart.calls)) == (2, 1)


def test_the_answer_model_is_read_at_each_call_and_not_when_the_writer_is_built(models):
    writer, fast, smart = pair()
    create(writer, model="answer-model")
    models.setenv("CLICKER_ANSWER_MODEL", "newer-answer-model")
    create(writer, model="answer-model")
    create(writer, model="newer-answer-model")
    assert (len(fast.calls), len(smart.calls)) == (1, 2)


def test_when_both_models_have_the_same_id_every_request_goes_to_smart(models):
    models.setenv("CLICKER_WRITER_MODEL", "one-model")
    models.setenv("CLICKER_ANSWER_MODEL", "one-model")
    writer, fast, smart = pair()
    compose_url(writer, "open example", [])
    assert (len(fast.calls), len(smart.calls)) == (0, 1)


def test_the_writer_calls_route_by_what_they_are_for(screen, make_item):
    fast, smart = FakeFacade("fast"), FakeFacade("smart", reply=ANSWER)
    writer = ChainWriter(fast, smart)
    assert compose_url(writer, "open example", []) == "https://example.com"
    answer = compose_answer(writer, "find the concert", screen, [make_item(0, "SEP 19")], [], "the goal is achieved")
    assert answer.text == "Sep 19 in Miami." and answer.achieved
    assert (len(fast.calls), len(smart.calls)) == (1, 1)
    image, text = smart.calls[0]["messages"][1]["content"]
    assert image["image_url"]["url"].startswith("data:image/png;base64,")
    assert json.loads(text["text"])["screen_text_in_reading_order"] == ["SEP 19"]
    assert smart.calls[0]["schema"]["required"] == ["achieved", "answer", "focus", "question"]


# -- what writer.py needs of a writer ------------------------------------------------------------------


@pytest.mark.parametrize(
    "error",
    [
        ProviderError("provider answered 500: boom", kind="server"),
        ProviderError("provider answered 400: bad", kind="bad_request"),
        AllProvidersFailed("llm", [("a", ProviderError("down", kind="timeout"))]),
    ],
    ids=["server", "bad_request", "exhausted"],
)
def test_a_provider_error_refuses_the_step_instead_of_crashing_the_run(error, screen):
    for call in (
        lambda w: compose_url(w, "open example", []),
        lambda w: compose_text(w, "type hello", screen, [], []),
    ):
        writer = ChainWriter(FakeFacade("fast", error=error), FakeFacade("smart", error=error))
        with pytest.raises(WriterError, match="the request failed") as caught:
            call(writer)
        assert caught.value.__cause__ is error


def test_a_bug_in_a_facade_is_not_swallowed_as_a_writer_error():
    writer = ChainWriter(FakeFacade("fast", error=KeyError("bug")), FakeFacade("smart"))
    with pytest.raises(KeyError):
        compose_url(writer, "open example", [])


def test_a_failover_inside_the_facade_is_an_event_on_its_chain_and_the_writer_just_gets_the_answer():
    class Down:
        def chat(self, messages, **kw):
            raise ProviderError("a answered 503", kind="server", provider="a")

    class Up:
        def chat(self, messages, **kw):
            return ChatResult(text=URL, usage=Usage(), provider="b", model="m", latency_s=0.0)

    class Facade:
        """What an LLM facade is: a chain of clients behind `.chat`."""

        def __init__(self, *slots):
            self.chain = Chain("llm", [Slot(n, c) for n, c in slots], ChainPolicy())

        def chat(self, messages, **kw):
            return self.chain.call(lambda slot: slot.client.chat(messages, **kw))

    fast = Facade(("a", Down()), ("b", Up()))
    assert compose_url(ChainWriter(fast, fast), "open example", []) == "https://example.com"
    assert [(e.from_slot, e.to_slot, e.kind) for e in fast.chain.events] == [("a", "b", "server")]

    dead = Facade(("a", Down()))
    with pytest.raises(WriterError, match="every llm provider failed"):
        compose_url(ChainWriter(dead, dead), "open example", [])


def test_provider_names_the_chain_and_not_a_vendor():
    writer, _, _ = pair()
    assert isinstance(writer.base_url, httpx.URL)
    assert (writer.base_url.scheme, writer.base_url.host, writer.base_url.port) == ("glide", "writer-chain", None)
    assert provider(writer).startswith("glide://writer-chain (anthropic API)  models: writer-model (writing), answer-model")


def test_the_schema_is_spelled_out_in_the_prompt_for_a_chain_whether_or_not_it_is_wrapped():
    for wrap in (lambda w: w, lambda w: MeteredWriter(w, Calls())):
        writer, fast, _ = pair()
        compose_url(wrap(writer), "open example", [])
        assert '"required": ["ok", "url", "reason"]' in fast.calls[0]["messages"][0]["content"]


def test_only_a_chain_spells_the_schema_out_among_the_writers_that_ask_anthropic():
    plain = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: None))
    assert spells_out_schema(ChainWriter(FakeFacade("a"), FakeFacade("b"))) is True
    assert spells_out_schema(MeteredWriter(ChainWriter(FakeFacade("a"), FakeFacade("b")), Calls())) is True
    assert spells_out_schema(plain) is False
    assert spells_out_schema(MeteredWriter(plain, Calls())) is False


def test_a_chain_writer_is_a_writer():
    assert ChainWriter in Writer.__value__.__args__


# -- usage ---------------------------------------------------------------------------------------------


def test_usage_comes_back_in_the_messages_shape_with_cached_tokens_apart():
    writer, _, _ = pair(usage=Usage(input_tokens=1200, output_tokens=30, cached_input_tokens=1000))
    usage = create(writer).usage
    assert (usage.input_tokens, usage.cache_read_input_tokens, usage.cache_creation_input_tokens, usage.output_tokens) == (
        200,
        1000,
        0,
        30,
    )


def test_a_facade_that_reports_more_cached_than_total_never_gives_a_negative_count():
    writer, _, _ = pair(usage=Usage(input_tokens=10, output_tokens=1, cached_input_tokens=50))
    assert create(writer).usage.input_tokens == 0


def test_the_metered_writer_counts_a_chain_request_in_the_requested_models_row():
    writer, _, _ = pair(usage=Usage(input_tokens=1200, output_tokens=30, cached_input_tokens=1000))
    calls = Calls()
    compose_url(MeteredWriter(writer, calls), "open example", [])
    (model, usage) = next(iter(calls.usage.items()))
    assert model == "writer-model"
    assert (usage.requests, usage.input_tokens, usage.cached_input_tokens, usage.output_tokens) == (1, 200, 1000, 30)
    assert calls.count["writer"] == 1


def test_a_facade_that_reports_no_usage_still_counts_the_request():
    writer, _, _ = pair()
    calls = Calls()
    compose_url(MeteredWriter(writer, calls), "open example", [])
    (usage,) = calls.usage.values()
    assert (usage.requests, usage.input_tokens, usage.cached_input_tokens, usage.output_tokens) == (1, 0, 0, 0)


# -- keys ----------------------------------------------------------------------------------------------


def test_a_key_held_by_a_facade_never_shows_in_a_repr_a_url_an_error_or_a_log(caplog, screen):
    caplog.set_level(logging.DEBUG)
    error = ProviderError("provider answered 500: upstream broke", kind="server", provider="fast:model-x")
    writer = ChainWriter(FakeFacade("fast", error=error), FakeFacade("smart", error=error))
    shown = [repr(writer), str(writer.base_url), provider(writer), repr(writer.messages)]
    with pytest.raises(WriterError) as caught:
        compose_url(writer, "open my private example", [])
    with pytest.raises(WriterError) as caught_text:
        compose_text(writer, "type hello", screen, [], [])
    shown += [str(caught.value), str(caught_text.value), repr(caught.value.__cause__), caplog.text]
    assert not any(SECRET in text for text in shown)
    # Nor is the request body in the error: only what the provider said.
    assert str(caught.value) == "the request failed: provider answered 500: upstream broke"

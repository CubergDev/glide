"""ChainWriter: the writer's one call, `generate`, answered by LLM facades. No network, no keys."""

from __future__ import annotations

import json
import logging

import pytest

from glide.computer.calls import Calls, MeteredWriter
from glide.computer.control import RunControl
from glide.computer.generation import GenerationError, GenerationRequest, GenerationUnavailable
from glide.computer.models import Abort
from glide.computer.writer import (
    WriterError,
    WriterUnavailable,
    compose_answer,
    compose_text,
    compose_url,
    provider,
)
from glide.providers.base import ChatResult, Usage
from glide.providers.chain import Chain, ChainPolicy, Slot
from glide.providers.errors import AllProvidersFailed, ProviderError
from glide.providers.writer_client import ChainWriter

URL = '{"ok": true, "url": "https://example.com", "reason": ""}'
ANSWER = '{"achieved": true, "answer": "Sep 19 in Miami.", "focus": "", "question": ""}'
SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": False}
SECRET = "sk-secret-123"


class FakeFacade:
    """Stands in for an LLM facade: records each `chat` and answers with one text, or raises."""

    def __init__(
        self, name: str, reply: str = URL, usage: Usage | None = None, error: BaseException | None = None, finish="stop"
    ):
        self.name = name
        self.reply = reply
        self.usage = usage or Usage()
        self.error = error
        self.finish = finish
        self.api_key = SECRET  # a facade may well hold a key; nothing of ChainWriter may show it
        self.calls: list[dict] = []

    def chat(self, messages, *, max_tokens=512, temperature=0.0, schema=None, timeout=None, exact_json=False):
        self.calls.append(
            {
                "exact_json": exact_json,
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
            finish_reason=self.finish,
        )


def pair(**kw) -> tuple[ChainWriter, FakeFacade, FakeFacade]:
    fast, smart = FakeFacade("fast", **kw), FakeFacade("smart", **kw)
    return ChainWriter(fast, smart), fast, smart


def generate(writer: ChainWriter, **overrides):
    request = {"model": "", "instructions": "Be brief.", "text": "hello", "schema": SCHEMA, "max_tokens": 200, **overrides}
    return writer.generate(GenerationRequest(**request))


# -- the shape, both ways ------------------------------------------------------------------------------


def test_text_and_an_image_go_to_the_facade_as_chat_messages_and_the_text_comes_back():
    writer, fast, _ = pair(reply="hi there")
    result = generate(writer, text="what is this?", image=b"\x89PNG-bytes")
    (call,) = fast.calls
    system, user = call["messages"]
    assert system == {"role": "system", "content": "Be brief."}
    image, text = user["content"]
    assert image["image_url"]["url"].startswith("data:image/png;base64,") and text == {
        "type": "text",
        "text": "<data>\nwhat is this?\n</data>",
    }  # the data is fenced
    assert (call["max_tokens"], call["schema"], call["temperature"]) == (200, SCHEMA, 0.0)
    assert call["exact_json"] is True  # a reply is read whole, never picked out of prose
    assert (result.text, result.model, result.completed) == ("hi there", "model-x", True)


def test_a_text_only_request_has_no_image_part_and_no_instructions_means_no_system_turn():
    writer, fast, _ = pair()
    generate(writer, instructions="")
    (user,) = fast.calls[0]["messages"]
    assert user["content"] == [{"type": "text", "text": "<data>\nhello\n</data>"}]


def test_an_empty_reply_is_empty_text_and_the_writer_then_refuses_the_step():
    writer, _, _ = pair(reply="")
    assert generate(writer).text == ""
    with pytest.raises(WriterError, match="without usable JSON"):
        compose_url(writer, "open example", [])


@pytest.mark.parametrize("finish", ["length", "content_filter", "error"])
def test_a_reply_that_was_cut_short_is_not_complete_and_the_writer_refuses_it(finish):
    writer, _, _ = pair(finish=finish)
    assert generate(writer).completed is False
    with pytest.raises(WriterError, match="did not complete"):
        compose_url(writer, "open example", [])


def test_the_request_deadline_is_passed_on_and_capped_by_the_configured_limits():
    fast, smart = FakeFacade("fast"), FakeFacade("smart")
    writer = ChainWriter(fast, smart, timeout=12.5, deadlines={"fast": 5})
    generate(writer, deadline_s=30)
    generate(writer, deadline_s=3)
    assert [c["timeout"] for c in fast.calls] == [5, 3]
    generate(writer, deadline_s=30, role="recovery")
    assert smart.calls[0]["timeout"] == 12.5


# -- routing -------------------------------------------------------------------------------------------


def test_each_role_goes_to_its_own_facade_and_never_by_model_id():
    facades = {name: FakeFacade(name) for name in ("fast", "smart", "planner", "research")}
    writer = ChainWriter(facades["fast"], facades["smart"], planner=facades["planner"], research=facades["research"])
    for role in ("writer", "task_routing", "recovery", "planner", "research_supervisor", "research_verification"):
        generate(writer, role=role, model="same-id-for-all")
    assert {name: len(f.calls) for name, f in facades.items()} == {"fast": 2, "smart": 1, "planner": 1, "research": 2}


def test_planner_and_research_default_to_the_smart_facade():
    writer, _, smart = pair()
    generate(writer, role="planner")
    generate(writer, role="research_supervisor")
    assert len(smart.calls) == 2


def test_a_role_nobody_serves_is_refused_and_reaches_no_facade():
    writer, fast, smart = pair()
    with pytest.raises(GenerationError, match="no provider chain serves"):
        generate(writer, role="nobody")
    assert fast.calls == smart.calls == []


def test_the_writer_calls_route_by_what_they_are_for(screen, make_item):
    fast, smart = FakeFacade("fast"), FakeFacade("smart", reply=ANSWER)
    writer = ChainWriter(fast, smart)
    assert compose_url(writer, "open example", []) == "https://example.com"
    answer = compose_answer(writer, "find the concert", screen, [make_item(0, "SEP 19")], [], "the goal is achieved")
    assert answer.text == "Sep 19 in Miami." and answer.achieved
    assert (len(fast.calls), len(smart.calls)) == (1, 1)
    image, text = smart.calls[0]["messages"][1]["content"]
    assert image["image_url"]["url"].startswith("data:image/png;base64,")
    assert json.loads(text["text"].removeprefix("<data>\n").removesuffix("\n</data>"))["screen_text_in_reading_order"] == [
        "SEP 19"
    ]
    assert smart.calls[0]["schema"]["required"] == ["achieved", "answer", "focus", "question"]


# -- cancelling and failing ----------------------------------------------------------------------------


def test_a_cancelled_run_sends_nothing_and_drops_an_answer_that_arrives_late():
    writer, fast, _ = pair()
    control = RunControl()
    control.cancel()
    with pytest.raises(Abort):
        writer.generate(GenerationRequest("", "x", "y", SCHEMA), control)
    assert fast.calls == []

    late = RunControl()

    class CancelsWhileAsked(FakeFacade):
        def chat(self, messages, **kw):
            late.cancel()
            return super().chat(messages, **kw)

    with pytest.raises(Abort):
        ChainWriter(CancelsWhileAsked("fast"), FakeFacade("smart")).generate(GenerationRequest("", "x", "y", SCHEMA), late)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (ProviderError("provider answered 500: boom", kind="server"), WriterUnavailable),
        (ProviderError("provider has no credit", kind="auth"), WriterUnavailable),
        (ProviderError("connection dropped", kind="transport"), WriterUnavailable),
        (AllProvidersFailed("llm", [("a", ProviderError("down", kind="timeout"))]), WriterUnavailable),
        (ProviderError("provider answered 400: bad", kind="bad_request"), WriterError),
    ],
    ids=["server", "auth", "transport", "exhausted", "bad_request"],
)
def test_a_provider_error_is_a_writer_error_and_only_a_bad_request_leaves_the_run_going(error, expected, screen):
    for call in (
        lambda w: compose_url(w, "open example", []),
        lambda w: compose_text(w, "type hello", screen, [], []),
    ):
        writer = ChainWriter(FakeFacade("fast", error=error), FakeFacade("smart", error=error))
        with pytest.raises(WriterError) as caught:
            call(writer)
        assert type(caught.value) is expected
        assert isinstance(caught.value, GenerationUnavailable) is (expected is WriterUnavailable)
        assert str(caught.value).startswith(str(error))


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
            return ChatResult(text=URL, usage=Usage(), provider="b", model="m", latency_s=0.0, finish_reason="stop")

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
    with pytest.raises(WriterUnavailable, match="every llm provider failed"):
        compose_url(ChainWriter(dead, dead), "open example", [])


def test_provider_names_the_slots_and_not_a_vendor_or_a_url():
    class Facade(FakeFacade):
        chain = Chain("llm", [Slot("openai:gpt-a", object())], ChainPolicy())

    writer = ChainWriter(Facade("f"), Facade("s"))
    line = provider(writer)
    assert "fast: openai:gpt-a" in line and "http" not in line
    assert provider(MeteredWriter(writer, Calls())) == line


# -- usage ---------------------------------------------------------------------------------------------


def test_usage_passes_through_with_cached_tokens_already_apart():
    writer, _, _ = pair(usage=Usage(input_tokens=200, output_tokens=30, cached_input_tokens=1000))
    usage = generate(writer).usage
    assert (usage.input_tokens, usage.cached_input_tokens, usage.output_tokens) == (200, 1000, 30)


def test_the_metered_writer_counts_a_chain_request_in_the_row_of_the_model_that_answered():
    writer, _, _ = pair(usage=Usage(input_tokens=200, output_tokens=30, cached_input_tokens=1000))
    calls = Calls()
    compose_url(MeteredWriter(writer, calls), "open example", [])
    (model, usage) = next(iter(calls.usage.items()))
    assert model == "model-x"
    assert (usage.requests, usage.input_tokens, usage.cached_input_tokens, usage.output_tokens) == (1, 200, 1000, 30)
    assert calls.count["writer"] == 1


def test_a_facade_that_reports_no_usage_still_counts_the_request():
    writer, _, _ = pair()
    calls = Calls()
    compose_url(MeteredWriter(writer, calls), "open example", [])
    (usage,) = calls.usage.values()
    assert (usage.requests, usage.input_tokens, usage.cached_input_tokens, usage.output_tokens) == (1, 0, 0, 0)


# -- keys ----------------------------------------------------------------------------------------------


def test_a_key_held_by_a_facade_never_shows_in_a_repr_an_error_or_a_log(caplog, screen):
    caplog.set_level(logging.DEBUG)
    error = ProviderError("provider answered 500: upstream broke", kind="server", provider="fast:model-x")
    writer = ChainWriter(FakeFacade("fast", error=error), FakeFacade("smart", error=error))
    shown = [repr(writer), provider(writer)]
    with pytest.raises(WriterError) as caught:
        compose_url(writer, "open my private example", [])
    with pytest.raises(WriterError) as caught_text:
        compose_text(writer, "type hello", screen, [], [])
    shown += [str(caught.value), str(caught_text.value), repr(caught.value.__cause__), caplog.text]
    assert not any(SECRET in text for text in shown)
    assert not any("my private example" in str(e) for e in (caught.value, caught_text.value))  # the request is not in the error

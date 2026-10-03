"""Recorded provider failures, through the writer boundary: mock transports only, no network.

HANDOFF section 7: a spent account (429 `credit_balance_exhausted`) was shown as a generic error, and a connection
dropped part-way through a reply while the partial text was still a candidate answer. Both must end in a safe,
bodyless message, one request to the failing provider, a visible switch, and no partial text used or replayed.
"""

from __future__ import annotations

import json
import logging

import httpx
import pytest

from glide.computer.calls import Calls, MeteredWriter
from glide.computer.generation import GenerationRequest, GenerationUnavailable
from glide.computer.writer import WriterUnavailable, compose_url
from glide.providers.chain import Chain, Slot, SwitchEvent
from glide.providers.errors import ProviderError
from glide.providers.llm import LLM, OpenAICompatLLM
from glide.providers.writer_client import ChainWriter

KEY = "sk-test-SECRET-0123456789abcdef"
BODY_SENTINEL = "BODY-SENTINEL-do-not-show"
PROMPT_SENTINEL = "my private goal sentinel 8841"
GOOD = '{"ok": true, "url": "https://example.com", "reason": "from B"}'
PARTIAL = '{"ok": true, "url": "https://partial.example", "rea'
SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": False}


class Recorder:
    def __init__(self, reply):
        self.reply = reply
        self.requests: list[dict] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append({"headers": dict(request.headers), "body": json.loads(request.content)})
        return self.reply(request)


class CutBody(httpx.SyncByteStream):
    """A 200 body that arrives in part and then the connection resets."""

    def __init__(self, first: bytes):
        self.first = first
        self.closed = False

    def __iter__(self):
        yield self.first
        raise httpx.ReadError("connection reset by peer")

    def close(self):
        self.closed = True


def slot(name: str, reply) -> tuple[OpenAICompatLLM, Recorder]:
    recorder = Recorder(reply)
    client = OpenAICompatLLM(name, "m", "https://llm.example.test/v1", KEY, transport=httpx.MockTransport(recorder))
    return client, recorder


def writer_over(*clients: OpenAICompatLLM, events: list | None = None) -> ChainWriter:
    chain = Chain("llm.fast", [Slot(c.name, c) for c in clients], on_event=(events.append if events is not None else None))
    llm = LLM(chain)
    return ChainWriter(llm, llm)


def credit_exhausted(request: httpx.Request) -> httpx.Response:
    body = {
        "error": {
            "type": "billing_error",
            "code": "credit_balance_exhausted",
            "message": f"Your credit balance is too low. {BODY_SENTINEL}",
        }
    }
    return httpx.Response(429, json=body)


def answers(text: str):
    return lambda request: httpx.Response(
        200, json={"choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]}
    )


def dropped_mid_body(request: httpx.Request) -> httpx.Response:
    start = b'{"choices": [{"index": 0, "message": {"role": "assistant", "content": ' + json.dumps(PARTIAL).encode()
    return httpx.Response(200, headers={"content-type": "application/json"}, stream=CutBody(start))


def shown(error: BaseException) -> str:
    return " | ".join([str(error), repr(error), repr(error.__cause__)])


# -- 429 credit_balance_exhausted ---------------------------------------------------------------------------


def test_a_spent_account_is_one_request_and_a_safe_message_with_no_body(caplog):
    caplog.set_level(logging.DEBUG)
    client, recorder = slot("a:model-a", credit_exhausted)
    writer = writer_over(client)

    with pytest.raises(WriterUnavailable) as caught:
        compose_url(MeteredWriter(writer, Calls()), PROMPT_SENTINEL, [])

    assert len(recorder.requests) == 1  # a spent account is not retried, here or anywhere
    assert isinstance(caught.value, GenerationUnavailable) and caught.value.halt is True
    text = shown(caught.value) + caplog.text
    assert "no credit or quota left" in str(caught.value)
    for secret in (BODY_SENTINEL, KEY, PROMPT_SENTINEL, "billing_error", "Your credit balance"):
        assert secret not in text


def test_a_spent_account_is_an_auth_error_that_rests_the_slot_and_another_provider_answers():
    events: list[SwitchEvent] = []
    a, rec_a = slot("a:model-a", credit_exhausted)
    b, rec_b = slot("b:model-b", answers(GOOD))
    writer = writer_over(a, b, events=events)

    result = writer.generate(GenerationRequest("", "i", "t", SCHEMA))

    assert json.loads(result.text)["reason"] == "from B"
    assert (len(rec_a.requests), len(rec_b.requests)) == (1, 1)
    assert [(e.from_slot, e.to_slot, e.kind) for e in events] == [("a:model-a", "b:model-b", "auth")]
    assert BODY_SENTINEL not in events[0].reason
    writer.generate(GenerationRequest("", "i", "t", SCHEMA))
    assert len(rec_a.requests) == 1  # the spent slot rests for the long auth cooldown: no third request to it


@pytest.mark.parametrize("status", [402, 429])
def test_the_adapter_maps_a_spent_account_whichever_status_says_so(status):
    reply = lambda request: httpx.Response(status, json={"error": {"code": "insufficient_quota", "message": BODY_SENTINEL}})  # noqa: E731
    client, _ = slot("a:model-a", reply)
    with pytest.raises(ProviderError) as caught:
        client.chat([{"role": "user", "content": "hi"}])
    assert (caught.value.kind, caught.value.status) == ("auth", status) and BODY_SENTINEL not in str(caught.value)


def test_an_ordinary_rate_limit_is_still_a_rate_limit_that_clears():
    client, _ = slot("a:model-a", lambda request: httpx.Response(429, text="slow down", headers={"retry-after": "3"}))
    with pytest.raises(ProviderError) as caught:
        client.chat([{"role": "user", "content": "hi"}])
    assert (caught.value.kind, caught.value.retry_after) == ("rate_limit", 3.0)


# -- a connection that closes mid-reply -------------------------------------------------------------------


def test_a_reply_cut_off_mid_body_is_a_transport_error_with_no_partial_text_and_no_replay():
    events: list[SwitchEvent] = []
    client, recorder = slot("a:model-a", dropped_mid_body)
    writer = writer_over(client, events=events)

    with pytest.raises(WriterUnavailable) as caught:
        compose_url(writer, PROMPT_SENTINEL, [])

    assert len(recorder.requests) == 1  # never replayed: the outcome of that request is unknown
    assert "partial.example" not in shown(caught.value)
    assert [(e.from_slot, e.to_slot, e.kind) for e in events] == [("a:model-a", None, "transport")]  # visible, and last


def test_a_reply_cut_off_mid_body_fails_over_visibly_and_only_the_next_slots_text_is_used():
    events: list[SwitchEvent] = []
    a, rec_a = slot("a:model-a", dropped_mid_body)
    b, rec_b = slot("b:model-b", answers(GOOD))
    writer = writer_over(a, b, events=events)

    result = writer.generate(GenerationRequest("", "i", "t", SCHEMA))

    assert result.text == GOOD and "partial" not in result.text
    assert (len(rec_a.requests), len(rec_b.requests)) == (1, 1)
    assert [(e.from_slot, e.to_slot, e.kind) for e in events] == [("a:model-a", "b:model-b", "transport")]
    assert "partial.example" not in events[0].reason


def test_a_writer_never_streams_so_a_reply_is_used_whole_or_not_at_all():
    a, rec_a = slot("a:model-a", answers(GOOD))
    writer_over(a).generate(GenerationRequest("", "i", "t", SCHEMA))
    assert rec_a.requests[0]["body"]["stream"] is False


# -- adapters never put a key, a header or a body in a message -----------------------------------------


def echoing(status: int):
    def reply(request: httpx.Request) -> httpx.Response:
        sent = json.loads(request.content)
        echoed = {
            "error": {
                "message": f"bad request {BODY_SENTINEL}",
                "request_body": sent,  # a server that quotes the whole request back
                "authorization": request.headers.get("authorization"),
                "headers": dict(request.headers),
            }
        }
        return httpx.Response(status, json=echoed)

    return reply


@pytest.mark.parametrize("status", [400, 401, 403, 404, 408, 500, 503])
def test_no_message_carries_the_key_the_headers_or_the_request_when_a_server_echoes_them(status, caplog):
    caplog.set_level(logging.DEBUG)
    for call in (
        lambda c: c.chat([{"role": "user", "content": PROMPT_SENTINEL}]),
        lambda c: list(c.stream([{"role": "user", "content": PROMPT_SENTINEL}])),
    ):
        client, _ = slot("a:model-a", echoing(status))
        with pytest.raises(ProviderError) as caught:
            call(client)
        text = shown(caught.value) + caplog.text
        assert KEY not in text and PROMPT_SENTINEL not in text and "Bearer" not in text and "authorization" not in text.lower()


def test_the_writer_boundary_carries_that_cleanliness_through_to_its_own_error(caplog):
    caplog.set_level(logging.DEBUG)
    client, _ = slot("a:model-a", echoing(500))
    with pytest.raises(WriterUnavailable) as caught:
        compose_url(writer_over(client), PROMPT_SENTINEL, [])
    text = shown(caught.value) + caplog.text
    assert KEY not in text and PROMPT_SENTINEL not in text and "Bearer" not in text


def test_a_reply_without_usable_json_never_repeats_what_the_provider_wrote():
    client, _ = slot("a:model-a", answers("I will not help with " + BODY_SENTINEL))
    with pytest.raises(Exception) as caught:
        compose_url(writer_over(client), "open example", [])
    assert BODY_SENTINEL not in shown(caught.value)

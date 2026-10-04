"""No provider text reaches an error message with the request or the key in it, whichever way the provider reports it.

A server may report a failure as a status, inside a 200 body, in a stream event, as a refusal, or (the TypeSafe SDK)
as an exception that carries its reply. Each path is cut the same way: the key, the quoted request and anything that
looks like an encoded blob (errors.py). The prompt here is private text no log may hold.
"""

from __future__ import annotations

import json

import httpx
import pytest
from test_llm import KEY, delta, event, make, ok, sse, status
from test_tts import caught_error, eleven, failing

from glide.providers.errors import ProviderError

PROMPT = "Summarise my private bank statement for March, account 12345678"
MESSAGES = [{"role": "system", "content": "You are terse."}, {"role": "user", "content": PROMPT}]
BLOB = "QUJD" * 40


def shown(error: BaseException) -> str:
    return " ".join([str(error), repr(error), repr(error.args)])


def clean(error: ProviderError) -> None:
    text = shown(error)
    assert PROMPT not in text and json.dumps(PROMPT)[1:-1] not in text, text
    assert KEY not in text and BLOB not in text, text


def chat_error(*steps) -> ProviderError:
    client, _ = make(*steps)
    with pytest.raises(ProviderError) as caught:
        client.chat(MESSAGES)
    return caught.value


def stream_error(*steps) -> ProviderError:
    client, _ = make(*steps)
    with pytest.raises(ProviderError) as caught:
        list(client.stream(MESSAGES))
    return caught.value


ECHO = f"rejected: {PROMPT} with key {KEY} and {BLOB}"


def completion_with(**choice) -> dict:
    return {"choices": [{"index": 0, **choice}]}


@pytest.mark.parametrize(
    "body",
    [
        {"error": {"code": 400, "message": ECHO}},  # an error with a status, inside a 200
        {"error": {"code": 429, "message": ECHO}},
        {"error": {"message": ECHO}},  # an error with no status
        {"error": ECHO},
        completion_with(error={"message": ECHO}),  # an error on the choice
        completion_with(error=ECHO, message={"content": "x"}),
        completion_with(finish_reason="error", message={"content": ECHO}),  # no message to quote: the fixed words
        completion_with(message={"content": "", "refusal": ECHO}, finish_reason="stop"),  # a refusal that quotes
        completion_with(message={"content": "", "refusal": json.dumps(ECHO)[1:-1]}, finish_reason="stop"),
    ],
)
def test_an_error_inside_a_200_reply_never_repeats_the_request_or_the_key(body):
    clean(chat_error(lambda: httpx.Response(200, json=body)))


@pytest.mark.parametrize(
    "events",
    [
        [{"error": {"code": 502, "message": ECHO}}],
        [{"error": {"message": ECHO}}],
        [delta("x"), {"error": ECHO}],
        [{"choices": [{"index": 0, "delta": {}, "error": {"message": ECHO}}]}],
        [{"choices": [{"index": 0, "delta": {"refusal": ECHO}, "finish_reason": "stop"}]}],
    ],
)
def test_an_error_inside_a_stream_never_repeats_the_request_or_the_key(events):
    clean(stream_error(sse(*events)))


def test_a_named_error_event_never_repeats_the_request_or_the_key():
    raw = b"event: error\n" + event({"message": ECHO})
    clean(stream_error(sse(raw, tail=b"")))


def test_a_status_reply_never_repeats_the_request_or_the_key():
    clean(chat_error(status(400, {"error": {"message": ECHO}})))
    clean(chat_error(status(500, ECHO)))
    clean(stream_error(status(400, {"error": {"message": ECHO}})))


def test_an_ordinary_reply_still_comes_through():
    client, _ = make(ok("fine"))
    assert client.chat(MESSAGES).text == "fine"


# -- the speech adapters: a body of any shape --------------------------------------------------------

SPOKEN = "Please read this private sentence aloud now."


@pytest.mark.parametrize(
    "body",
    [
        {"unexpected": [SPOKEN, {"echo": SPOKEN}]},  # a shape nothing knows
        [SPOKEN, SPOKEN],
        {"detail": [{"loc": ["body", "text"], "msg": "bad", "input": SPOKEN}]},
        {"detail": {"message": f"cannot say: {SPOKEN}"}},
        SPOKEN,
        f"<html>{SPOKEN}</html>",
        {"error": {"message": SPOKEN}},
        {"error": {"message": json.dumps(SPOKEN)}},
        {"payload": {"input": SPOKEN}},
        {"text": SPOKEN.replace(" ", "  ")},  # a quote the verbatim cut would miss
        "cannot say: " + SPOKEN.replace(" ", "  "),  # plain text, spaces doubled
        "cannot say:\n" + SPOKEN.replace(" ", "\n"),  # plain text, spaces as line breaks
        "cannot say: " + SPOKEN.upper(),  # plain text, other case
    ],
)
def test_a_speech_error_body_of_any_shape_never_repeats_the_spoken_text(body):
    raw = body.encode() if isinstance(body, str) else json.dumps(body).encode()
    error = caught_error(lambda: list(eleven(failing(500, raw)).stream(SPOKEN)))
    text = " ".join(shown(error).lower().split())
    assert "private sentence" not in text, text
    assert error.kind == "server" and error.status == 500


# -- the classifier's SDK client: an exception that carries its reply -----------------------------------

SCREEN = "Account 12345678 holds 9,400 euros and the password hint is blue-heron"


@pytest.mark.parametrize("bare", [False, True], ids=["adapter", "bare sdk client"])
@pytest.mark.parametrize(
    "body",
    [
        {"error": f"cannot classify: {SCREEN}"},
        {"detail": {"state": SCREEN}},
        {"error": {"message": f"bad state {SCREEN}"}},
        f"plain text: {SCREEN}",
    ],
)
def test_a_classifier_server_that_quotes_the_state_does_not_get_it_into_the_error(body, bare):
    import httpx2
    from test_classifier import Server, questions, typesafe

    from glide.providers.chain import Chain, Slot
    from glide.providers.classifier import ChainedClassifier

    def respond(request):
        return httpx2.Response(400, text=body) if isinstance(body, str) else httpx2.Response(400, json=body)

    client = typesafe(Server(respond))
    if bare:  # the SDK's own client in a chain slot, with no adapter in between
        client = ChainedClassifier(Chain("classifier", [Slot("raw", client._client)]))
    with pytest.raises(ProviderError) as caught:
        client.system_one(state={"screen": SCREEN}, questions=questions())
    text = shown(caught.value)
    assert "12345678" not in text and "blue-heron" not in text, text
    assert caught.value.kind == "bad_request" and caught.value.status == 400

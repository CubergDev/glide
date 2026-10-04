"""errors.py's one scrubber and http.py's shared request helpers, tested on their own (the adapters are tested in theirs)."""

from __future__ import annotations

import json
import threading
import traceback
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

import httpx
import pytest

from glide.computer.control import RunControl, controlled
from glide.providers import http
from glide.providers.errors import (
    CANCELLED,
    NO_MESSAGE,
    ProviderError,
    echoed,
    redact,
    retry_after,
    scrub,
    status_error,
    what_it_said,
)

KEY = "sk-test-SECRET-0123456789"


# -- errors.py ------------------------------------------------------------------------------------


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
        ({"retry-after": "inf"}, None),
        ({}, None),
    ],
)
def test_retry_after_in_seconds_milliseconds_or_as_a_date(headers, expected):
    assert retry_after(headers) == expected
    assert retry_after(httpx.Headers(headers)) == expected


def test_retry_after_as_a_date_is_the_seconds_until_then():
    when = format_datetime(datetime.now(UTC) + timedelta(seconds=90), usegmt=True)
    assert 80 <= retry_after({"retry-after": when}) <= 91


def test_redact_replaces_every_secret_and_leaves_short_ones_alone_when_asked():
    assert redact("a KEY b KEY", ["KEY"], "***") == "a *** b ***"
    assert redact("a KEY", ["", "KEY"]) == "a [redacted]"
    assert redact("an abc here", ["abc"], "***", min_len=4) == "an abc here"
    assert redact("same", []) == "same"


def test_scrub_also_cuts_anything_that_looks_like_an_encoded_blob():
    assert scrub(f"x {KEY} y {'QUJD' * 40} z", [KEY]) == "x [redacted] y [...] z"
    assert scrub("a short token QUJD stays", []) == "a short token QUJD stays"


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (json.dumps({"error": {"message": "bad key"}}), "bad key"),
        (json.dumps({"error": "bad key"}), "bad key"),
        (json.dumps({"message": "bad key"}), "bad key"),
        (json.dumps({"detail": "bad key"}), "bad key"),
        (json.dumps({"detail": {"message": "bad key", "status": "X"}}), "bad key"),
        (json.dumps({"detail": {"status": "quota_exceeded"}}), "quota_exceeded"),
        (json.dumps({"detail": [{"loc": ["body", "text"], "msg": "too long", "input": "SECRET"}]}), "body.text: too long"),
        (json.dumps({"error": {"message": ""}, "detail": "second"}), "second"),
        (json.dumps({"unknown": {"input": "SECRET"}}), NO_MESSAGE),
        (json.dumps(["SECRET"]), NO_MESSAGE),
        (json.dumps(42), NO_MESSAGE),
        (json.dumps({"error": {"message": "Bearer abc.def-ghi rejected"}}), "Bearer [redacted] rejected"),
        ("plain text, Bearer abc.def-ghi quoted", "plain text, Bearer [redacted] quoted"),
        ("<html>gateway</html>", "<html>gateway</html>"),
    ],
)
def test_an_error_reply_is_reduced_to_the_message_the_server_meant(body, expected):
    assert what_it_said(body) == expected
    assert "SECRET" not in what_it_said(body)


def test_a_quoted_request_is_cut_in_every_form_it_can_be_quoted():
    text = "Please summarise my private bank statement"
    quoted = [
        text,
        text.upper(),
        text.replace(" ", "  "),
        text.replace(" ", "\n"),
        json.dumps(text + '\nsaid "hi"')[1:-1],
    ]
    for form in quoted:
        assert "summarise" not in echoed(f"bad request: {form}", [text, text + '\nsaid "hi"']).lower()
    assert echoed("No.", ["No."]) == "No."  # too short to be worth cutting: it would garble ordinary replies
    assert echoed("x", []) == "x"
    assert echoed(f"a {text} b", [text], "***") == "a *** b"


def test_a_failed_status_is_one_safe_error_whatever_the_provider():
    error = status_error(
        400,
        json.dumps({"error": {"message": f"{KEY} rejected: {'QUJD' * 40} Please summarise my private bank statement"}}),
        provider="p",
        headers={"retry-after": "3"},
        secrets=[KEY],
        request_texts=["Please summarise my private bank statement"],
    )
    assert error.kind == "bad_request" and error.status == 400 and error.provider == "p" and error.retry_after == 3.0
    assert str(error) == "p answered 400: [redacted] rejected: [...] [redacted]"


def test_a_spent_account_is_fixed_text_judged_on_the_whole_reply():
    error = status_error(
        429, json.dumps({"error": {"code": "insufficient_quota", "message": f"{KEY} secret reply"}}), provider="p", secrets=[KEY]
    )
    assert error.kind == "auth" and error.status == 429
    assert "secret reply" not in str(error) and KEY not in str(error) and "no credit or quota left" in str(error)
    assert status_error(402, "anything", provider="p").kind == "auth"


def test_a_status_with_no_body_still_says_the_status():
    assert str(status_error(503, "", provider="p")).startswith("p answered 503")
    assert status_error(503, "", provider="p").kind == "server"
    assert status_error(404, "", provider="p").kind == "unsupported"
    assert status_error(401, "", provider="p").kind == "auth"
    assert status_error(429, "", provider="p").kind == "rate_limit"


# -- http.py --------------------------------------------------------------------------------------


def client_for(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_translate_returns_the_value_and_turns_an_httpx_error_into_one_with_no_context():
    assert http.translate("p", lambda: 7) == 7

    def boom():
        raise httpx.ReadError("secret", request=httpx.Request("GET", "https://h.test/", headers={"x-key": KEY}))

    with pytest.raises(ProviderError) as caught:
        http.translate("p", boom)
    error = caught.value
    assert error.kind == "transport" and error.provider == "p"
    assert error.__context__ is None and error.__cause__ is None  # the httpx exception holds the request and its key
    assert KEY not in "".join(traceback.format_exception(error))


def test_translate_leaves_other_errors_alone_unless_asked():
    with pytest.raises(ValueError):
        http.translate("p", lambda: int("x"))
    with pytest.raises(ProviderError) as caught:
        http.translate("p", lambda: int("x"), also=(ValueError,))
    assert caught.value.kind == "transport"
    with pytest.raises(ProviderError) as caught:
        http.translate("p", lambda: (_ for _ in ()).throw(ProviderError("ours", kind="content")))
    assert caught.value.kind == "content"


def test_a_connection_closed_by_a_cancel_is_the_cancel_not_a_transport_fault():
    control = RunControl()
    control.cancel("stopped")

    def boom():
        raise httpx.ReadError("closed")

    with controlled(control), pytest.raises(ProviderError) as caught:
        http.translate("p", boom)
    assert caught.value.kind == CANCELLED


def test_translated_passes_items_through_translates_a_failure_and_closes_what_it_wraps():
    closed = []

    def items():
        try:
            yield 1
            yield 2
            raise httpx.ReadError("dropped")
        finally:
            closed.append(True)

    got = []
    with pytest.raises(ProviderError) as caught:
        for item in http.translated("p", items()):
            got.append(item)
    assert got == [1, 2] and caught.value.kind == "transport" and caught.value.__context__ is None and closed == [True]

    closed.clear()
    inner = items()  # referenced here, so only an explicit close can end it
    stream = http.translated("p", inner)
    assert next(stream) == 1
    stream.close()  # the consumer stops early: the wrapped generator is closed by that, not left to a collector
    assert closed == [True]


def test_translated_is_cancel_aware():
    control = RunControl()

    def items():
        yield 1
        raise httpx.ReadError("closed by the cancel")

    with controlled(control):
        stream = http.translated("p", items())
        assert next(stream) == 1
        control.cancel("stopped")
        with pytest.raises(ProviderError) as caught:
            next(stream)
    assert caught.value.kind == CANCELLED


def test_read_body_reads_in_pieces_to_a_limit_and_a_deadline():
    class Body(httpx.SyncByteStream):
        def __init__(self, pieces):
            self.pieces = pieces

        def __iter__(self):
            yield from self.pieces

    def response(pieces):
        return httpx.Response(200, stream=Body(pieces))

    assert http.read_body(response([b"ab", b"cd"]), provider="p") == b"abcd"
    assert http.read_body(response([b"ab", b"cd", b"ef"]), provider="p", limit=3) == b"abcd"  # the piece that crosses it is kept

    now = [0.0]

    def clock():
        now[0] += 1
        return now[0]

    with pytest.raises(ProviderError) as caught:
        http.read_body(response([b"a", b"b", b"c"]), provider="p", deadline=2.5, clock=clock)
    assert caught.value.kind == "timeout"


def test_open_response_sends_one_request_with_nothing_read_and_streaming_closes_it():
    seen = []

    class Hello(httpx.SyncByteStream):
        def __iter__(self):
            yield b"hello"

    def handler(request):
        seen.append(request)
        return httpx.Response(200, stream=Hello())

    with client_for(handler) as client:
        response = http.open_response(client, "p", "POST", url="https://h.test/x", json={"a": 1}, headers={"h": "v"}, timeout=3)
        assert not response.is_closed and response.status_code == 200
        response.read()
        response.close()
        assert json.loads(seen[0].content) == {"a": 1} and seen[0].headers["h"] == "v" and seen[0].method == "POST"

        with http.streaming(client, "p", "POST", url="https://h.test/x", timeout=3) as streamed:
            assert not streamed.is_closed
        assert streamed.is_closed


def test_streaming_closes_the_response_when_the_block_raises():
    class One(httpx.SyncByteStream):
        def __iter__(self):
            yield b"x"

    with client_for(lambda r: httpx.Response(200, stream=One())) as client:
        with pytest.raises(RuntimeError), http.streaming(client, "p", "GET", url="https://h.test/", timeout=3) as response:
            raise RuntimeError("stop")
        assert response.is_closed


def test_a_cancel_closes_a_response_that_is_open_while_the_block_runs():
    """The reader blocked on the body is released by the close the cancel runs, not by its own deadline."""
    control = RunControl()
    released = threading.Event()

    class Held(httpx.SyncByteStream):
        def __iter__(self):
            yield b"first"
            released.wait(3)

        def close(self):
            released.set()

    with (
        client_for(lambda r: httpx.Response(200, stream=Held())) as client,
        controlled(control),
        http.streaming(client, "p", "GET", url="https://h.test/", timeout=3) as response,
    ):
        threading.Timer(0.05, lambda: control.cancel("stopped")).start()
        assert b"".join(response.iter_bytes()) == b"first"
    assert released.is_set() and control.cancelled.is_set()


class Late(httpx.SyncByteStream):
    """The body of a response that arrives after the cancel: `closed` is set when whoever got it gives it back."""

    def __init__(self) -> None:
        self.closed = threading.Event()

    def __iter__(self):
        yield b"late"

    def close(self) -> None:
        self.closed.set()


def test_a_response_that_arrives_after_the_cancel_is_closed_on_arrival_and_the_call_is_the_cancel():
    """While a request is still being connected there is no socket for `Call.abort` to shut (interrupt.py), so the
    thread making it stays in httpx until the connection comes. What is guaranteed is that the connection which then
    arrives is closed at once and the call ends as the cancel, never as an answer."""
    control, gate, body = RunControl(), threading.Event(), Late()
    started = threading.Event()

    def handler(request):
        started.set()
        gate.wait(3)
        return httpx.Response(200, stream=body)

    outcome: dict = {}

    def call():
        with controlled(control), client_for(handler) as client:
            try:
                outcome["response"] = http.open_response(client, "p", "GET", url="https://h.test/", timeout=3)
            except ProviderError as error:
                outcome["error"] = error

    thread = threading.Thread(target=call, daemon=True)
    thread.start()
    assert started.wait(3)
    control.cancel("stopped")
    gate.set()
    thread.join(3)
    assert outcome["error"].kind == CANCELLED and "response" not in outcome
    assert body.closed.is_set()

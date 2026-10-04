"""Speech to text: the ElevenLabs realtime and batch adapters, the OpenAI-style adapter, the replay buffer
and the facade. No network: sockets are fakes patched into `websocket.create_connection`, HTTP is
`httpx.MockTransport`, and the chain's own behaviour is covered in test_chain.py."""

from __future__ import annotations

import base64
import contextlib
import gc
import hashlib
import io
import json
import queue
import re
import socket
import threading
import time
import wave
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
import websocket

from glide.providers import stt
from glide.providers.base import Audio, ProviderSpec, Transcript
from glide.providers.chain import Chain, ChainPolicy, Slot
from glide.providers.errors import AllProvidersFailed, ProviderError
from glide.providers.stt import STT, ElevenLabsSTT, OpenAICompatSTT, ReplayBuffer, build_client

KEY = "xi-SECRET-key-0123456789"


# -- fakes ------------------------------------------------------------------------------------------


class FakeSocket:
    """A realtime socket that answers from a script instead of a server.

    `on_audio` maps "after the n-th audio chunk" to events to deliver, `on_commit` lists events to deliver
    when the commit arrives, and `initial` events are there from the start. An event is a dict (sent as
    JSON), "" (a close frame), or an exception (raised by recv). Like the real one it blocks in recv.
    """

    def __init__(self, *, initial=(), on_audio=None, on_commit=()):
        self.sent: list[dict] = []
        self.closed = False
        self.timeout = "never set"
        self.on_audio = on_audio or {}
        self.on_commit = list(on_commit)
        self._inbox: queue.Queue = queue.Queue()
        for item in initial:
            self.push(item)

    def push(self, item):
        self._inbox.put(item)

    def settimeout(self, timeout):
        self.timeout = timeout

    def send(self, data):
        if self.closed:
            raise websocket.WebSocketConnectionClosedException("closed")
        message = json.loads(data)
        self.sent.append(message)
        if message["commit"]:
            for item in self.on_commit:
                self.push(item)
        else:
            for item in self.on_audio.get(len(self.audio), ()):
                self.push(item)

    def recv(self):
        item = self._inbox.get(timeout=5)  # a script that says too little fails here instead of hanging
        if item is None:
            raise websocket.WebSocketConnectionClosedException("closed")
        if isinstance(item, BaseException):
            raise item
        return item if isinstance(item, str) else json.dumps(item)

    def close(self, *args, **kwargs):
        self.closed = True
        self._inbox.put(None)

    @property
    def audio(self) -> list[bytes]:
        return [base64.b64decode(m["audio_base_64"]) for m in self.sent if not m["commit"]]


class Sockets:
    """Stands in for `websocket.create_connection`: hands out the queued sockets, or raises the queued error."""

    def __init__(self):
        self.queue: list = []
        self.calls: list[dict] = []

    def add(self, *items):
        self.queue.extend(items)

    def __call__(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        item = self.queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


@pytest.fixture
def sockets(monkeypatch):
    hub = Sockets()
    monkeypatch.setattr(websocket, "create_connection", hub)
    monkeypatch.setattr(stt, "LANGUAGE_GRACE_S", 0.05)  # keep the wait for a language stamp short in tests
    return hub


def partial(text):
    return {"message_type": "partial_transcript", "text": text}


def committed(text):
    return {"message_type": "committed_transcript", "text": text}


def stamped(text, language):
    return {"message_type": "committed_transcript_with_timestamps", "text": text, "language_code": language, "words": []}


def server_error(kind, message="nope"):
    return {"message_type": kind, "error": message}


CHUNKS = [bytes([i + 1]) * 640 for i in range(6)]  # six distinct 20 ms chunks at 16 kHz


def eleven(**options) -> ElevenLabsSTT:
    return ElevenLabsSTT("elevenlabs", "scribe_v2_realtime", KEY, options=options)


def run(client, chunks=CHUNKS, **kwargs) -> list[Transcript]:
    return list(client.stream(iter(chunks), **kwargs))


def failure(callable_, *args, **kwargs) -> ProviderError:
    with pytest.raises(ProviderError) as caught:
        callable_(*args, **kwargs)
    return caught.value


def first(client, chunks=CHUNKS, **kwargs) -> ProviderError:
    return failure(lambda: next(client.stream(iter(chunks), **kwargs)))


def assert_clean(error: ProviderError, *secrets: str):
    """No key, header or body in anything a log line could print from this error."""
    for text in (str(error), repr(error), *(str(a) for a in error.args)):
        for secret in (KEY, *secrets):
            assert secret not in text
    for linked in (error.__cause__, error.__context__):
        assert linked is None or KEY not in repr(linked)


def paced(chunks, gap=0.02):
    """A microphone at its real pace: a chunk every 20 ms, so the socket's reader gets a turn between them."""
    for chunk in chunks:
        yield chunk
        time.sleep(gap)


def sleepy(limit=3000):
    """A microphone: chunks forever, each a millisecond apart, and a failure if nothing ever stops it."""
    for i in range(limit):
        time.sleep(0.001)
        yield bytes([i % 250 + 1]) * 640
    raise AssertionError("the stream never stopped reading the microphone")


# -- realtime: the request ----------------------------------------------------------------------------


def test_the_session_is_opened_with_the_key_in_a_header_and_never_in_the_url(sockets):
    ws = FakeSocket(on_commit=[committed("hello")])
    sockets.add(ws)
    run(eleven())
    call = sockets.calls[0]
    url = urlparse(call["url"])
    assert (url.scheme, url.netloc, url.path) == ("wss", "api.elevenlabs.io", "/v1/speech-to-text/realtime")
    assert parse_qs(url.query) == {
        "model_id": ["scribe_v2_realtime"],
        "audio_format": ["pcm_16000"],
        "commit_strategy": ["manual"],
        "include_language_detection": ["true"],
    }
    assert call["header"] == [f"xi-api-key: {KEY}"]
    assert KEY not in call["url"]
    assert ws.timeout is None  # the connect timeout must not stay on the socket, or a pause in speech would raise


def test_audio_goes_out_as_chunk_messages_and_the_end_of_the_audio_as_a_commit(sockets):
    ws = FakeSocket(on_commit=[committed("hello")])
    sockets.add(ws)
    run(eleven())
    assert ws.audio == CHUNKS
    for message in ws.sent[:-1]:
        assert message == {
            "message_type": "input_audio_chunk",
            "audio_base_64": message["audio_base_64"],
            "commit": False,
            "sample_rate": 16000,
        }
    assert ws.sent[-1] == {"message_type": "input_audio_chunk", "audio_base_64": "", "commit": True, "sample_rate": 16000}
    assert ws.closed


def test_a_chosen_language_is_sent_as_is_and_none_means_auto_detect(sockets):
    sockets.add(
        FakeSocket(on_commit=[committed("a")]), FakeSocket(on_commit=[committed("b")]), FakeSocket(on_commit=[committed("c")])
    )
    sent = []
    for language in ("yue", "zh-HK", "auto"):
        run(eleven(), language=language)
        sent.append(parse_qs(urlparse(sockets.calls[-1]["url"]).query))
    assert sent[0]["language_code"] == ["yue"] and "include_language_detection" not in sent[0]
    assert sent[1]["language_code"] == ["zh"]  # a locale is cut to its language
    assert "language_code" not in sent[2] and sent[2]["include_language_detection"] == ["true"]


def test_a_forced_language_comes_back_on_the_transcript(sockets):
    sockets.add(FakeSocket(on_commit=[committed("你好")]))
    assert run(eleven(), language="yue")[-1].language == "yue"


def test_url_builder_handles_regions_plain_http_and_extra_parameters():
    regional = stt.realtime_url(
        "https://api.us.elevenlabs.io/v1/", model="m", sample_rate=8000, language=None, commit_strategy="manual"
    )
    assert regional.startswith("wss://api.us.elevenlabs.io/v1/speech-to-text/realtime?")
    local = stt.realtime_url(
        "http://127.0.0.1:9", model="m", sample_rate=16000, language="en", commit_strategy="vad",
        extra={"vad_threshold": 0.4, "keyterms": ["Glide", "Mac"], "model_id": "ignored"},
    )  # fmt: skip
    query = parse_qs(urlparse(local).query)
    assert local.startswith("ws://127.0.0.1:9/")
    assert query["vad_threshold"] == ["0.4"] and query["keyterms"] == ["Glide", "Mac"]
    assert query["model_id"] == ["m"]  # what the module sets wins over a passthrough


def test_an_unsupported_sample_rate_is_refused_before_connecting_and_lets_the_chain_move_on(sockets):
    error = first(eleven(), sample_rate=11025)
    assert error.kind == "unsupported"
    assert sockets.calls == []


def test_a_missing_key_fails_as_auth_without_touching_the_network(sockets):
    client = ElevenLabsSTT("elevenlabs", "scribe_v2_realtime", "")
    assert first(client).kind == "auth"
    assert failure(client.transcribe, Audio(b"\x00\x00" * 100)).kind == "auth"
    assert sockets.calls == []


def test_odd_bytes_are_carried_over_and_oversized_chunks_are_split(sockets):
    ws = FakeSocket(on_commit=[committed("x")])
    sockets.add(ws)
    big = b"\x07\x00" * 35000  # 70 000 bytes, more than the one second the module sends at a time
    run(eleven(), [b"\x01\x02\x03", b"\x04\x05\x06", big])
    assert [len(c) for c in ws.audio] == [2, 4, 32000, 32000, 6000]
    assert b"".join(ws.audio) == b"\x01\x02\x03\x04\x05\x06" + big


# -- realtime: what comes back -----------------------------------------------------------------------


def test_partials_come_first_then_one_final_and_repeats_and_blanks_are_skipped(sockets):
    ws = FakeSocket(
        initial=[
            {"message_type": "session_started", "session_id": "s", "config": {}},
            {"message_type": "warning", "warning": "w"},
        ],
        on_audio={1: [partial("hel")], 2: [partial("hello"), partial("hello")], 3: [partial("")]},
        on_commit=[committed("hello world")],
    )
    sockets.add(ws)
    out = run(eleven())
    assert [(t.text, t.partial) for t in out] == [("hel", True), ("hello", True), ("hello world", False)]
    final = out[-1]
    assert (final.provider, final.model) == ("elevenlabs:scribe_v2_realtime", "scribe_v2_realtime")
    assert final.latency_s >= 0


def test_a_silent_utterance_still_ends_with_an_empty_final(sockets):
    sockets.add(FakeSocket(on_commit=[committed("")]))
    out = run(eleven())
    assert [(t.text, t.partial) for t in out] == [("", False)]


def test_no_audio_at_all_sends_no_commit_and_still_ends_with_a_final(sockets):
    ws = FakeSocket()
    sockets.add(ws)
    out = run(eleven(), [])
    assert [(t.text, t.partial) for t in out] == [("", False)]
    assert ws.sent == [] and ws.closed


def test_segments_the_server_committed_on_its_own_are_joined_into_the_final(sockets):
    ws = FakeSocket(
        on_audio={2: [committed("First sentence.")], 3: [partial("Second")]}, on_commit=[committed("Second sentence.")]
    )
    sockets.add(ws)
    out = run(eleven(), paced(CHUNKS))
    assert out[-1].text == "First sentence. Second sentence." and not out[-1].partial
    assert ("First sentence. Second", True) in [(t.text, t.partial) for t in out]  # a partial shows the whole utterance so far


def test_chinese_segments_are_joined_without_a_space():
    assert stt._join("你好", "世界") == "你好世界"
    assert stt._join("hello", "world") == "hello world"
    assert stt._join("", "x") == "x" and stt._join("x", "") == "x"


def test_the_language_comes_from_the_stamped_commit_and_the_text_is_not_counted_twice(sockets):
    sockets.add(
        FakeSocket(on_commit=[stamped("你好 hello", "yue")]),  # a stamped commit on its own
        FakeSocket(on_commit=[committed("hi there"), stamped("hi there", "eng")]),  # a plain commit and its stamped copy
        FakeSocket(on_commit=[committed("hi there")]),  # plain only: no language is known
    )
    only_stamped, both, plain = (run(eleven())[-1] for _ in range(3))
    assert (only_stamped.text, only_stamped.language) == ("你好 hello", "yue")
    assert (both.text, both.language) == ("hi there", "en")
    assert (plain.text, plain.language) == ("hi there", None)


def test_the_language_of_a_mixed_utterance_is_that_of_its_longest_segment(sockets):
    ws = FakeSocket(on_audio={2: [stamped("ok", "eng")]}, on_commit=[stamped("我想听一首很长的歌", "yue")])
    sockets.add(ws)
    final = run(eleven(), paced(CHUNKS))[-1]
    assert final.language == "yue" and final.text == "ok我想听一首很长的歌"


def test_vad_mode_ends_the_stream_at_the_first_committed_text_and_sends_no_commit(sockets):
    ws = FakeSocket(on_audio={2: [partial("hello"), committed("hello there")]})
    sockets.add(ws)
    out = list(eleven(commit_strategy="vad").stream(sleepy()))
    assert out[-1].text == "hello there" and not out[-1].partial
    assert parse_qs(urlparse(sockets.calls[0]["url"]).query)["commit_strategy"] == ["vad"]
    assert all(not m["commit"] for m in ws.sent) and ws.closed


def test_an_unknown_commit_strategy_is_a_configuration_error():
    with pytest.raises(ValueError):
        eleven(commit_strategy="sometimes")


# -- realtime: errors --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "kind"),
    [(401, "auth"), (403, "auth"), (429, "rate_limit"), (500, "server"), (404, "unsupported"), (400, "bad_request")],
)
def test_a_handshake_refusal_maps_by_status_and_leaks_neither_headers_nor_the_key(sockets, status, kind):
    sockets.add(
        websocket.WebSocketBadStatusException(
            f"Handshake status {status} -+-+- {{'set-cookie': 'COOKIE-VALUE'}} -+-+- body",
            status,
            resp_headers={"set-cookie": "COOKIE-VALUE"},
            resp_body=f'{{"detail": "refused {KEY}"}}'.encode(),
        )
    )
    error = first(eleven())
    assert (error.kind, error.status) == (kind, status)
    assert "refused" in str(error)  # the server's reason is kept
    assert_clean(error, "COOKIE-VALUE")


@pytest.mark.parametrize(
    ("raised", "kind"),
    [
        (ConnectionRefusedError("refused"), "transport"),
        (websocket.WebSocketAddressException("no host"), "transport"),
        (websocket.WebSocketTimeoutException("slow"), "timeout"),
        (TimeoutError("slow"), "timeout"),
    ],
)
def test_a_connection_that_cannot_be_made_is_transport_or_timeout(sockets, raised, kind):
    sockets.add(raised)
    error = first(eleven())
    assert error.kind == kind
    assert_clean(error)


def test_an_auth_error_event_is_auth_even_when_the_server_echoes_the_key(sockets):
    sockets.add(FakeSocket(initial=[server_error("auth_error", f"invalid api key {KEY}")]))
    error = first(eleven())
    assert error.kind == "auth" and "invalid api key" in str(error)
    assert_clean(error)


@pytest.mark.parametrize(
    ("event", "kind"),
    [
        ("auth_error", "auth"),
        ("unaccepted_terms", "auth"),
        ("quota_exceeded", "rate_limit"),
        ("rate_limited", "rate_limit"),
        ("commit_throttled", "rate_limit"),
        ("error", "server"),
        ("transcriber_error", "server"),
        ("queue_overflow", "server"),
        ("resource_exhausted", "server"),
        ("session_time_limit_exceeded", "timeout"),
        ("invalid_request", "unsupported"),
        ("input_error", "content"),
        ("chunk_size_exceeded", "content"),
        ("insufficient_audio_activity", "content"),
        ("something_new_error", "content"),
    ],
)
def test_every_server_error_event_becomes_a_provider_error_of_the_right_kind(sockets, event, kind):
    sockets.add(FakeSocket(initial=[server_error(event, "details")]))
    error = first(eleven())
    assert error.kind == kind
    assert event in str(error) and "details" in str(error)
    assert_clean(error)


def test_a_server_that_hangs_up_before_the_transcript_is_complete_is_transport(sockets):
    ws = FakeSocket(initial=[partial("hel"), ""])  # a partial, then a close frame
    sockets.add(ws)
    seen = []
    with pytest.raises(ProviderError) as caught:
        for transcript in eleven().stream(iter(CHUNKS)):
            seen.append(transcript.text)
    assert seen == ["hel"] and caught.value.kind == "transport"
    assert ws.closed
    assert_clean(caught.value)


@pytest.mark.parametrize("gone", [websocket.WebSocketConnectionClosedException("gone"), OSError("reset by peer"), ""])
def test_a_socket_that_dies_is_transport_with_no_raw_exception(sockets, gone):
    sockets.add(FakeSocket(initial=[gone]))
    error = first(eleven())
    assert error.kind == "transport"
    assert_clean(error)


def test_a_send_that_fails_reports_the_error_event_the_server_sent_first(sockets):
    class Hangs(FakeSocket):
        def send(self, data):
            raise websocket.WebSocketConnectionClosedException("Connection to remote host was lost.")

    sockets.add(Hangs(initial=[server_error("auth_error", "bad key")]))
    assert first(eleven()).kind == "auth"
    sockets.add(Hangs(initial=[]))
    assert first(eleven()).kind == "transport"


def test_a_commit_that_is_never_answered_times_out(sockets):
    ws = FakeSocket()
    sockets.add(ws)
    started = time.monotonic()
    error = first(eleven(final_timeout_s=0.2))
    assert error.kind == "timeout" and time.monotonic() - started < 3
    assert ws.closed
    assert_clean(error)


def test_a_failing_microphone_is_not_a_provider_error_and_still_closes_the_socket(sockets):
    ws = FakeSocket()
    sockets.add(ws)

    def mic():
        yield CHUNKS[0]
        raise RuntimeError("the microphone died")

    with pytest.raises(RuntimeError, match="microphone died"):
        list(eleven().stream(mic()))
    assert ws.closed


def test_abandoning_a_stream_closes_the_socket(sockets):
    ws = FakeSocket(on_audio={1: [partial("a")]})
    sockets.add(ws)
    stream = eleven().stream(sleepy())
    assert next(stream).text == "a"
    stream.close()
    assert ws.closed


# -- WAV and the batch endpoint ----------------------------------------------------------------------


def test_pcm_is_wrapped_in_a_valid_wav():
    pcm = b"".join(i.to_bytes(2, "little", signed=True) for i in (0, 1, -1, 32767, -32768))
    with wave.open(io.BytesIO(stt.pcm_to_wav(pcm, 22050))) as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate(), wav.getnframes()) == (1, 2, 22050, 5)
        assert wav.readframes(5) == pcm
    with wave.open(io.BytesIO(stt.pcm_to_wav(pcm + b"\x01", 16000))) as wav:  # a stray odd byte is dropped, not an error
        assert wav.getnframes() == 5
    with wave.open(io.BytesIO(stt.pcm_to_wav(b"", 16000))) as wav:
        assert wav.getnframes() == 0


def form(request: httpx.Request) -> dict[str, bytes]:
    """The parts of a multipart request, by field name."""
    boundary = re.search(r"boundary=([^;]+)", request.headers["content-type"]).group(1).encode()
    fields = {}
    for part in request.content.split(b"--" + boundary):
        head, sep, body = part.partition(b"\r\n\r\n")
        match = re.search(rb'name="([^"]+)"', head)
        if sep and match:
            fields[match.group(1).decode()] = body.removesuffix(b"\r\n")
    return fields


def http(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


AUDIO = Audio(b"\x01\x00" * 1600)


def test_the_batch_request_is_a_multipart_wav_with_the_key_in_a_header():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={"text": "  hello  ", "language_code": "eng", "words": []})

    client = ElevenLabsSTT("elevenlabs", "scribe_v2_realtime", KEY, client=http(handler))
    result = client.transcribe(AUDIO, language=None, prompt="ignored")
    request = seen[0]
    assert (request.method, str(request.url)) == ("POST", "https://api.elevenlabs.io/v1/speech-to-text")
    assert request.headers["xi-api-key"] == KEY and KEY not in str(request.url)
    fields = form(request)
    assert fields["model_id"] == b"scribe_v2"  # the batch twin of the configured realtime model
    assert fields["tag_audio_events"] == b"false" and fields["timestamps_granularity"] == b"none"
    assert "language_code" not in fields and "prompt" not in fields
    with wave.open(io.BytesIO(fields["file"])) as wav:
        assert (wav.getframerate(), wav.getnframes()) == (16000, 1600)
    assert (result.text, result.language, result.model, result.partial) == ("hello", "en", "scribe_v2", False)
    assert result.provider == "elevenlabs:scribe_v2_realtime"


def test_the_batch_language_is_sent_when_chosen_and_the_reply_language_is_folded_to_the_short_code():
    seen = []

    def handler(request):
        seen.append(form(request))
        return httpx.Response(200, json={"text": "你好", "language_code": "yue"})

    client = ElevenLabsSTT("elevenlabs", "scribe_v2", KEY, client=http(handler))
    result = client.transcribe(AUDIO, language="yue")
    assert seen[0]["language_code"] == b"yue" and result.language == "yue"


def test_empty_audio_is_an_empty_transcript_without_a_request():
    def handler(request):
        raise AssertionError("no request should be made")

    client = ElevenLabsSTT("elevenlabs", "scribe_v2", KEY, client=http(handler))
    assert client.transcribe(Audio(b"")).text == ""


@pytest.mark.parametrize(
    ("status", "kind"),
    [
        (401, "auth"),
        (403, "auth"),
        (429, "rate_limit"),
        (500, "server"),
        (503, "server"),
        (422, "bad_request"),
        (404, "unsupported"),
    ],
)
def test_batch_http_errors_map_by_status_and_never_carry_the_key(status, kind):
    def handler(request):
        return httpx.Response(status, headers={"retry-after": "7"}, json={"detail": f"rejected key {KEY}"})

    client = ElevenLabsSTT("elevenlabs", "scribe_v2", KEY, client=http(handler))
    error = failure(client.transcribe, AUDIO)
    assert (error.kind, error.status) == (kind, status)
    assert "rejected key" in str(error)
    assert_clean(error)
    if status == 429:
        assert error.retry_after == 7.0


@pytest.mark.parametrize(
    ("raised", "kind"),
    [
        (httpx.ReadTimeout("slow"), "timeout"),
        (httpx.ConnectError("refused"), "transport"),
        (httpx.RemoteProtocolError("garbled"), "transport"),
    ],
)
def test_batch_transport_failures_are_mapped(raised, kind):
    def handler(request):
        raise raised

    client = ElevenLabsSTT("elevenlabs", "scribe_v2", KEY, client=http(handler))
    error = failure(client.transcribe, AUDIO)
    assert error.kind == kind
    assert_clean(error)
    assert error.__context__ is None  # the httpx exception holds the request, and the request holds the key


@pytest.mark.parametrize(
    "reply", [httpx.Response(200, text="not json"), httpx.Response(200, json=["x"]), httpx.Response(200, json={})]
)
def test_a_batch_reply_that_cannot_be_used_is_content(reply):
    client = ElevenLabsSTT("elevenlabs", "scribe_v2", KEY, client=http(lambda request: reply))
    assert failure(client.transcribe, AUDIO).kind == "content"


def test_passthrough_options_reach_the_batch_form():
    seen = []

    def handler(request):
        seen.append(form(request))
        return httpx.Response(200, json={"text": "x"})

    client = ElevenLabsSTT(
        "elevenlabs",
        "scribe_v2",
        KEY,
        options={"batch_fields": {"no_verbatim": True, "model_id": "ignored"}},
        client=http(handler),
    )
    client.transcribe(AUDIO)
    assert seen[0]["no_verbatim"] == b"true" and seen[0]["model_id"] == b"scribe_v2"


# -- OpenAI-compatible -------------------------------------------------------------------------------


def openai(handler, **kwargs) -> OpenAICompatSTT:
    return OpenAICompatSTT("openai", "whisper-1", KEY, base_url="https://stt.example.test/v1/", client=http(handler), **kwargs)


def test_openai_style_request_shape_and_reply():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={"text": " hi "})

    result = openai(handler).transcribe(AUDIO, language="en", prompt="Glide, Mac")
    request = seen[0]
    assert str(request.url) == "https://stt.example.test/v1/audio/transcriptions"
    assert request.headers["authorization"] == f"Bearer {KEY}" and KEY not in str(request.url)
    fields = form(request)
    assert fields["model"] == b"whisper-1" and fields["response_format"] == b"json"
    assert fields["language"] == b"en" and fields["prompt"] == b"Glide, Mac"
    with wave.open(io.BytesIO(fields["file"])) as wav:
        assert wav.getnframes() == 1600
    assert (result.text, result.language, result.model) == ("hi", "en", "whisper-1")


def test_openai_style_language_is_dropped_unless_it_is_a_two_letter_code():
    seen = []

    def handler(request):
        seen.append(form(request))
        return httpx.Response(200, json={"text": "x"})

    client = openai(handler)
    for language in ("yue", "zh-HK", None, "auto"):
        client.transcribe(AUDIO, language=language)
    # "yue" would be a 400 there, and a 400 is not retried on another provider; "zh-HK" is cut to "zh"
    assert [f.get("language") for f in seen] == [None, b"zh", None, None]


def test_openai_style_without_a_key_sends_no_authorization_header():
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={"text": "x"})

    OpenAICompatSTT("local", "m", "", base_url="http://127.0.0.1:8000/v1", client=http(handler)).transcribe(AUDIO)
    assert "authorization" not in seen[0].headers


def test_openai_style_needs_a_base_url():
    with pytest.raises(ValueError):
        OpenAICompatSTT("openai", "whisper-1", KEY)


def test_openai_style_stream_waits_for_the_end_then_transcribes_once():
    seen = []

    def handler(request):
        seen.append(form(request)["file"])
        return httpx.Response(200, json={"text": "all of it", "language": "english"})

    pulled = []

    def mic():
        for chunk in CHUNKS:
            pulled.append(chunk)
            yield chunk

    out = list(openai(handler).stream(mic(), sample_rate=16000, language="en"))
    assert len(out) == 1 and (out[0].text, out[0].partial) == ("all of it", False)
    assert len(seen) == 1 and len(pulled) == len(CHUNKS)  # one request, made after every chunk was in
    with wave.open(io.BytesIO(seen[0])) as wav:
        assert wav.readframes(wav.getnframes()) == b"".join(CHUNKS)


def test_openai_style_stream_of_nothing_is_an_empty_final_without_a_request():
    def handler(request):
        raise AssertionError("no request should be made")

    out = list(openai(handler).stream(iter([])))
    assert [(t.text, t.partial) for t in out] == [("", False)]


@pytest.mark.parametrize(("status", "kind"), [(401, "auth"), (429, "rate_limit"), (500, "server"), (400, "bad_request")])
def test_openai_style_errors_map_and_never_carry_the_key(status, kind):
    client = openai(lambda request: httpx.Response(status, json={"error": {"message": f"Incorrect API key provided: {KEY}"}}))
    error = failure(client.transcribe, AUDIO)
    assert error.kind == kind
    assert_clean(error)
    error = failure(lambda: next(client.stream(iter(CHUNKS))))  # and the same through stream()
    assert error.kind == kind
    assert_clean(error)


def test_a_base_url_httpx_cannot_use_is_a_provider_error_not_a_raw_exception():
    client = OpenAICompatSTT("openai", "whisper-1", KEY, base_url="not-a-url")  # a real httpx client: it fails before any network
    error = failure(client.transcribe, AUDIO)
    assert error.kind == "transport"
    assert_clean(error)


def test_openai_style_timeout_is_a_timeout():
    def handler(request):
        raise httpx.ConnectTimeout("slow")

    assert failure(openai(handler).transcribe, AUDIO).kind == "timeout"


# -- the key, everywhere ---------------------------------------------------------------------------


def test_no_adapter_repr_or_str_shows_the_key():
    clients = [
        ElevenLabsSTT("elevenlabs", "scribe_v2_realtime", KEY),
        OpenAICompatSTT("openai", "whisper-1", KEY, base_url="https://stt.example.test/v1"),
        build_client(ProviderSpec("e", "elevenlabs"), "scribe_v2", KEY),
        build_client(ProviderSpec("o", "openai_compat", base_url="https://stt.example.test/v1"), "m", KEY),
    ]
    for client in clients:
        assert KEY not in repr(client) and KEY not in str(client)
        assert type(client).__name__ in repr(client) and client.name in repr(client)
    assert KEY not in repr(STT(Chain("stt", [Slot("a", clients[0])])).chain.status())


# -- build_client ----------------------------------------------------------------------------------


def test_build_client_dispatches_on_kind_and_merges_options():
    spec = ProviderSpec("elevenlabs", "elevenlabs", options={"commit_strategy": "vad", "final_timeout_s": 3})
    client = build_client(spec, "scribe_v2", KEY, {"final_timeout_s": 4})
    assert isinstance(client, ElevenLabsSTT) and client.name == "elevenlabs"
    assert client.commit_strategy == "vad" and client.final_timeout_s == 4  # the call's options win over the spec's
    other = build_client(ProviderSpec("groq", "openai_compat", base_url="https://stt.example.test/v1/"), "whisper", "", None)
    assert isinstance(other, OpenAICompatSTT) and other.base_url == "https://stt.example.test/v1"
    with pytest.raises(ValueError, match="macos_say"):
        build_client(ProviderSpec("say", "macos_say"), "", "")


def test_model_ids_are_mapped_between_batch_and_realtime_and_can_be_overridden():
    a = ElevenLabsSTT(model="scribe_v2_realtime")
    b = ElevenLabsSTT(model="scribe_v2")
    c = ElevenLabsSTT(model="scribe_v2", options={"batch_model": "scribe_v1", "realtime_model": "scribe_next_realtime"})
    assert (a.batch_model, a.realtime_model) == ("scribe_v2", "scribe_v2_realtime")
    assert (b.batch_model, b.realtime_model) == ("scribe_v2", "scribe_v2_realtime")
    assert (c.batch_model, c.realtime_model) == ("scribe_v1", "scribe_next_realtime")
    assert ElevenLabsSTT(model="").model == stt.DEFAULT_REALTIME_MODEL


# -- the replay buffer and the facade ---------------------------------------------------------------


def test_the_replay_buffer_gives_every_attempt_everything_pulled_so_far_then_the_rest():
    source = iter(CHUNKS)
    buffer = ReplayBuffer(source)
    one = buffer.attempt()
    assert [next(one) for _ in range(3)] == CHUNKS[:3]
    two = buffer.attempt()  # the first attempt failed after three chunks
    assert [next(two) for _ in range(4)] == CHUNKS[:4]
    assert list(two) == CHUNKS[4:]
    assert list(buffer.attempt()) == CHUNKS  # a third attempt still sees it all


def test_a_failing_source_fails_every_later_attempt_the_same_way():
    def mic():
        yield CHUNKS[0]
        raise RuntimeError("mic died")

    buffer = ReplayBuffer(mic())
    with pytest.raises(RuntimeError, match="mic died"):
        list(buffer.attempt())
    with pytest.raises(RuntimeError, match="mic died"):  # not a silently shorter recording
        list(buffer.attempt())


class FakeSTT:
    """A client that records what it was given. `fail_after` raises `error` once that many chunks were taken."""

    model = "fake"

    def __init__(self, name, *, error=None, fail_after=None, text="done", partial_first=False):
        self.name = name
        self.error = error
        self.fail_after = fail_after
        self.text = text
        self.partial_first = partial_first
        self.calls = 0
        self.received: list[bytes] = []
        self.transcribe_args = None

    def transcribe(self, audio, *, language=None, prompt=None, timeout=None):
        self.calls += 1
        self.transcribe_args = (audio, language, prompt, timeout)
        if self.error is not None:
            raise self.error
        return Transcript(self.text, "en", "ignored", "fake", 0.01)

    def stream(self, chunks, *, sample_rate=16000, language=None):
        self.calls += 1
        for chunk in chunks:
            self.received.append(chunk)
            if self.fail_after is not None and len(self.received) == self.fail_after:
                if self.partial_first:
                    yield Transcript("par", None, "ignored", "fake", 0.0, partial=True)
                raise self.error
        yield Transcript(self.text, "en", "ignored", "fake", 0.01)


def facade(*clients, **kwargs) -> STT:
    return STT(Chain("stt", [Slot(c.name, c) for c in clients], **kwargs))


def transport(message="connection dropped") -> ProviderError:
    return ProviderError(message, kind="transport")


def test_stream_failover_gives_the_next_provider_the_audio_from_the_start():
    pulls = []

    def mic():
        for chunk in CHUNKS:
            pulls.append(chunk)
            yield chunk

    a = FakeSTT("a", error=transport(), fail_after=3)
    b = FakeSTT("b")
    stt_ = facade(a, b)
    out = list(stt_.stream(mic()))
    assert a.received == CHUNKS[:3]
    assert b.received == CHUNKS  # all six, starting with the three A had already taken
    assert pulls == CHUNKS  # and the microphone was read once, never twice
    assert [(t.text, t.partial, t.provider) for t in out] == [("done", False, "b")]
    event = stt_.chain.events[-1]
    assert (event.from_slot, event.to_slot, event.kind) == ("a", "b", "transport")


def test_stream_failover_after_the_whole_utterance_was_taken_still_replays_it():
    a = FakeSTT("a", error=transport(), fail_after=len(CHUNKS))  # the OpenAI-style case: reads everything, then fails
    b = FakeSTT("b")
    list(facade(a, b).stream(iter(CHUNKS)))
    assert b.received == CHUNKS


def test_stream_does_not_fail_over_once_a_transcript_has_been_yielded():
    a = FakeSTT("a", error=transport("dropped midway"), fail_after=2, partial_first=True)
    b = FakeSTT("b")
    seen = []
    with pytest.raises(ProviderError) as caught:
        for transcript in facade(a, b).stream(iter(CHUNKS)):
            seen.append(transcript.text)
    assert seen == ["par"] and caught.value.kind == "stream"
    assert b.calls == 0


def test_stream_does_not_retry_a_bad_request_elsewhere():
    a = FakeSTT("a", error=ProviderError("bad audio", kind="bad_request"), fail_after=1)
    b = FakeSTT("b")
    with pytest.raises(ProviderError) as caught:
        list(facade(a, b).stream(iter(CHUNKS)))
    assert caught.value.kind == "bad_request" and b.calls == 0


def test_stream_reports_every_provider_when_all_fail():
    a, b = FakeSTT("a", error=transport(), fail_after=1), FakeSTT("b", error=ProviderError("x", kind="server"), fail_after=1)
    with pytest.raises(AllProvidersFailed) as caught:
        list(facade(a, b).stream(iter(CHUNKS)))
    assert [(n, e.kind) for n, e in caught.value.errors] == [("a", "transport"), ("b", "server")]


def test_a_dead_microphone_propagates_as_the_bug_it_is_and_no_provider_is_blamed():
    def mic():
        yield CHUNKS[0]
        raise RuntimeError("mic died")

    a, b = FakeSTT("a"), FakeSTT("b")
    stt_ = facade(a, b)
    with pytest.raises(RuntimeError, match="mic died"):
        list(stt_.stream(mic()))
    assert b.calls == 0 and stt_.chain.status()[0]["failures"] == 0


def test_the_real_adapter_replays_the_utterance_to_the_next_provider(sockets):
    """The same scenario as above, through two ElevenLabsSTT clients and their sockets."""
    dropped = FakeSocket(on_audio={3: [""]})  # hangs up after the third chunk
    healthy = FakeSocket(on_commit=[committed("hello there")])
    sockets.add(dropped, healthy)
    a = ElevenLabsSTT("el-a", "scribe_v2_realtime", KEY)
    b = ElevenLabsSTT("el-b", "scribe_v2_realtime", KEY)
    stt_ = facade(a, b)
    out = list(stt_.stream(iter(CHUNKS)))
    assert healthy.audio == CHUNKS  # every chunk, in order, from the first
    assert len(dropped.audio) >= 3
    assert [(t.text, t.partial, t.provider) for t in out] == [("hello there", False, "el-b")]
    assert (stt_.chain.events[-1].from_slot, stt_.chain.events[-1].kind) == ("el-a", "transport")
    assert dropped.closed and healthy.closed


def test_abandoning_the_facade_stream_closes_the_provider_socket(sockets):
    """The explicit path: whoever stops reading closes the stream, and the facade closes the client's own stream with it,
    though something else still holds that one, so the socket is not left to a collector."""
    ws = FakeSocket(on_audio={1: [partial("a")]})
    sockets.add(ws)
    client, held = ElevenLabsSTT("el", "scribe_v2_realtime", KEY), []
    original = client.stream
    client.stream = lambda *args, **kwargs: held.append(original(*args, **kwargs)) or held[-1]
    stream = facade(client).stream(sleepy())
    with contextlib.closing(stream):
        assert next(stream).text == "a"
        assert not ws.closed  # still open while it is being read
    assert ws.closed


def test_a_collected_facade_stream_closes_the_provider_socket(sockets):
    """The best-effort path, for a caller that forgets to close: collection closes the generator. The collection is
    forced here, so the test does not depend on when an interpreter happens to run it."""
    ws = FakeSocket(on_audio={1: [partial("a")]})
    sockets.add(ws)
    stream = facade(ElevenLabsSTT("el", "scribe_v2_realtime", KEY)).stream(sleepy())
    assert next(stream).text == "a"
    del stream
    gc.collect()
    assert ws.closed


def test_transcribe_fails_over_and_reports_the_slot_that_answered():
    a, b = FakeSTT("a", error=ProviderError("busy", kind="rate_limit")), FakeSTT("b", text="from b")
    stt_ = facade(a, b)
    result = stt_.transcribe(AUDIO, language="yue", prompt="p", timeout=9.0)
    assert (result.text, result.provider) == ("from b", "b")
    assert b.transcribe_args == (AUDIO, "yue", "p", 9.0)
    assert stt_.chain.events[-1].kind == "rate_limit"


def test_transcribe_may_race_a_slow_provider_when_the_policy_asks():
    release = threading.Event()

    class Slow(FakeSTT):
        def transcribe(self, audio, **kwargs):
            release.wait(5)
            return Transcript("slow", None, "ignored", "fake", 5.0)

    stt_ = facade(Slow("a"), FakeSTT("b", text="fast"), policy=ChainPolicy(hedge_after_s=0.05))
    try:
        assert stt_.transcribe(AUDIO).text == "fast"
    finally:
        release.set()
    assert stt_.chain.events[0].kind == "slow"


def test_the_facade_is_usable_with_real_adapters_and_chain_pins(sockets):
    a = OpenAICompatSTT(
        "openai", "whisper-1", KEY, base_url="https://stt.example.test/v1", client=http(lambda r: httpx.Response(500))
    )
    b = OpenAICompatSTT(
        "groq",
        "whisper-large",
        KEY,
        base_url="https://stt2.example.test/v1",
        client=http(lambda r: httpx.Response(200, json={"text": "ok"})),
    )
    stt_ = facade(a, b)
    assert stt_.transcribe(AUDIO).provider == "groq"  # `a` answered 500, so `b` took over
    stt_.chain.pin("groq")
    assert next(iter(stt_.stream(iter(CHUNKS)))).provider == "groq"


# -- the real websocket-client against a real server on this machine ---------------------------------
#
# The fakes above encode what we believe websocket-client does (recv returns "" on a close frame, a
# handshake refusal carries a status and a body, settimeout(None) lifts the connect timeout, close(timeout=0)
# does not stall). These tests check those beliefs against the library itself, over 127.0.0.1 only.

GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
CLOSE_1000 = b"\x88\x02\x03\xe8"


class LoopbackServer:
    """The smallest websocket server that speaks to websocket-client: one connection, then a script.

    It records the request head and the audio it received, answers audio chunk n with `on_audio[n]` and a
    commit with `on_commit`, and may refuse the handshake or hang up after a chunk.
    """

    def __init__(self, *, refuse=None, on_audio=None, on_commit=(), hang_up_after=None, silent_close=False):
        self.head = ""
        self.audio: list[bytes] = []
        self.commits = 0
        self.refuse = refuse  # (status, detail)
        self.on_audio = on_audio or {}
        self.on_commit = list(on_commit)
        self.hang_up_after = hang_up_after
        self.silent_close = silent_close  # never answer the client's close frame, as a busy server might
        self._listener = socket.socket()
        self._listener.bind(("127.0.0.1", 0))
        self._listener.listen(1)
        self.port = self._listener.getsockname()[1]
        threading.Thread(target=self._serve, daemon=True).start()

    def stop(self):
        with contextlib.suppress(OSError):
            self._listener.close()

    def _serve(self):
        try:
            conn, _ = self._listener.accept()
            with conn:
                self._talk(conn)
        except OSError:
            pass

    @staticmethod
    def _exact(conn, n):
        data = b""
        while len(data) < n:
            part = conn.recv(n - len(data))
            if not part:
                return None
            data += part
        return data

    def _read_frame(self, conn):
        head = self._exact(conn, 2)
        if head is None:
            return None
        opcode, size = head[0] & 0x0F, head[1] & 0x7F
        if size == 126:
            size = int.from_bytes(self._exact(conn, 2), "big")
        elif size == 127:
            size = int.from_bytes(self._exact(conn, 8), "big")
        mask = self._exact(conn, 4)
        payload = self._exact(conn, size) if size else b""
        if mask is None or payload is None:
            return None
        return opcode, bytes(b ^ mask[i % 4] for i, b in enumerate(payload))

    @staticmethod
    def _send(conn, event):
        data = json.dumps(event).encode()
        size = bytes([len(data)]) if len(data) < 126 else b"\x7e" + len(data).to_bytes(2, "big")
        conn.sendall(b"\x81" + size + data)

    def _talk(self, conn):
        head = b""
        while b"\r\n\r\n" not in head:
            part = conn.recv(4096)
            if not part:
                return
            head += part
        self.head = head.decode("latin-1")
        if self.refuse:
            status, detail = self.refuse
            body = json.dumps({"detail": detail}).encode()
            conn.sendall(
                f"HTTP/1.1 {status} Refused\r\nContent-Type: application/json\r\nSet-Cookie: SERVER-COOKIE\r\n"
                f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode()
                + body
            )
            return
        key = re.search(r"sec-websocket-key: (.+)\r\n", self.head, re.I).group(1).strip()
        accept = base64.b64encode(hashlib.sha1((key + GUID).encode()).digest()).decode()
        conn.sendall(
            f"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: {accept}\r\n\r\n".encode()
        )
        while (frame := self._read_frame(conn)) is not None:
            opcode, payload = frame
            if opcode == 8:
                if self.silent_close:
                    time.sleep(6)
                else:
                    conn.sendall(b"\x88\x00")
                return
            if opcode != 1:
                continue
            message = json.loads(payload)
            if message["commit"]:
                self.commits += 1
                for event in self.on_commit:
                    self._send(conn, event)
                continue
            self.audio.append(base64.b64decode(message["audio_base_64"]))
            for event in self.on_audio.get(len(self.audio), ()):
                self._send(conn, event)
            if self.hang_up_after == len(self.audio):
                conn.sendall(CLOSE_1000)
                return


@pytest.fixture
def loopback(monkeypatch):
    # conftest refuses websocket.create_connection for everyone; a test that needs it patches it back,
    # and the socket guard there still lets nothing but 127.0.0.1 through.
    monkeypatch.setattr(websocket, "create_connection", websocket._core.create_connection)
    monkeypatch.setattr(stt, "LANGUAGE_GRACE_S", 0.05)
    servers: list[LoopbackServer] = []

    def start(**kwargs) -> tuple[LoopbackServer, ElevenLabsSTT]:
        options = kwargs.pop("options", {})
        server = LoopbackServer(**kwargs)
        servers.append(server)
        return server, ElevenLabsSTT(
            "elevenlabs", "scribe_v2_realtime", KEY, base_url=f"http://127.0.0.1:{server.port}", options=options
        )

    yield start
    for server in servers:
        server.stop()


def test_real_library_a_full_session_with_the_key_only_in_a_header(loopback):
    server, client = loopback(
        on_audio={n: [partial(f"word {n}")] for n in range(1, 41)}, on_commit=[stamped("word 40 and then some", "eng")]
    )
    chunks = [bytes([i % 250 + 1]) * 640 for i in range(60)]
    out = run(client, chunks)  # sending and receiving at once, on two threads, over a real socket
    assert server.audio == chunks and server.commits == 1
    assert out[-1].text == "word 40 and then some" and out[-1].language == "en" and not out[-1].partial
    assert [t.text for t in out[:-1]] == [f"word {n}" for n in range(1, 41)]  # every partial, in order, none lost
    request_line = server.head.splitlines()[0]
    assert request_line.startswith("GET /v1/speech-to-text/realtime?") and KEY not in request_line
    assert f"xi-api-key: {KEY}" in server.head


def test_real_library_a_pause_in_speech_longer_than_the_connect_timeout_is_not_an_error(loopback):
    _server, client = loopback(on_commit=[committed("still here")], options={"connect_timeout_s": 0.3})

    def mic():
        yield CHUNKS[0]
        time.sleep(0.9)  # the socket would raise a timeout here if the connect timeout were still on it
        yield CHUNKS[1]

    assert list(client.stream(mic()))[-1].text == "still here"


def test_real_library_a_handshake_refusal_carries_status_and_body_and_no_headers(loopback):
    _server, client = loopback(refuse=(401, f"Invalid API key {KEY}"))
    error = first(client)
    assert (error.kind, error.status) == ("auth", 401)
    assert "Invalid API key" in str(error)
    assert_clean(error, "SERVER-COOKIE")


def test_real_library_a_close_frame_before_the_final_is_transport(loopback):
    _server, client = loopback(on_audio={2: [partial("hel")]}, hang_up_after=3)
    seen = []
    with pytest.raises(ProviderError) as caught:
        for transcript in client.stream(paced(CHUNKS)):
            seen.append(transcript.text)
    assert seen == ["hel"] and caught.value.kind == "transport"
    assert_clean(caught.value)


def test_real_library_a_server_error_event_over_the_wire(loopback):
    _server, client = loopback(on_audio={1: [server_error("quota_exceeded", "usage limit reached")]})
    error = failure(lambda: list(client.stream(paced(CHUNKS))))
    assert error.kind == "rate_limit" and "usage limit reached" in str(error)


def test_real_library_closing_does_not_stall_the_caller(loopback):
    _server, client = loopback(on_commit=[committed("bye")], silent_close=True)
    started = time.monotonic()
    run(client)
    assert time.monotonic() - started < 2.0  # websocket-client's default close waits up to 3 s for a reply that never comes


def test_real_library_nothing_listening_is_transport(loopback):
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    listener.close()  # the port is now closed
    client = ElevenLabsSTT("elevenlabs", "scribe_v2_realtime", KEY, base_url=f"http://127.0.0.1:{port}")
    error = first(client)
    assert error.kind == "transport"
    assert_clean(error)


# -- R2 audit: free text a server may quote ------------------------------------------------------------


def test_a_server_error_that_quotes_the_prompt_field_does_not_repeat_it():
    prompt = "Earlier in this conversation the user asked about the transfer to Dmitri Volkov."

    def handler(request):
        return httpx.Response(400, json={"error": {"message": f"Invalid 'prompt': {prompt}"}})

    with pytest.raises(ProviderError) as caught:
        openai(handler).transcribe(AUDIO, prompt=prompt)
    assert "Volkov" not in str(caught.value) and "Invalid" in str(caught.value)


def test_an_error_event_does_not_repeat_a_long_token_or_the_configured_keyterms():
    blob = "A" * 90
    event = stt.Event("input_error", error=f"bad {blob} near Dmitri Volkov Savings Account in keyterms")
    error = stt.error_for_event(event, "p", KEY, {"keyterms": ["Dmitri Volkov Savings Account"]})
    assert blob not in str(error) and "Volkov" not in str(error)

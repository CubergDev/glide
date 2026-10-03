"""The optional realtime text-to-speech adapter against a fake socket: shape, errors, and what never leaks."""

from __future__ import annotations

import base64
import json
import queue

import pytest
import websocket
from guards_voice import no_real_audio  # noqa: F401

from glide.providers.errors import ProviderError
from glide.speech.elevenlabs import RealtimeElevenLabsTTS

KEY = "xi-SECRET-key-0123456789"
TEXT = "A sentence that must not appear in an error."


def audio(pcm: bytes) -> dict:
    return {"audio": base64.b64encode(pcm).decode()}


class Socket:
    def __init__(self, events):
        self.inbox = queue.Queue()
        for event in events:
            self.inbox.put(event)
        self.sent, self.closed, self.timeout = [], False, None

    def send(self, data):
        self.sent.append(json.loads(data))

    def settimeout(self, timeout):
        self.timeout = timeout

    def recv(self):
        item = self.inbox.get(timeout=2)
        if isinstance(item, BaseException):
            raise item
        return item if isinstance(item, str) else json.dumps(item)

    def close(self, *a, **k):
        self.closed = True


@pytest.fixture
def connect(monkeypatch):
    calls = []
    hub = {"next": None}

    def create(url, **kwargs):
        calls.append({"url": url, **kwargs})
        if isinstance(hub["next"], BaseException):
            raise hub["next"]
        return hub["next"]

    monkeypatch.setattr(websocket, "create_connection", create)
    return SimpleNamespaceHub(calls, hub)


class SimpleNamespaceHub:
    def __init__(self, calls, hub):
        self.calls, self._hub = calls, hub

    def serve(self, item):
        self._hub["next"] = item


def client(**kw):
    kw.setdefault("voice", "voice-x")
    return RealtimeElevenLabsTTS("rt", "model-x", KEY, base_url="https://tts.example", sample_rate=16000, **kw)


def test_audio_chunks_come_out_whole_samples_and_the_request_is_sent_once(connect):
    socket = Socket([audio(b"\x01\x00\x02"), audio(b"\x00\x03\x00"), {"is_final": True}])
    connect.serve(socket)
    chunks = list(client().stream(TEXT, language="en"))
    assert b"".join(chunks) == b"\x01\x00\x02\x00\x03\x00" and all(len(c) % 2 == 0 for c in chunks)
    assert socket.sent[1] == {"inputs": [{"text": TEXT, "voice_id": "voice-x", "new_turn": True}]}
    assert socket.closed
    call = connect.calls[0]
    assert call["url"].startswith("wss://tts.example/") and "model_id=model-x" in call["url"] and "pcm_16000" in call["url"]
    assert KEY not in call["url"] and call["header"] == [f"xi-api-key: {KEY}"]


def test_nothing_is_connected_until_the_first_chunk_and_blank_text_sends_nothing(connect):
    stream = client().stream(TEXT)
    assert connect.calls == []
    assert list(client().stream("  ")) == [] and connect.calls == []
    del stream


def test_a_voice_comes_from_the_call_the_language_map_or_the_option_and_there_is_no_default(connect):
    connect.serve(Socket([audio(b"\x00\x00"), {"is_final": True}]))
    list(client(voice=None, voices={"yue": "cantonese-voice"}).stream(TEXT, language="yue"))
    assert connect.calls
    with pytest.raises(ProviderError) as caught:
        list(client(voice=None).stream(TEXT, language="en"))
    assert caught.value.kind == "unsupported"


def test_errors_are_provider_errors_and_never_carry_the_key_the_text_or_the_servers_words(connect):
    connect.serve(Socket([{"error": f"bad {KEY} {TEXT}"}]))
    with pytest.raises(ProviderError) as caught:
        list(client().stream(TEXT))
    assert caught.value.kind == "server" and KEY not in str(caught.value) and TEXT not in str(caught.value)


@pytest.mark.parametrize(
    ("failure", "kind"),
    [
        (websocket.WebSocketTimeoutException("slow"), "timeout"),
        (websocket.WebSocketException("x"), "transport"),
        (OSError(f"refused {KEY}"), "transport"),
    ],
)
def test_a_connect_failure_maps_to_one_kind_without_the_exceptions_text(connect, failure, kind):
    connect.serve(failure)
    with pytest.raises(ProviderError) as caught:
        list(client().stream(TEXT))
    assert caught.value.kind == kind and KEY not in str(caught.value)


def test_a_refused_key_is_an_auth_error_from_the_status_alone(connect):
    connect.serve(
        websocket.WebSocketBadStatusException("Handshake status %d %s", 401, "Unauthorized", {"set-cookie": KEY}, b"body")
    )
    with pytest.raises(ProviderError) as caught:
        list(client().stream(TEXT))
    assert caught.value.kind == "auth" and caught.value.status == 401 and KEY not in str(caught.value)


def test_a_socket_that_closes_mid_audio_is_a_stream_error_and_a_final_with_no_audio_is_content(connect):
    connect.serve(Socket([audio(b"\x00\x00"), ""]))
    with pytest.raises(ProviderError) as caught:
        list(client().stream(TEXT))
    assert caught.value.kind == "stream"
    connect.serve(Socket([{"is_final": True}]))
    with pytest.raises(ProviderError) as caught:
        list(client().stream(TEXT))
    assert caught.value.kind == "content"


def test_the_key_is_checked_before_anything_is_connected(connect):
    with pytest.raises(ProviderError) as caught:
        list(RealtimeElevenLabsTTS("rt", "m", "", base_url="https://tts.example", voice="v").stream(TEXT))
    assert caught.value.kind == "auth" and connect.calls == []


def test_synthesize_joins_the_stream_and_carries_the_rate(connect):
    connect.serve(Socket([audio(b"\x01\x00"), audio(b"\x02\x00"), {"is_final": True}]))
    result = client().synthesize(TEXT)
    assert result.pcm == b"\x01\x00\x02\x00" and result.sample_rate == 16000


def test_bad_settings_fail_at_build_time():
    with pytest.raises(ValueError, match="sample_rate"):
        RealtimeElevenLabsTTS("rt", "m", KEY, base_url="https://tts.example", sample_rate=12345)
    with pytest.raises(ValueError, match="base_url"):
        RealtimeElevenLabsTTS("rt", "m", KEY, base_url="")

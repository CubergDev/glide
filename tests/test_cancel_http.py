"""A cancel through the real adapters, over `httpx.MockTransport` and the websocket fake (D2).

The transport's response body is a `SyncByteStream` that blocks on an Event and records `close()`, so what is proved
is exactly this: the adapter registered the response with the control, the cancel closed it from the cancelling
thread, the caller was released with `ProviderError(kind="cancelled")`, nothing was sent again and no provider was
tried instead. Whether closing a real socket wakes a real read is proved separately, over loopback, in
test_cancel_wire.py.
"""

from __future__ import annotations

import json
import threading

import httpx
from test_stt import KEY as STT_KEY
from test_stt import FakeSocket, sockets  # noqa: F401  (a fixture)
from test_tts import KEY as TTS_KEY

from glide.computer.control import RunControl, controlled
from glide.computer.models import Abort
from glide.providers.base import Audio
from glide.providers.chain import Chain, Slot
from glide.providers.classifier import ChainedClassifier
from glide.providers.errors import CANCELLED, ProviderError
from glide.providers.llm import LLM, OpenAICompatLLM
from glide.providers.stt import STT, ElevenLabsSTT
from glide.providers.tts import TTS, ElevenLabsTTS
from glide.providers.writer_client import ChainWriter

WAIT = 3.0
SSE = b'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n'
COMPLETION = json.dumps({"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}).encode()


class HeldBody(httpx.SyncByteStream):
    """A response body that sends `first`, then waits. `close()` is what a cancel is supposed to call."""

    def __init__(self, first: bytes = b"") -> None:
        self.first = first
        self.sent = threading.Event()
        self.closed = threading.Event()

    def __iter__(self):
        if self.first:
            yield self.first
        self.sent.set()
        self.closed.wait(WAIT)

    def close(self) -> None:
        self.closed.set()


class Server:
    """A MockTransport handler: counts requests, and either holds the body of its answer or blocks before any."""

    def __init__(self, *, first: bytes = b"", status: int = 200, headers=None, block_before_headers: bool = False) -> None:
        self.body = HeldBody(first)
        self.status, self.headers = status, headers or {}
        self.requests = 0
        self.entered = threading.Event()
        self.release = threading.Event()
        self.block_before_headers = block_before_headers
        self.then: Exception | None = None

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests += 1
        self.entered.set()
        if self.block_before_headers:
            self.release.wait(WAIT)
            if self.then is not None:
                raise self.then
        return httpx.Response(self.status, headers=self.headers, stream=self.body)

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self))


def in_thread(call):
    done, box = threading.Event(), {}

    def target():
        try:
            box["value"] = call()
        except BaseException as exc:
            box["error"] = exc
        finally:
            done.set()

    threading.Thread(target=target, daemon=True).start()
    return done, box


def llm_over(server: Server) -> LLM:
    client = OpenAICompatLLM("t:m", "m", "http://llm.example.test/v1", "k", transport=httpx.MockTransport(server))
    return LLM(Chain("llm", [Slot("t:m", client)]))


def under(control, call):
    def run():
        with controlled(control):
            return call()

    return in_thread(run)


# -- the chat model ---------------------------------------------------------------------------------


def test_a_stream_the_user_stops_has_its_response_closed_and_the_caller_released():
    server = Server(first=SSE, headers={"content-type": "text/event-stream"})
    llm, control, seen = llm_over(server), RunControl(), []

    def consume():
        for delta in llm.stream([{"role": "user", "content": "hello"}]):
            seen.append(delta)

    done, box = under(control, consume)
    assert server.body.sent.wait(WAIT) and seen == ["Hi"]
    control.cancel("stopped by the user")
    assert done.wait(WAIT), "the caller is still blocked after the cancel"
    assert server.body.closed.is_set()
    assert box["error"].kind == CANCELLED
    assert server.requests == 1  # a closed connection looks stale; it must not be asked again
    assert list(llm.chain.events) == [] and llm.chain.status()[0]["failures"] == 0


def test_a_chat_the_user_stops_mid_body_closes_the_response_and_is_not_retried_or_failed_over():
    server = Server(first=b'{"choices":', headers={"content-type": "application/json"})
    llm, control = llm_over(server), RunControl()
    done, box = under(control, lambda: llm.chat([{"role": "user", "content": "hello"}]))
    assert server.body.sent.wait(WAIT)
    control.cancel("stopped by the user")
    assert done.wait(WAIT)
    assert server.body.closed.is_set() and box["error"].kind == CANCELLED
    assert server.requests == 1 and list(llm.chain.events) == []


def test_a_chat_still_waiting_for_its_first_byte_returns_at_once_and_is_never_sent_again():
    """Nothing is open to close yet, so the request is left to end on its own and its late answer is dropped."""
    server = Server(headers={"content-type": "application/json"}, block_before_headers=True)
    server.body = HeldBody(COMPLETION)
    llm, control = llm_over(server), RunControl()
    done, box = under(control, lambda: llm.chat([{"role": "user", "content": "hello"}]))
    assert server.entered.wait(WAIT)
    control.cancel("stopped by the user")
    assert done.wait(WAIT)
    assert box["error"].kind == CANCELLED and "value" not in box
    server.release.set()  # the transport answers late: nobody is listening
    assert server.requests == 1 and list(llm.chain.events) == []


def test_a_connection_closed_by_the_cancel_is_not_treated_as_a_stale_one_to_retry():
    """`RemoteProtocolError` is what a closed socket raises, and is also what a kept-alive connection the server had
    closed raises, which the client answers with one more request. After a cancel it must not."""
    server = Server(block_before_headers=True)
    server.then = httpx.RemoteProtocolError("server disconnected")
    llm, control = llm_over(server), RunControl()
    done, box = under(control, lambda: llm.chat([{"role": "user", "content": "hello"}]))
    assert server.entered.wait(WAIT)
    control.cancel()
    assert done.wait(WAIT) and box["error"].kind == CANCELLED
    server.release.set()
    assert server.requests == 1


def test_a_cancel_before_the_chat_sends_nothing():
    server = Server()
    llm, control = llm_over(server), RunControl()
    control.cancel()
    done, box = under(control, lambda: llm.chat([{"role": "user", "content": "hello"}]))
    assert done.wait(WAIT) and box["error"].kind == CANCELLED and server.requests == 0


def test_without_a_control_the_adapter_behaves_as_before():
    server = Server(first=COMPLETION, headers={"content-type": "application/json"})
    server.body.closed.set()
    assert llm_over(server).chat([{"role": "user", "content": "hello"}]).text == "ok"


# -- text to speech ---------------------------------------------------------------------------------


def tts_over(server: Server) -> TTS:
    client = ElevenLabsTTS("eleven", "model-x", TTS_KEY, voice="v", client=server.client())
    return TTS(Chain("tts", [Slot("eleven", client)]))


def test_speech_the_user_stops_has_its_response_closed_and_no_second_voice_is_tried():
    server = Server(first=b"\x01\x00\x02\x00")
    tts, control, got = tts_over(server), RunControl(), []

    def speak():
        for chunk in tts.stream("Hello there."):
            got.append(chunk.pcm)

    done, box = under(control, speak)
    assert server.body.sent.wait(WAIT) and got == [b"\x01\x00\x02\x00"]
    control.cancel("stopped by the user")
    assert done.wait(WAIT)
    assert server.body.closed.is_set() and box["error"].kind == CANCELLED
    assert server.requests == 1 and list(tts.chain.events) == []


def test_speech_that_a_cancel_cut_short_does_not_end_as_a_finished_sentence():
    """If closing the body only ends the iteration, the sentence must not look complete."""
    server = Server(first=b"\x01\x00")
    client = ElevenLabsTTS("eleven", "m", TTS_KEY, voice="v", client=server.client())
    control = RunControl()
    done, box = under(control, lambda: list(client.stream("Hello there.")))
    assert server.body.sent.wait(WAIT)
    control.cancel()
    assert done.wait(WAIT) and box["error"].kind == CANCELLED


# -- speech to text ---------------------------------------------------------------------------------


def test_a_batch_transcription_the_user_stops_has_its_response_closed():
    server = Server(first=b'{"text":', headers={"content-type": "application/json"})
    client = ElevenLabsSTT("eleven", "scribe_v2_realtime", STT_KEY, client=server.client())
    stt = STT(Chain("stt", [Slot("eleven", client)]))
    control = RunControl()
    done, box = under(control, lambda: stt.transcribe(Audio(b"\x01\x00" * 1600)))
    assert server.body.sent.wait(WAIT)
    control.cancel("stopped by the user")
    assert done.wait(WAIT)
    assert server.body.closed.is_set() and box["error"].kind == CANCELLED
    assert server.requests == 1 and list(stt.chain.events) == []


class AbortableSocket(FakeSocket):
    """The fake realtime socket, which, like websocket-client's, can be aborted from another thread."""

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self.aborted = threading.Event()

    def abort(self) -> None:
        self.aborted.set()
        self.push(None)  # a shut-down socket ends the read blocked on it


def test_a_stop_during_streaming_transcription_closes_the_socket_and_releases_the_caller(sockets):  # noqa: F811
    socket_ = AbortableSocket()
    sockets.add(socket_)
    client = ElevenLabsSTT("eleven", "scribe_v2_realtime", STT_KEY)
    stt = STT(Chain("stt", [Slot("eleven", client)]))
    control, heard, sent_audio = RunControl(), threading.Event(), threading.Event()

    def microphone():
        yield b"\x01\x00" * 800
        sent_audio.set()
        heard.wait(WAIT)  # the person keeps talking: the transcription is waiting for more audio, and then for the commit
        yield b"\x01\x00" * 800

    def listen():
        return list(stt.stream(microphone()))

    done, box = under(control, listen)
    assert sent_audio.wait(WAIT)
    control.cancel("stopped by the user")
    assert done.wait(WAIT), "the caller is still blocked after the stop"
    assert socket_.aborted.is_set() and socket_.closed
    assert box["error"].kind == CANCELLED
    assert list(stt.chain.events) == [] and stt.chain.status()[0]["failures"] == 0
    heard.set()


def test_a_stop_while_waiting_for_the_final_transcript_closes_the_socket(sockets):  # noqa: F811
    socket_ = AbortableSocket()  # never answers the commit
    sockets.add(socket_)
    client = ElevenLabsSTT("eleven", "scribe_v2_realtime", STT_KEY)
    control, committed = RunControl(), threading.Event()
    real_send = socket_.send

    def send(data):
        real_send(data)
        if json.loads(data)["commit"]:
            committed.set()

    socket_.send = send
    done, box = under(control, lambda: list(client.stream(iter([b"\x01\x00" * 800]))))
    assert committed.wait(WAIT)
    control.cancel("stopped by the user")
    assert done.wait(WAIT) and socket_.aborted.is_set()
    assert box["error"].kind == CANCELLED


def test_a_socket_that_cannot_be_aborted_is_still_closed_and_the_caller_released(sockets):  # noqa: F811
    socket_ = FakeSocket()  # no `abort`: only `close`
    sockets.add(socket_)
    client = ElevenLabsSTT("eleven", "scribe_v2_realtime", STT_KEY)
    control, committed = RunControl(), threading.Event()
    real_send = socket_.send

    def send(data):
        real_send(data)
        if json.loads(data)["commit"]:
            committed.set()

    socket_.send = send
    done, box = under(control, lambda: list(client.stream(iter([b"\x01\x00" * 800]))))
    assert committed.wait(WAIT)
    control.cancel()
    assert done.wait(WAIT) and socket_.closed and box["error"].kind == CANCELLED


# -- the classifier and the writer: what the run loop sees -----------------------------------------------


class HangingClassifier:
    name = "slow"

    def __init__(self) -> None:
        self.entered, self.release = threading.Event(), threading.Event()

    def system_one(self, **request):
        self.entered.set()
        self.release.wait(WAIT)
        raise ProviderError("slow could not be reached", kind="transport", provider="slow")


def test_a_cancelled_classifier_call_is_an_abort_with_the_reason_not_a_provider_failure():
    slow, fallback, events = HangingClassifier(), HangingClassifier(), []
    fallback.name = "other"
    classifier = ChainedClassifier.from_clients([slow, fallback], on_event=events.append)
    control = RunControl()
    done, box = under(control, lambda: classifier.system_one(state={}, questions={"q": object()}))
    assert slow.entered.wait(WAIT)
    control.cancel("stopped by the user")
    assert done.wait(WAIT)
    assert isinstance(box["error"], Abort) and str(box["error"]) == "stopped by the user"
    assert not fallback.entered.is_set() and events == []
    slow.release.set()


class HangingChat:
    """A chat client that never answers, in a slot of a real chain, as the writer's facade has it."""

    name = "slow"

    def __init__(self) -> None:
        self.entered, self.release, self.calls = threading.Event(), threading.Event(), 0

    def chat(self, messages, **kw):
        self.calls += 1
        self.entered.set()
        self.release.wait(WAIT)
        raise ProviderError("slow could not be reached", kind="transport", provider="slow")


def test_a_writer_call_the_user_stops_is_an_abort_and_returns_without_waiting_for_the_deadline():
    from glide.computer.generation import GenerationRequest

    client = HangingChat()
    facade = LLM(Chain("llm", [Slot("slow", client), Slot("other", HangingChat())]))
    writer, control = ChainWriter(facade, facade), RunControl()
    request = GenerationRequest(model="m", role="writer", instructions="i", text="t", schema={}, deadline_s=600.0)
    done, box = in_thread(lambda: writer.generate(request, control))  # the control is passed, not made current
    assert client.entered.wait(WAIT)
    control.cancel("stopped by the user")
    assert done.wait(WAIT)
    assert isinstance(box["error"], Abort) and str(box["error"]) == "stopped by the user"
    assert client.calls == 1 and list(facade.chain.events) == []
    client.release.set()

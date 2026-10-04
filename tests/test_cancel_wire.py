"""A cancel over a real socket: the connection is shut down, and the server on the other end sees it go (D2).

The fakes in test_cancel_http.py prove that an adapter registers its response with the control and that a close
releases the caller. They cannot prove that closing a socket wakes a read blocked on it, or that a request still
waiting for its first byte can be cut at all: that is what real sockets are for. Everything here talks to a server
on 127.0.0.1 and nothing else (conftest refuses any other address).

Two things are checked that no fake can check. A request that has no response yet is found in httpcore's pool and
its connection shut down, also when the connection is a kept-alive one used before; and an open response is shut
down so that the thread blocked reading it ends instead of waiting for its deadline. If a new httpcore moves the
private attributes `interrupt.Call` reads, `test_httpx_alone_has_what_the_helper_reads` (the tripwire) fails.
"""

from __future__ import annotations

import contextlib
import json
import socket
import threading

import httpx
import pytest

from glide.computer.control import RunControl, controlled
from glide.providers import interrupt
from glide.providers.chain import Chain, Slot
from glide.providers.errors import CANCELLED
from glide.providers.llm import LLM, OpenAICompatLLM
from glide.providers.tts import TTS, ElevenLabsTTS

WAIT = 3.0
ANSWER = json.dumps({"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}).encode()


class Wire:
    """A one-connection-at-a-time HTTP server on loopback that answers by script and records what happened to the socket.

    `script` is a list of steps, one per request on whichever connection arrives. A step is "answer" (a whole JSON
    reply), "headers" (the headers of a stream and one chunk, then silence), "silent" (nothing at all) or "speech"
    (the headers of an audio reply and one chunk of samples, then silence). After a step that goes silent the server
    waits on the connection: `gone` is set when the client closes it, `accepted` counts connections.
    """

    def __init__(self, *script: str) -> None:
        self.script = list(script)
        self.accepted = 0
        self.requests = 0
        self.entered = threading.Event()  # a request has arrived and is being held
        self.gone = threading.Event()  # the client closed the connection that was held
        self._listener = socket.socket()
        self._listener.bind(("127.0.0.1", 0))
        self._listener.listen(4)
        self.port = self._listener.getsockname()[1]
        threading.Thread(target=self._accept, daemon=True).start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def stop(self) -> None:
        with contextlib.suppress(OSError):
            self._listener.close()

    def _accept(self) -> None:
        while True:
            try:
                conn, _ = self._listener.accept()
            except OSError:
                return
            self.accepted += 1
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn: socket.socket) -> None:
        with conn:
            while self.script:
                if not self._read_request(conn):
                    return
                self.requests += 1
                step = self.script.pop(0)
                if step == "answer":
                    conn.sendall(
                        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n" % len(ANSWER) + ANSWER
                    )
                    continue
                if step == "headers":
                    event = b'data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n'
                    conn.sendall(
                        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n"
                        + b"%x\r\n" % len(event)
                        + event
                        + b"\r\n"
                    )
                if step == "speech":
                    conn.sendall(
                        b"HTTP/1.1 200 OK\r\nContent-Type: audio/pcm\r\nTransfer-Encoding: chunked\r\n\r\n4\r\n\x01\x00\x02\x00\r\n"
                    )
                self.entered.set()
                with contextlib.suppress(OSError):
                    while conn.recv(4096):  # held until the client goes away
                        pass
                self.gone.set()
                return

    @staticmethod
    def _read_request(conn: socket.socket) -> bool:
        head = b""
        while b"\r\n\r\n" not in head:
            try:
                part = conn.recv(4096)
            except OSError:
                return False
            if not part:
                return False
            head += part
        head, _, body = head.partition(b"\r\n\r\n")
        length = 0
        for line in head.split(b"\r\n")[1:]:
            name, _, value = line.partition(b":")
            if name.strip().lower() == b"content-length":
                length = int(value)
        while len(body) < length:
            body += conn.recv(length - len(body))
        return True


@pytest.fixture
def wire():
    servers = []

    def start(*script: str) -> Wire:
        servers.append(Wire(*script))
        return servers[-1]

    yield start
    for server in servers:
        server.stop()


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


def llm_at(server: Wire) -> LLM:
    client = OpenAICompatLLM("t:m", "m", server.url + "/v1", "k")
    return LLM(Chain("llm", [Slot("t:m", client)]))


def run_under(control: RunControl, call):
    def run():
        with controlled(control):
            return call()

    return in_thread(run)


MESSAGES = [{"role": "user", "content": "hello"}]


def test_a_request_waiting_for_its_first_byte_has_its_connection_shut_down_when_the_user_stops(wire):
    server = wire("silent")
    llm, control = llm_at(server), RunControl()
    done, box = run_under(control, lambda: llm.chat(MESSAGES))
    assert server.entered.wait(WAIT)
    control.cancel("stopped by the user")
    assert done.wait(WAIT) and box["error"].kind == CANCELLED
    assert server.gone.wait(WAIT), "the server is still holding a request the user stopped"
    assert server.requests == 1 and list(llm.chain.events) == []


def test_a_kept_alive_connection_that_is_waiting_for_its_first_byte_is_the_one_shut_down(wire):
    """The second request reuses the first one's connection, so no connect event names it: it is found in the pool."""
    server = wire("answer", "silent")
    llm, control = llm_at(server), RunControl()
    assert llm.chat(MESSAGES).text == "ok"
    done, box = run_under(control, lambda: llm.chat(MESSAGES))
    assert server.entered.wait(WAIT)
    control.cancel("stopped by the user")
    assert done.wait(WAIT) and box["error"].kind == CANCELLED
    assert server.gone.wait(WAIT)
    assert server.accepted == 1 and server.requests == 2  # the same connection, asked once each, and never again


def test_a_stream_whose_headers_have_arrived_is_shut_down_and_the_reading_thread_ends(wire):
    server = wire("headers")
    llm, control, seen = llm_at(server), RunControl(), []
    first = threading.Event()

    def consume():
        for delta in llm.stream(MESSAGES):
            seen.append(delta)
            first.set()

    done, box = run_under(control, consume)
    assert first.wait(WAIT)  # the headers and one delta have arrived; the thread is now blocked reading the body
    control.cancel("stopped by the user")
    assert done.wait(WAIT) and box["error"].kind == CANCELLED and seen == ["Hi"]
    assert server.gone.wait(WAIT)
    assert server.requests == 1


def test_speech_that_has_started_is_shut_down_and_the_server_sees_it_go(wire):
    server = wire("speech")
    client = ElevenLabsTTS("eleven", "m", "xi-key-0123456789", voice="v", base_url=server.url)
    tts, control, got = TTS(Chain("tts", [Slot("eleven", client)])), RunControl(), []
    first = threading.Event()

    def speak():
        for chunk in tts.stream("Hello there."):
            got.append(chunk.pcm)
            first.set()

    done, box = run_under(control, speak)
    assert first.wait(WAIT)
    control.cancel("stopped by the user")
    assert done.wait(WAIT) and box["error"].kind == CANCELLED
    assert server.gone.wait(WAIT) and got == [b"\x01\x00\x02\x00"]


def test_a_client_whose_request_was_cut_is_still_usable_for_the_next_one(wire):
    """Only the one connection is shut down, not the client: the next request works, on a new connection."""
    server = wire("silent", "answer")
    llm, control = llm_at(server), RunControl()
    done, _ = run_under(control, lambda: llm.chat(MESSAGES))
    assert server.entered.wait(WAIT)
    control.cancel()
    assert done.wait(WAIT) and server.gone.wait(WAIT)
    assert llm.chat(MESSAGES).text == "ok" and server.accepted == 2


def test_httpx_alone_has_what_the_helper_reads(wire):
    """THE tripwire for a new httpcore, and the only place a moved private attribute may be noticed: `interrupt.Call`
    degrades silently (it finds no socket and shuts nothing down), so this test is what fails instead.

    A request is held by a real server, and the helper must find, through httpcore's pool, the one socket that
    carries it, and shut it down so that the server sees the connection go."""
    server = wire("silent")
    with httpx.Client() as client:
        assert isinstance(client._transport._pool._requests, list)
        call = interrupt.Call(client)
        assert call.socket() is None  # nothing is in flight yet
        request = client.build_request("POST", server.url, content=b"{}", extensions=call.extensions)
        done, box = in_thread(lambda: client.send(request, stream=True))
        assert server.entered.wait(WAIT)
        assert call.socket() is not None, (
            "httpcore moved what interrupt.Call reads: a cancel can no longer reach a waiting request"
        )
        call.abort()
        assert server.gone.wait(WAIT), "the socket was found but shutting it down did not reach the server"
        assert done.wait(WAIT)
        assert isinstance(box.get("error"), httpx.HTTPError)  # the caller is woken with an error, not left to its deadline

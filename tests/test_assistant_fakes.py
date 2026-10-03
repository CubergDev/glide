"""Fakes shared by the assistant tests: a scripted LLM, TTS, STT and player, and a configuration to hold them.

Nothing here reaches a microphone, speaker, screen, network or key. Each fake records what it was asked, in
a log that is safe to append to from any thread, so a test can assert the ORDER of calls across the
assistant's threads. A few tests of the fakes themselves sit at the bottom so a broken fake cannot hide a
broken assistant.
"""

from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace

from glide.providers.base import ChatResult, Transcript, Usage
from glide.providers.config import ConfigError, NoUsableProvider

WAIT = 3.0  # how long a test waits for another thread before deciding it is not coming


def wait_until(predicate, timeout: float = WAIT) -> bool:
    """Poll until `predicate()` is true. For threads whose progress a test cannot block on directly."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


class Log(list):
    """A list that can be appended to from any thread, to record the order of events."""

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()

    def add(self, *entry) -> None:
        with self._lock:
            self.append(entry)


def route_json(route: str = "answer", reply: str = "", goal: str = "", language: str = "en") -> str:
    return json.dumps({"route": route, "reply": reply, "goal": goal, "language": language})


class FakeLLM:
    """The `LLM` facade: `chat` answers from `route`, `stream` yields `deltas`.

    `route` is the router's reply text (or an exception to raise, or a callable(messages) returning either).
    `gates` maps a delta's index to an Event the stream waits for before yielding it, so a test can prove
    that something happened while the stream was still going: `gate_results[i]` is whether the wait ended
    because the Event was set (True) or because it timed out (False).
    """

    def __init__(self, route=None, deltas=(), log: Log | None = None, gates=None) -> None:
        self.route = route if route is not None else route_json("answer", reply="Hello.")
        self.deltas = list(deltas)
        self.log = log if log is not None else Log()
        self.gates = dict(gates or {})
        self.gate_results: dict[int, bool] = {}
        self.chat_calls: list[dict] = []
        self.stream_calls: list[dict] = []
        self.stream_closed = False
        self.stream_error: Exception | None = None
        self.stream_error_after: int | None = None

    def chat(self, messages, *, hedge=None, **kw) -> ChatResult:
        self.chat_calls.append({"messages": list(messages), **kw})
        route = self.route(messages) if callable(self.route) else self.route
        if isinstance(route, BaseException):
            raise route
        return ChatResult(route, Usage(), "fake:fast", "fake", 0.0)

    def stream(self, messages, **kw):
        self.stream_calls.append({"messages": list(messages), **kw})
        try:
            for i, delta in enumerate(self.deltas):
                if self.stream_error is not None and i == self.stream_error_after:
                    raise self.stream_error
                gate = self.gates.get(i)
                if gate is not None:
                    self.gate_results[i] = gate.wait(WAIT)
                self.log.add("llm", i)
                yield delta
            if self.stream_error is not None and self.stream_error_after is None:
                raise self.stream_error
            self.log.add("llm_end")
        finally:
            self.stream_closed = True


class FakeTTS:
    """The `TTS` facade: each sentence becomes `chunks` chunks of two samples. `release` holds the first chunk back."""

    def __init__(
        self, log: Log | None = None, chunks: int = 2, rate: int = 24000, release: threading.Event | None = None
    ) -> None:
        self.log = log if log is not None else Log()
        self.chunks = chunks
        self.rate = rate
        self.release = release
        self.calls: list[tuple[str, str | None]] = []
        self.closed: list[str] = []
        self.error: Exception | None = None
        self._events: dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    def event_for(self, text: str) -> threading.Event:
        with self._lock:
            return self._events.setdefault(text, threading.Event())

    def stream(self, text, *, voice=None, language=None):
        self.calls.append((text, language))
        self.log.add("tts", text)
        self.event_for(text).set()
        try:
            if self.error is not None:
                raise self.error
            if self.release is not None:
                self.release.wait(WAIT)
            for _ in range(self.chunks):
                yield (b"\x01\x00\x02\x00", self.rate)
        finally:
            self.closed.append(text)


class FakePlayer:
    """The `Player` as the speaker uses it: records what was played, how often it was cancelled."""

    def __init__(self) -> None:
        self.played: list[tuple[bytes, int]] = []
        self.cancels = 0
        self.closed = False
        self.idle_waits = 0
        self._lock = threading.Lock()

    def play(self, pcm: bytes, sample_rate: int) -> None:
        with self._lock:
            self.played.append((pcm, sample_rate))

    def cancel(self) -> None:
        with self._lock:
            self.cancels += 1
            self.played.clear()

    def wait_idle(self, timeout=None) -> bool:
        self.idle_waits += 1
        return True

    def close(self) -> None:
        self.closed = True


class FakeSTT:
    """The `STT` facade. `stream` consumes the chunks it is given, yields `partials`, then `final` (or raises)."""

    def __init__(self, final: str = "", partials=(), language: str | None = "en", batch: str | None = None) -> None:
        self.final = final
        self.partials = list(partials)
        self.language = language
        self.batch = batch
        self.heard: list[bytes] = []
        self.stream_error: Exception | None = None
        self.batch_calls: list = []
        self.batch_error: Exception | None = None

    def stream(self, chunks, *, sample_rate=16000, language=None):
        for chunk in chunks:
            self.heard.append(chunk)
        if self.stream_error is not None:
            raise self.stream_error
        for text in self.partials:
            yield Transcript(text, self.language, "fake:stt", "fake", 0.0, partial=True)
        yield Transcript(self.final, self.language, "fake:stt", "fake", 0.0)

    def transcribe(self, audio, *, language=None, prompt=None, timeout=None):
        self.batch_calls.append(audio)
        if self.batch_error is not None:
            raise self.batch_error
        return Transcript(self.batch or "", self.language, "fake:stt", "fake", 0.0)


class FakeChain:
    def __init__(self, role: str) -> None:
        self.role = role
        self.events: list = []

    def status(self) -> list[dict]:
        return [
            {
                "name": f"{self.role}-slot",
                "pinned": False,
                "resting_s": 0.0,
                "failures": 0,
                "calls": 0,
                "avg_latency_s": None,
                "last_error": None,
            }
        ]


class FakeClassifier:
    """A classifier stand-in for `runner.run`: a context manager with `system_one`, as `TypeSafeClient` is."""

    def __init__(self, policy) -> None:
        self.inner = policy

    def __enter__(self):
        return self.inner

    def __exit__(self, *exc) -> bool:
        return False


class FakeConfig:
    """The `GlideConfig` surface the assistant and the CLI use, over fakes. A role left out is unusable."""

    source = "fake.toml"
    defaulted: tuple = ()
    warnings: tuple = ()

    def __init__(self, llm=None, stt=None, tts=None, classifier=None, writer=None, secret: str = "") -> None:
        self.fast = llm
        self.smart = llm
        self._stt = stt
        self._tts = tts
        self._classifier = classifier
        self._writer = writer
        self.secret = secret
        self.calls = SimpleNamespace(llm=0, stt=0, tts=0, classifier=0, writer=0)
        self.listeners: list = []
        self.pins: list[tuple] = []
        self.unpins: list[str] = []
        self.closed = False

    def __repr__(self) -> str:
        return "<FakeConfig>"

    def _need(self, role: str, value):
        if value is None:
            raise NoUsableProvider(f"no usable {role} provider. Set at least one of: FAKE_{role.upper()}_KEY.")
        return value

    def llm(self, role: str = "fast"):
        self.calls.llm += 1
        return self._need("llm." + role, self.fast if role == "fast" else self.smart)

    def stt(self):
        self.calls.stt += 1
        return self._need("stt", self._stt)

    def tts(self):
        self.calls.tts += 1
        return self._need("tts", self._tts)

    def classifier(self):
        self.calls.classifier += 1
        return self._need("classifier", self._classifier)

    def writer(self, *, timeout=None):
        self.calls.writer += 1
        return self._writer

    def chain(self, role: str):
        if role not in ("llm.fast", "stt"):
            raise ConfigError(f"no usable {role} provider.")
        return FakeChain(role)

    def on_switch(self, callback):
        self.listeners.append(callback)
        return lambda: self.listeners.remove(callback)

    def pin(self, role, name, strict=False):
        self.pins.append((role, name, strict))
        return f"{name}-full"

    def unpin(self, role):
        self.unpins.append(role)

    def scrub(self, text: str) -> str:
        return text.replace(self.secret, "***") if self.secret else text

    def close(self) -> None:
        self.closed = True


# -- the fakes, checked -----------------------------------------------------------------------------


def test_the_fake_llm_gates_a_delta_until_its_event_is_set():
    gate = threading.Event()
    llm = FakeLLM(deltas=["a", "b"], gates={1: gate})
    stream = llm.stream([])
    assert next(stream) == "a"
    threading.Timer(0.01, gate.set).start()
    assert next(stream) == "b"
    assert llm.gate_results == {1: True}
    stream.close()
    assert llm.stream_closed


def test_the_fake_tts_records_the_call_and_closes_its_stream():
    tts = FakeTTS(chunks=1)
    assert list(tts.stream("hi", language="en")) == [(b"\x01\x00\x02\x00", 24000)]
    assert tts.calls == [("hi", "en")] and tts.closed == ["hi"] and tts.event_for("hi").is_set()

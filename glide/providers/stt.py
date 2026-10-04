"""Speech to text: ElevenLabs Scribe (batch and realtime), OpenAI-style servers, and the `STT` facade.

`ElevenLabsSTT` transcribes a finished utterance over the batch endpoint and streams live audio over the
Scribe realtime websocket. `OpenAICompatSTT` speaks `POST {base_url}/audio/transcriptions`, which most
hosted and local servers imitate, and has no realtime mode, so its `stream()` waits for the end of the
audio and transcribes once. `STT` puts a `Chain` of either in front of the caller (chain.py), so the
caller never learns which vendor answered.

Everything about ElevenLabs' wire protocol (URLs, header, field and event names, model ids) lives in the
constants and the small builder and parser functions directly below. Vendors rename these; when one
does, that block is the only place to change. It was read from elevenlabs.io/docs on 2026-10-03.

The one subtle part is `STT.stream`. A chain fails over only before the first transcript arrives, but
by then the first provider has already pulled audio from the microphone, and a microphone is a one-shot
iterator. Without care the next provider would start in the middle of the utterance, so `ReplayBuffer`
records every chunk pulled and hands each attempt the recording followed by the rest of the source.
"""

from __future__ import annotations

import base64
import contextlib
import io
import itertools
import json
import queue
import threading
import time
import wave
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace
from urllib.parse import urlencode

import httpx
import websocket

from . import interrupt
from .base import Audio, ProviderSpec, STTClient, Transcript
from .chain import Chain
from .errors import ProviderError, redact, snippet, status_error
from .http import streaming, translate

# ---------------------------------------------------------------------------------------------------
# ElevenLabs wire protocol. Every vendor-specific string is here. Verified against the docs on
# 2026-10-03 except where a comment says "unverified".
# ---------------------------------------------------------------------------------------------------

ELEVENLABS_BASE_URL = "https://api.elevenlabs.io"  # regional hosts (api.us..., api.eu.residency...) go in the spec's base_url
BATCH_PATH = "/v1/speech-to-text"  # POST, multipart/form-data
REALTIME_PATH = "/v1/speech-to-text/realtime"  # websocket; https becomes wss, http becomes ws
AUTH_HEADER = "xi-api-key"  # the key goes here and nowhere else, never into the URL

# Fallbacks only, used when the configuration names no model. The batch endpoint and the realtime socket
# take different model ids, so the configured one is mapped: "scribe_v2_realtime" drives the socket and
# its batch twin is the same id without the suffix, and the other way round.
DEFAULT_BATCH_MODEL = "scribe_v2"
DEFAULT_REALTIME_MODEL = "scribe_v2_realtime"
REALTIME_SUFFIX = "_realtime"

PCM_RATES = (8000, 16000, 22050, 24000, 44100, 48000)  # the pcm_<rate> values of `audio_format`
MAX_CHUNK_SECONDS = 1  # unverified: the docs name a chunk size limit (`chunk_size_exceeded`) but not its value

COMMIT_MANUAL, COMMIT_VAD = "manual", "vad"  # `commit_strategy`; the docs name manual as the default

CLIENT_AUDIO = "input_audio_chunk"  # `message_type` of every client message, audio and commit alike
EVENT_PARTIAL = "partial_transcript"
EVENT_COMMITTED = "committed_transcript"
EVENT_COMMITTED_STAMPED = "committed_transcript_with_timestamps"  # also carries `language_code`

# Server error events. The docs say every one of them closes the connection. Which kind each maps to
# decides what a chain does next (errors.py): a provider that failed for its own reasons is skipped, and
# `bad_request` is never used here because it would stop the chain from trying anyone else.
ERROR_EVENT_KINDS = {
    "auth_error": "auth",
    "unaccepted_terms": "auth",
    "unaccepted_terms_error": "auth",  # the SDK knows this alias; the docs list `unaccepted_terms`
    "quota_exceeded": "rate_limit",
    "rate_limited": "rate_limit",
    "commit_throttled": "rate_limit",
    "error": "server",
    "transcriber_error": "server",
    "queue_overflow": "server",
    "resource_exhausted": "server",
    "session_time_limit_exceeded": "timeout",
    "invalid_request": "unsupported",  # rejected parameters, such as a language this model lacks
    "input_error": "content",
    "chunk_size_exceeded": "content",
    "insufficient_audio_activity": "content",
}
UNKNOWN_ERROR_KIND = "content"  # an unrecognised `*_error` event

# Language codes: ElevenLabs takes ISO 639-1 or 639-3 and answers in whichever it likes, so replies are
# folded to the short code a caller would pick a voice by. "yue" (Cantonese) has no short form.
LONG_TO_SHORT_LANGUAGE = {
    "eng": "en", "zho": "zh", "cmn": "zh", "jpn": "ja", "kor": "ko", "spa": "es", "fra": "fr", "deu": "de",
    "ita": "it", "por": "pt", "rus": "ru", "hin": "hi", "ara": "ar", "vie": "vi", "tha": "th", "ind": "id",
}  # fmt: skip

# Waits, in seconds, all overridable in a provider's options.
CONNECT_TIMEOUT_S = 10.0
FINAL_TIMEOUT_S = 10.0  # how long the final transcript may take after the last audio is sent
LANGUAGE_GRACE_S = 0.5  # how long to wait for a language-stamped copy of a commit that came without one
BATCH_TIMEOUT_S = 30.0
READER_JOIN_S = 1.0


def _base_code(language: str | None) -> str | None:
    """The first subtag of a language tag, lower case; None for 'auto' and for nothing."""
    if not language:
        return None
    code = language.strip().lower().replace("_", "-").split("-")[0]
    return None if code in ("", "auto", "und", "mul") else code


def elevenlabs_language(language: str | None) -> str | None:
    """The `language_code` to send, or None to leave the language to auto-detection.

    None is the default on purpose: forcing one language on speech that mixes two (English with
    Cantonese, say) degrades the part in the other. A caller who knows better passes a code.
    """
    code = _base_code(language)
    return code if code and code.isalpha() and 2 <= len(code) <= 3 else None


def openai_language(language: str | None) -> str | None:
    """OpenAI-style servers want ISO 639-1 and answer 400 to anything else, which a chain treats as the
    caller's own fault and does not retry elsewhere, so an unfamiliar code is dropped to auto-detect."""
    code = _base_code(language)
    return code if code and len(code) == 2 and code.isalpha() else None


def language_out(code: str | None) -> str | None:
    base = _base_code(code)
    return LONG_TO_SHORT_LANGUAGE.get(base, base) if base else None


def _origin(base_url: str) -> str:
    """The host part of a configured base url, with any trailing slash or /v1 removed."""
    url = (base_url or ELEVENLABS_BASE_URL).rstrip("/")
    return url.removesuffix("/v1")


def _query_value(value: object) -> object:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]  # unverified: repeated keys is how an "array" parameter is usually sent
    return str(value)


def realtime_url(
    base_url: str, *, model: str, sample_rate: int, language: str | None, commit_strategy: str, extra: dict | None = None
) -> str:
    """The websocket url for one session. Contains no secret; the key travels in a header.

    `extra` is the provider's `realtime_params` option, passed through for anything this module does not
    model (vad_threshold, keyterms, ...). The parameters set here win over it.
    """
    params: dict = dict(extra or {})
    params.update(model_id=model, audio_format=f"pcm_{sample_rate}", commit_strategy=commit_strategy)
    code = elevenlabs_language(language)
    if code:
        params["language_code"] = code
    else:
        # Without it the committed event does not say what was spoken. Unverified: how the server
        # behaves when `language_code` is simply absent; the docs imply it detects the language.
        params.setdefault("include_language_detection", True)
    origin = _origin(base_url)
    scheme, rest = origin.split("://", 1) if "://" in origin else ("https", origin)
    socket_scheme = "wss" if scheme == "https" else "ws"
    return f"{socket_scheme}://{rest}{REALTIME_PATH}?{urlencode({k: _query_value(v) for k, v in params.items()}, doseq=True)}"


def audio_message(pcm: bytes, sample_rate: int) -> str:
    """One audio chunk. `commit` is false: segments are committed by `commit_message`, or by the server."""
    return json.dumps(
        {
            "message_type": CLIENT_AUDIO,
            "audio_base_64": base64.b64encode(pcm).decode("ascii"),
            "commit": False,
            "sample_rate": sample_rate,
        }
    )


def commit_message(sample_rate: int) -> str:
    """Ends the segment: the same message with no audio and `commit` true (this is what the official SDK sends)."""
    return json.dumps({"message_type": CLIENT_AUDIO, "audio_base_64": "", "commit": True, "sample_rate": sample_rate})


def batch_fields(model: str, language: str | None, extra: dict | None = None) -> dict:
    """The multipart form fields besides the file. Audio-event tags such as "(laughter)" are switched off
    because they would be spoken to the assistant as words, and word timestamps are not wanted."""
    fields: dict = dict(extra or {})
    fields.update(model_id=model, tag_audio_events="false", timestamps_granularity="none")
    code = elevenlabs_language(language)
    if code:
        fields["language_code"] = code
    return fields


@dataclass
class Event:
    """One server event, reduced to what this module reads."""

    type: str
    text: str = ""
    language: str | None = None
    error: str = ""
    received: float = 0.0  # when the reader thread got it (time.monotonic); set by the session, not by the parser


def parse_event(raw: str | bytes) -> Event | None:
    """A server message as an Event, or None when it is not one this module can use."""
    try:
        data = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("message_type"), str):
        return None
    language = data.get("language_code")
    return Event(
        type=data["message_type"],
        text=str(data.get("text") or ""),
        language=language_out(language) if isinstance(language, str) else None,
        error=str(data.get("error") or data.get("message") or ""),
    )


def is_error_event(kind: str) -> bool:
    return kind in ERROR_EVENT_KINDS or kind.endswith("_error")


def error_for_event(event: Event, provider: str, key: str = "") -> ProviderError:
    """The error for a server error event. The event's text may quote the key, which is cut out."""
    kind = ERROR_EVENT_KINDS.get(event.type, UNKNOWN_ERROR_KIND)
    said = snippet(redact(event.error, [key], "***"))
    return ProviderError(f"{provider} reported {event.type}: {said}".rstrip(": "), kind=kind, provider=provider)


# ---------------------------------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------------------------------


def pcm_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    """Raw 16-bit mono PCM as an in-memory WAV file, which every batch endpoint accepts."""
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm[: len(pcm) - len(pcm) % 2])
    return out.getvalue()


def _frames(chunks: Iterable[bytes], sample_rate: int) -> Iterator[bytes]:
    """Chunks cut to whole 16-bit samples (a stray odd byte is carried to the next chunk, never sent
    alone) and to a size the server accepts."""
    limit = sample_rate * 2 * MAX_CHUNK_SECONDS
    carry = b""
    for chunk in chunks:
        data = carry + bytes(chunk)
        cut = len(data) - len(data) % 2
        data, carry = data[:cut], data[cut:]
        for start in range(0, len(data), limit):
            yield data[start : start + limit]


def _join(left: str, right: str) -> str:
    """Two pieces of one utterance, with a space between them unless the join falls inside Chinese or
    Japanese text, which is written without spaces."""
    if not left or not right:
        return left or right

    def unspaced(ch: str) -> bool:
        o = ord(ch)
        return 0x2E80 <= o <= 0x9FFF or 0xF900 <= o <= 0xFAFF or 0xFF00 <= o <= 0xFFEF

    return left + right if unspaced(left[-1]) or unspaced(right[0]) else f"{left} {right}"


# ---------------------------------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------------------------------


def _text(body: object) -> str:
    if isinstance(body, bytes):
        return body.decode("utf-8", "replace")
    return body if isinstance(body, str) else ""


class _HTTPAdapter:
    """What the two batch adapters share: one httpx client, error mapping, and a repr without the key."""

    def __init__(self, name: str, model: str, api_key: str, *, timeout_s: float, client: httpx.Client | None = None) -> None:
        self.name = name
        self.model = model
        self._key = api_key
        self._timeout_s = timeout_s
        self._http = client or httpx.Client()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, model={self.model!r})"

    def close(self) -> None:
        self._http.close()

    def _post(self, url: str, *, headers: dict, data: dict, wav: bytes, timeout: float | None) -> tuple[dict, float]:
        """POST the audio as multipart and return the JSON reply and how long it took, or raise a ProviderError."""
        started = time.monotonic()
        # ValueError: a base_url (or a key) that httpx cannot make a request of
        body = translate(self.name, lambda: self._exchange(url, headers, data, wav, timeout), also=(ValueError,))
        if not isinstance(body, dict):
            raise ProviderError(
                f"{self.name} answered with something other than a JSON object", kind="content", provider=self.name
            )
        return body, time.monotonic() - started

    def _exchange(self, url: str, headers: dict, data: dict, wav: bytes, timeout: float | None) -> object:
        """One request. The reply's JSON, or None when it is not JSON; a failed status is the ProviderError it maps to."""
        files = {"file": ("audio.wav", wav, "audio/wav")}
        with streaming(
            self._http, self.name, "POST", url=url, headers=headers, data=data, files=files, timeout=timeout or self._timeout_s
        ) as response:  # streamed, so a cancel can close the body too
            response.read()
            if not response.is_success:
                raise status_error(
                    response.status_code,
                    response.text,
                    provider=self.name,
                    headers=response.headers,
                    secrets=[self._key],
                    mark="***",
                )
            try:
                return response.json()
            except ValueError:
                return None

    def _empty(self, model: str, language: str | None) -> Transcript:
        return Transcript(
            text="", language=language_out(language), provider=f"{self.name}:{self.model}", model=model, latency_s=0.0
        )


class ElevenLabsSTT(_HTTPAdapter):
    """ElevenLabs Scribe. `transcribe` uses the batch endpoint, `stream` the realtime websocket.

    Options (all optional): `batch_model` and `realtime_model` (otherwise derived from the model, see
    REALTIME_SUFFIX), `commit_strategy` ("manual" by default: the segment ends when the audio does;
    "vad": the server ends it at a pause, and the stream stops at the first committed text),
    `timeout_s`, `connect_timeout_s`, `final_timeout_s`, and `realtime_params` / `batch_fields`, extra
    query parameters and form fields passed through untouched.
    """

    def __init__(
        self,
        name: str = "elevenlabs",
        model: str = "",
        api_key: str = "",
        *,
        base_url: str = "",
        options: dict | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        options = dict(options or {})
        model = model or DEFAULT_REALTIME_MODEL
        super().__init__(name, model, api_key, timeout_s=float(options.get("timeout_s", BATCH_TIMEOUT_S)), client=client)
        self.base_url = _origin(base_url)
        self.batch_model = options.get("batch_model") or model.removesuffix(REALTIME_SUFFIX)
        self.realtime_model = options.get("realtime_model") or (
            model if model.endswith(REALTIME_SUFFIX) else model + REALTIME_SUFFIX
        )
        self.commit_strategy = options.get("commit_strategy", COMMIT_MANUAL)
        if self.commit_strategy not in (COMMIT_MANUAL, COMMIT_VAD):
            raise ValueError(f"commit_strategy must be {COMMIT_MANUAL!r} or {COMMIT_VAD!r}, not {self.commit_strategy!r}")
        self.connect_timeout_s = float(options.get("connect_timeout_s", CONNECT_TIMEOUT_S))
        self.final_timeout_s = float(options.get("final_timeout_s", FINAL_TIMEOUT_S))
        self._realtime_params = dict(options.get("realtime_params") or {})
        self._batch_fields = dict(options.get("batch_fields") or {})

    def _need_key(self) -> None:
        if not self._key:
            raise ProviderError(f"{self.name} has no API key configured", kind="auth", provider=self.name)

    # -- batch ----------------------------------------------------------------------------------

    def transcribe(
        self, audio: Audio, *, language: str | None = None, prompt: str | None = None, timeout: float | None = None
    ) -> Transcript:
        """One finished utterance. `prompt` is ignored: the batch endpoint has no free-text prompt field."""
        self._need_key()
        if not audio.pcm:
            return self._empty(self.batch_model, language)
        reply, latency = self._post(
            f"{self.base_url}{BATCH_PATH}",
            headers={AUTH_HEADER: self._key},
            data=batch_fields(self.batch_model, language, self._batch_fields),
            wav=pcm_to_wav(audio.pcm, audio.sample_rate),
            timeout=timeout,
        )
        text = reply.get("text")
        if not isinstance(text, str):
            raise ProviderError(f"{self.name} answered without a transcript", kind="content", provider=self.name)
        spoken = reply.get("language_code")
        return Transcript(
            text=text.strip(),
            language=language_out(spoken) if isinstance(spoken, str) else language_out(language),
            provider=f"{self.name}:{self.model}",
            model=self.batch_model,
            latency_s=latency,
        )

    # -- realtime -------------------------------------------------------------------------------

    def stream(self, chunks: Iterable[bytes], *, sample_rate: int = 16000, language: str | None = None) -> Iterator[Transcript]:
        """Send PCM as it is captured, yield interim transcripts (partial=True), then one final.

        The segment ends when `chunks` ends (a commit is sent) or, with commit_strategy "vad", when the
        server hears a pause. Several committed segments (the server also commits on its own after about
        36 s) are joined into the one final text. This is a generator, so nothing is connected until the
        first item is asked for, and a failure to connect surfaces there, where a chain can fail over.
        """
        self._need_key()
        if sample_rate not in PCM_RATES:
            raise ProviderError(
                f"{self.name} cannot take {sample_rate} Hz audio (it takes {', '.join(map(str, PCM_RATES))})",
                kind="unsupported",
                provider=self.name,
            )
        url = realtime_url(
            self.base_url,
            model=self.realtime_model,
            sample_rate=sample_rate,
            language=language,
            commit_strategy=self.commit_strategy,
            extra=self._realtime_params,
        )
        run = _Run(self, self._open(url), sample_rate=sample_rate, language=language)
        try:
            with interrupt.closing(
                run.abort, self.name
            ):  # a stop closes the socket, which wakes the reader and the waiting caller
                sent = 0
                for pcm in _frames(chunks, sample_rate):
                    run.send(audio_message(pcm, sample_rate))
                    sent += len(pcm)
                    yield from run.pending()
                    if run.done:
                        break
                if not run.done:
                    if sent:
                        run.commit_at = time.monotonic()  # before the send: the reply can be faster than the next line
                        run.send(commit_message(sample_rate))
                        yield from run.wait()
                    else:
                        run.done = True  # no audio, so nothing to commit and nothing to wait for
                final = run.final()
                run.close()
                yield final
        finally:
            run.close()

    def _open(self, url: str):
        """Connect, or raise the ProviderError that says why not. The key is sent as a header."""
        failure: ProviderError | None = None
        ws = None
        try:
            ws = websocket.create_connection(
                url, timeout=self.connect_timeout_s, header=[f"{AUTH_HEADER}: {self._key}"], enable_multithread=True
            )
        except websocket.WebSocketBadStatusException as e:
            # Never str(e): it carries the response headers. Status and body are all that is needed.
            if e.status_code >= 400:
                failure = status_error(e.status_code, _text(e.resp_body), provider=self.name, secrets=[self._key], mark="***")
            else:
                failure = ProviderError(
                    f"{self.name} did not upgrade the connection ({e.status_code})", kind="transport", provider=self.name
                )
        except (websocket.WebSocketTimeoutException, TimeoutError):
            failure = ProviderError(f"{self.name} timed out connecting", kind="timeout", provider=self.name)
        except (websocket.WebSocketException, OSError, ValueError) as e:
            failure = ProviderError(
                f"{self.name} could not be reached ({type(e).__name__})", kind="transport", provider=self.name
            )
        if failure is not None:
            raise failure
        # The connect timeout would otherwise stay on the socket and make a silent pause in speech raise.
        # Deadlines are enforced by the queue the reader feeds, not by the socket.
        ws.settimeout(None)
        return ws


@dataclass
class _Segment:
    text: str
    language: str | None


class _Assembler:
    """The utterance so far: the committed segments plus the one still being spoken.

    A commit can arrive as `committed_transcript`, as `committed_transcript_with_timestamps`, or as both
    (the docs do not say which; unverified). A stamped event that repeats the text of a plain one just
    before it adds that segment's language and is not counted twice.
    """

    def __init__(self, requested: str | None) -> None:
        self.requested = language_out(requested)
        self.segments: list[_Segment] = []
        self.partial = ""
        self.shown = ""  # the last text handed to the caller as a partial
        self.awaiting_stamp = ""  # text of a plain commit whose stamped twin may still follow

    @property
    def text(self) -> str:
        out = ""
        for piece in (*(s.text for s in self.segments), self.partial):
            out = _join(out, piece)
        return out

    @property
    def language(self) -> str | None:
        """The language of the longest segment, which is the best guess for speech that mixes two."""
        stamped = [s for s in self.segments if s.language]
        return max(stamped, key=lambda s: len(s.text)).language if stamped else self.requested

    def committed(self, event: Event) -> None:
        text = event.text.strip()
        self.partial = ""
        if event.type == EVENT_COMMITTED_STAMPED and self.awaiting_stamp == text and text:
            self.awaiting_stamp = ""
            if self.segments and event.language:
                self.segments[-1].language = event.language
            return
        self.awaiting_stamp = text if event.type == EVENT_COMMITTED else ""
        if text:
            self.segments.append(_Segment(text, event.language))


class _Closed:
    """Put on the inbox when the server closed the socket."""


@dataclass
class _Broken:
    """Put on the inbox when reading the socket raised."""

    exc: BaseException


class _Run:
    """One live realtime session: the socket, the thread reading it, and the transcript being assembled.

    The caller's thread sends audio and pulls the microphone iterator, so a failed attempt never leaves a
    thread behind that is still pulling chunks the next attempt needs. Reading happens on a helper thread
    that fills `inbox`, which keeps the socket timeout out of it and gives every wait a deadline.
    """

    def __init__(self, client: ElevenLabsSTT, ws, *, sample_rate: int, language: str | None) -> None:
        self.client = client
        self.ws = ws
        self.sample_rate = sample_rate
        self.vad = client.commit_strategy == COMMIT_VAD
        self.inbox: queue.Queue = queue.Queue()
        self.asm = _Assembler(language)
        self.language_wanted = elevenlabs_language(language) is None
        self.started = time.monotonic()
        self.commit_at: float | None = None
        self.done = False
        self._closed = False
        self._reader = threading.Thread(target=self._read, name="stt-reader", daemon=True)
        self._reader.start()

    def _read(self) -> None:
        try:
            while True:
                raw = self.ws.recv()
                if not raw:  # websocket-client returns "" for a close frame
                    self.inbox.put(_Closed())
                    return
                event = parse_event(raw)
                if event is not None:
                    event.received = time.monotonic()
                    self.inbox.put(event)
        except BaseException as e:  # whatever ended the read is reported to the other thread
            self.inbox.put(_Broken(e))

    # -- sending --------------------------------------------------------------------------------

    def send(self, message: str) -> None:
        failure: ProviderError | None = None
        try:
            self.ws.send(message)
        except Exception as e:  # websocket-client raises its own classes, OSError and ssl errors
            interrupt.check(self.client.name)  # a socket closed by a cancel is not a transport fault
            failure = self._why_broken(e)
        if failure is not None:
            raise failure

    def _why_broken(self, exc: BaseException) -> ProviderError:
        """A send failed. The reader may be holding a better explanation (an error event the server sent
        before it hung up), so look for one briefly before settling for the bare transport failure."""
        deadline = time.monotonic() + 1.0
        while True:
            try:
                item = self.inbox.get(timeout=max(0.0, deadline - time.monotonic()))
            except queue.Empty:
                break
            if isinstance(item, Event) and is_error_event(item.type):
                return error_for_event(item, self.client.name, self.client._key)
            if isinstance(item, (_Closed, _Broken)):
                break
        return _socket_error(self.client.name, exc)

    # -- receiving ------------------------------------------------------------------------------

    def pending(self) -> Iterator[Transcript]:
        """Whatever has already arrived, without waiting."""
        while not self.done:
            try:
                item = self.inbox.get_nowait()
            except queue.Empty:
                return
            transcript = self._handle(item)
            if transcript is not None:
                yield transcript

    def wait(self) -> Iterator[Transcript]:
        """Block until the commit is answered, within the deadline."""
        deadline = (self.commit_at or time.monotonic()) + self.client.final_timeout_s
        while not self.done:
            try:
                item = self.inbox.get(timeout=max(0.0, deadline - time.monotonic()))
            except queue.Empty:
                raise ProviderError(
                    f"{self.client.name} gave no transcript within {self.client.final_timeout_s:g}s of the end of the audio",
                    kind="timeout",
                    provider=self.client.name,
                ) from None
            transcript = self._handle(item)
            if transcript is not None:
                yield transcript
        self._await_language()

    def _handle(self, item: object) -> Transcript | None:
        name = self.client.name
        if isinstance(item, (_Closed, _Broken)):
            interrupt.check(name)  # the socket was closed by a cancel, not by the server
        if isinstance(item, _Closed):
            raise ProviderError(
                f"{name} closed the connection before the transcript was complete", kind="transport", provider=name
            )
        if isinstance(item, _Broken):
            raise _socket_error(name, item.exc)
        assert isinstance(item, Event)
        if is_error_event(item.type):
            raise error_for_event(item, name, self.client._key)
        if item.type == EVENT_PARTIAL:
            self.asm.partial = item.text.strip()
            text = self.asm.text
            if not text or text == self.asm.shown:  # partials double as keepalives, so most add nothing
                return None
            self.asm.shown = text
            return self._transcript(text, partial=True)
        if item.type in (EVENT_COMMITTED, EVENT_COMMITTED_STAMPED):
            self.asm.committed(item)
            # A commit that arrived before ours was sent (the server also commits by itself) is one more
            # segment, however late this thread got round to reading it. Only one that arrived after
            # answers the commit.
            answers_commit = self.commit_at is not None and item.received >= self.commit_at
            self.done = answers_commit or (self.vad and bool(item.text.strip()))
        return None

    def _await_language(self) -> None:
        """A commit that arrives without a language may be followed at once by a stamped copy that has one."""
        if not (self.language_wanted and self.asm.awaiting_stamp):
            return
        deadline = time.monotonic() + LANGUAGE_GRACE_S
        while self.asm.awaiting_stamp:
            try:
                item = self.inbox.get(timeout=max(0.0, deadline - time.monotonic()))
            except queue.Empty:
                return
            if isinstance(item, (_Closed, _Broken)):
                return
            if isinstance(item, Event) and item.type == EVENT_COMMITTED_STAMPED:
                self.asm.committed(item)

    # -- results --------------------------------------------------------------------------------

    def _transcript(self, text: str, *, partial: bool) -> Transcript:
        since = self.started if partial or self.commit_at is None else self.commit_at
        return Transcript(
            text=text,
            language=self.asm.language,
            provider=f"{self.client.name}:{self.client.model}",
            model=self.client.realtime_model,
            latency_s=time.monotonic() - since,
            partial=partial,
        )

    def final(self) -> Transcript:
        """The whole utterance. Its latency is measured from the end of the audio, which is what a user waits for."""
        self.asm.partial = ""
        return self._transcript(self.asm.text, partial=False)

    def abort(self) -> None:
        """Cut the socket now, from any thread, and wake whoever waits on it. `close` still has to be called."""
        with contextlib.suppress(Exception):
            self.ws.abort()  # shuts the socket down, which a read blocked on it does notice
        with contextlib.suppress(Exception):
            self.ws.close(timeout=0)
        self.inbox.put(_Closed())

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # timeout=0: send the close frame and drop the socket without waiting for the server's reply.
        with contextlib.suppress(Exception):
            self.ws.close(timeout=0)
        self._reader.join(timeout=READER_JOIN_S)


def _socket_error(provider: str, exc: BaseException) -> ProviderError:
    """The error for a socket that failed while open. Only the exception's type is reported."""
    if isinstance(exc, (websocket.WebSocketTimeoutException, TimeoutError)):
        return ProviderError(f"{provider} timed out", kind="timeout", provider=provider)
    return ProviderError(f"{provider} connection failed ({type(exc).__name__})", kind="transport", provider=provider)


class OpenAICompatSTT(_HTTPAdapter):
    """Any server that imitates `POST {base_url}/audio/transcriptions` (OpenAI, Groq, a local whisper server).

    There is no realtime protocol here, so `stream` collects the audio and transcribes once. Options
    (all optional): `timeout_s`, `response_format` ("json" by default, the one every server has) and
    `fields`, extra form fields passed through untouched. The key may be empty for a local server.
    """

    def __init__(
        self,
        name: str = "openai",
        model: str = "",
        api_key: str = "",
        *,
        base_url: str = "",
        options: dict | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        if not base_url:
            raise ValueError(f"the {name} speech-to-text provider needs a base_url")
        options = dict(options or {})
        super().__init__(name, model, api_key, timeout_s=float(options.get("timeout_s", BATCH_TIMEOUT_S)), client=client)
        self.base_url = base_url.rstrip("/")
        self.response_format = options.get("response_format", "json")
        self._fields = dict(options.get("fields") or {})

    def transcribe(
        self, audio: Audio, *, language: str | None = None, prompt: str | None = None, timeout: float | None = None
    ) -> Transcript:
        if not audio.pcm:
            return self._empty(self.model, language)
        data: dict = {**self._fields, "model": self.model, "response_format": self.response_format}
        code = openai_language(language)
        if code:
            data["language"] = code
        if prompt:
            data["prompt"] = prompt
        reply, latency = self._post(
            f"{self.base_url}/audio/transcriptions",
            headers={"Authorization": f"Bearer {self._key}"} if self._key else {},
            data=data,
            wav=pcm_to_wav(audio.pcm, audio.sample_rate),
            timeout=timeout,
        )
        text = reply.get("text")
        if not isinstance(text, str):
            raise ProviderError(f"{self.name} answered without a transcript", kind="content", provider=self.name)
        spoken = reply.get("language")  # only a verbose_json reply has it
        return Transcript(
            text=text.strip(),
            language=language_out(spoken) if isinstance(spoken, str) else language_out(language),
            provider=f"{self.name}:{self.model}",
            model=self.model,
            latency_s=latency,
        )

    def stream(self, chunks: Iterable[bytes], *, sample_rate: int = 16000, language: str | None = None) -> Iterator[Transcript]:
        """Wait for the end of the audio, then transcribe it once and yield that as the only, final, transcript."""
        pcm = b"".join(chunks)
        yield self.transcribe(Audio(pcm, sample_rate), language=language)


# ---------------------------------------------------------------------------------------------------
# Facade
# ---------------------------------------------------------------------------------------------------


class ReplayBuffer:
    """A one-shot iterator of audio chunks that each attempt can read again from the start.

    Whoever pulls a chunk from the source records it first, so `attempt()` always returns every chunk
    pulled so far followed by the rest of the source. It holds one utterance of audio (about 2 MB a
    minute at 16 kHz) and is dropped with it. Only one attempt reads at a time, because a chain starts
    the next provider only after the previous one has failed.

    If the source itself raises (a microphone that died), that is not a provider failing and no other
    provider could do better, so the same error is raised again to every later attempt.
    """

    def __init__(self, source: Iterable[bytes]) -> None:
        self._source = iter(source)
        self._seen: list[bytes] = []
        self._error: BaseException | None = None

    def attempt(self) -> Iterator[bytes]:
        return itertools.chain(list(self._seen), self._pull())

    def _pull(self) -> Iterator[bytes]:
        if self._error is not None:
            raise self._error
        while True:
            try:
                chunk = next(self._source)
            except StopIteration:
                return
            except BaseException as e:
                self._error = e
                raise
            self._seen.append(chunk)
            yield chunk


class STT:
    """Speech to text through a chain of providers, with the methods of a single `STTClient`.

    Every transcript's `provider` is the name of the chain slot that produced it, so callers and
    `glide doctor` see the same names the chain reports in its switch events.
    """

    def __init__(self, chain: Chain[STTClient]) -> None:
        self.chain = chain

    def transcribe(
        self, audio: Audio, *, language: str | None = None, prompt: str | None = None, timeout: float | None = None
    ) -> Transcript:
        """A finished utterance. Transcribing twice is harmless, so the chain may race two providers when
        its policy sets `hedge_after_s`."""

        def attempt(slot) -> Transcript:
            result = slot.client.transcribe(audio, language=language, prompt=prompt, timeout=timeout)
            return replace(result, provider=slot.name)

        return self.chain.call(attempt, hedge=True)

    def stream(self, chunks: Iterable[bytes], *, sample_rate: int = 16000, language: str | None = None) -> Iterator[Transcript]:
        """Interim transcripts and then a final one, from the first provider that produces any.

        Failover happens only before the first transcript (chain.py), and each provider is given all the
        audio so far, so the one that takes over does not start mid-utterance.
        """
        replay = ReplayBuffer(chunks)

        def attempt(slot) -> Iterator[Transcript]:
            inner = iter(slot.client.stream(replay.attempt(), sample_rate=sample_rate, language=language))
            try:
                for transcript in inner:
                    yield replace(transcript, provider=slot.name)
            finally:
                close = getattr(inner, "close", None)
                if close is not None:
                    close()  # an abandoned stream closes its socket now, not whenever it is collected

        return self.chain.stream(attempt)


def build_client(spec: ProviderSpec, model: str, api_key: str, options: dict | None = None) -> STTClient:
    """The speech-to-text client for one provider spec. `options` override the spec's own options.

    Raises ValueError for a kind this module does not know, which is a mistake in glide.toml, not a
    provider failing. A missing key is not an error here: the client raises an `auth` ProviderError when
    it is used, so a chain can skip it and carry on with the next provider.
    """
    merged = {**spec.options, **(options or {})}
    if spec.kind == "elevenlabs":
        return ElevenLabsSTT(spec.name, model, api_key, base_url=spec.base_url, options=merged)
    if spec.kind == "openai_compat":
        return OpenAICompatSTT(spec.name, model, api_key, base_url=spec.base_url, options=merged)
    raise ValueError(f"no speech-to-text adapter for provider kind {spec.kind!r} (known: elevenlabs, openai_compat)")

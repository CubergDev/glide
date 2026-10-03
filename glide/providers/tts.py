"""Text to speech: three vendors behind one shape, and a facade that fails over between them.

`ElevenLabsTTS` and `OpenAICompatTTS` stream raw PCM over HTTP, so the first audio can play while the
rest is still being made. `MacSayTTS` is the fallback that needs no network and no key: the macOS
`say` command, run for synthesis only (it writes a file and never plays), then read back as PCM.
`TTS` holds a `Chain` of them (chain.py) and tags every chunk with the sample rate of the slot that
produced it, because that differs between slots and a player has to know it. `build_client` makes one
client from a `[providers.<name>]` entry.

Audio is raw PCM, 16-bit little-endian, mono (base.py). A chunk is always a whole number of samples:
an HTTP read can end in the middle of a 16-bit sample, and a player fed half a sample plays noise.
"""

from __future__ import annotations

import contextlib
import json
import re
import subprocess
import tempfile
import threading
import time
import wave
from collections.abc import Callable, Iterable, Iterator, Mapping
from functools import partial
from pathlib import Path
from typing import NamedTuple
from urllib.parse import quote

import httpx

from . import interrupt
from .base import ProviderSpec, SpeechAudio, TTSClient
from .chain import Chain
from .errors import ProviderError, from_exception, from_status, snippet

# -- Protocol constants -----------------------------------------------------------------------------
# Everything a vendor decides lives in this block, so a changed endpoint or model is a one-place edit.
# Read from the vendors' own documentation on 2026-10-03; what could not be confirmed is marked.

# ElevenLabs, "Stream speech" (elevenlabs.io/docs/api-reference/text-to-speech/stream).
#   POST {base}/v1/text-to-speech/{voice_id}/stream, auth header `xi-api-key`, JSON body.
#   `voice_id` is a path parameter: a voice from the account's library, not a name.
#   Query `output_format=pcm_<rate>` asks for raw PCM. The rates listed were 8000, 16000, 22050,
#   24000, 32000, 44100 and 48000; the docs say 44.1 kHz needs a Pro subscription or above, so the
#   default is lower. Sample format 16-bit little-endian is stated on the capabilities page only.
#   Body: `text` (required), `model_id` (optional; the server picks its own default when omitted, so
#   an empty configured model sends none), `language_code` (documented as ISO 639-1, ignored by a
#   model that does not support it, and "not supported for multilingual_v2 models"), `voice_settings`.
#   Latency: query `optimize_streaming_latency` 0-4 is marked deprecated in the docs. The other
#   latency lever is the choice of model, which is configuration here. Both `language_code` and the
#   latency query are therefore opt-in, never sent by default: a rejected parameter is a 422, which
#   the chain treats as our own fault and does not fail over, so one bad option would silence the voice.
ELEVENLABS_BASE_URL = "https://api.elevenlabs.io"
ELEVENLABS_STREAM_PATH = "/v1/text-to-speech/{voice_id}/stream"
ELEVENLABS_KEY_HEADER = "xi-api-key"
ELEVENLABS_DEFAULT_FORMAT = "pcm_24000"
ELEVENLABS_PCM_FORMAT = re.compile(r"pcm_(\d{4,5})")

# OpenAI-style speech, "Create speech" (the guide at developers.openai.com/api/docs/guides/text-to-speech;
# the API-reference page answered 403, so this rests on the guide).
#   POST {base}/audio/speech, `Authorization: Bearer <key>`, JSON {model, input, voice, response_format}.
#   response_format "pcm" is documented as raw samples at 24 kHz, 16-bit signed little-endian, no header.
#   `stream_format` is not sent: the plain audio stream is what a PCM player wants.
#   Other servers that copy this API may answer at another rate; `sample_rate` is an option for them.
OPENAI_BASE_URL = "https://api.openai.com/v1"
OPENAI_SPEECH_PATH = "/audio/speech"
OPENAI_PCM_FORMAT = "pcm"
OPENAI_PCM_RATE = 24000
OPENAI_DEFAULT_VOICE = "alloy"  # a voice name from the guide's list; servers with other names set `voice`

# macOS `say` (man say). `-o file` writes instead of playing, `--file-format=WAVE` and
# `--data-format=LEI16@<rate>` ask for 16-bit little-endian integer PCM at <rate> Hz, `-f -` reads the
# text from standard input so it is never an option or part of the process list, `-v ?` lists voices.
# The man page warns that not every voice supports every file format, so the read-back checks the file.
SAY_BINARY = "say"
SAY_SAMPLE_RATE = 22050
SAY_DEFAULT_VOICES = {"yue": "Sinji", "zh-hk": "Sinji", "zh": "Tingting", "cmn": "Tingting", "en": "Samantha"}

DEFAULT_TIMEOUT_S = 30.0
ERROR_BODY_LIMIT = 4096  # bytes of an error reply read before giving up on the rest
SAMPLE_BYTES = 2  # 16-bit mono
# A server that echoes our text back has it replaced in the error message. Short text is left alone:
# replacing "No." everywhere in a reply would garble it, and says nothing worth hiding.
MIN_SCRUBBED_TEXT = 10


class SpeechChunk(NamedTuple):
    """A piece of audio and the rate it plays at: `pcm, rate = chunk` or `chunk.pcm`, `chunk.sample_rate`.

    The rate travels with every chunk because a failover can change it from one chunk (or one
    sentence) to the next, and a player that kept the first rate would play the rest too fast or slow.
    """

    pcm: bytes
    sample_rate: int


# -- Shared helpers ---------------------------------------------------------------------------------


def _language_keys(language: str | None, default: str = "") -> list[str]:
    """Lookup keys for a language tag, most specific first: 'zh-HK' gives ['zh-hk', 'zh']."""
    tag = (language or default).strip().lower().replace("_", "-")
    if not tag:
        return []
    return [tag, tag.split("-")[0]] if "-" in tag else [tag]


def _normalized(mapping: Mapping | None) -> dict[str, str]:
    return {str(k).strip().lower().replace("_", "-"): str(v) for k, v in (mapping or {}).items() if v}


def _lookup(mapping: Mapping[str, str], language: str | None, default: str = "") -> str | None:
    for key in _language_keys(language, default):
        if mapping.get(key):
            return mapping[key]
    return None


def _pick_voice(voice: str | None, language: str | None, voices: Mapping[str, str], default: str | None) -> str | None:
    """An explicit voice, else the voice configured for the language, else the configured default."""
    return voice or _lookup(voices, language) or default or None


def _rate_from(source: Mapping, default: int) -> int:
    value = source.get("sample_rate", default)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"sample_rate must be a positive whole number of Hz, not {value!r}")
    return value


@contextlib.contextmanager
def _provider_errors(name: str) -> Iterator[None]:
    """Turn what httpx raises into a ProviderError that carries no request detail.

    The new error is raised `from None`: an httpx exception holds the request, and a traceback that
    chained to it could show headers.
    """
    try:
        yield
    except (httpx.HTTPError, httpx.InvalidURL, httpx.StreamError) as exc:
        interrupt.check(name)  # a connection closed by a cancel is not a transport fault
        error = from_exception(exc, provider=name)
        raise error from None


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after", "")
    return float(value) if value.replace(".", "", 1).isdigit() else None


def _describe(raw: str, secrets: Iterable[str]) -> str:
    """What a server said, safe for an error message: the reason only, and no secret or input echoed.

    Validation errors (FastAPI style `detail` lists) keep where and why but drop `input`, which can be
    the text we sent. Whatever is left has each secret replaced.
    """
    text = raw
    try:
        body = json.loads(raw)
    except ValueError:
        body = None
    if isinstance(body, dict):
        detail = body.get("detail", body.get("error", body.get("message")))
        if isinstance(detail, list):
            parts = []
            for item in detail:
                if isinstance(item, dict):
                    where = ".".join(map(str, item["loc"])) if isinstance(item.get("loc"), list) else ""
                    parts.append(f"{where}: {item.get('msg', '')}".strip(": "))
            text = "; ".join(parts)
        elif isinstance(detail, dict):
            text = str(detail.get("message") or detail.get("status") or "")
        elif isinstance(detail, str):
            text = detail
    for secret in secrets:
        if secret:
            text = text.replace(secret, "***")
    return text


class _HTTPStreamTTS:
    """What the two HTTP vendors share: one streaming POST, sample-aligned chunks, and safe errors.

    A subclass says what the request is (`_request`). The key is kept in a private attribute, never
    put in a URL or a message, and `__repr__` leaves it out.
    """

    name: str
    model: str
    sample_rate: int

    def __init__(
        self,
        name: str,
        model: str,
        api_key: str,
        *,
        base_url: str,
        timeout: float = DEFAULT_TIMEOUT_S,
        client: httpx.Client | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.name = name
        self.model = model
        self.timeout = timeout
        self._key = (api_key or "").strip()  # an environment variable often ends in a newline
        self._base_url = base_url.rstrip("/")
        self._clock = clock
        self._owns_client = client is None
        self._http = client or httpx.Client()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, model={self.model!r}, sample_rate={self.sample_rate})"

    def close(self) -> None:
        if self._owns_client:
            self._http.close()

    # -- what a vendor decides ------------------------------------------------------------------

    def _request(self, text: str, voice: str | None, language: str | None) -> dict:
        """Keyword arguments for `httpx.Client.stream("POST", ...)`: url, params, headers, json."""
        raise NotImplementedError

    def _header_key(self, *, required: bool) -> str:
        """The key as a header value, or an auth error if it is missing or could not be sent as one."""
        key = self._key
        if not key and not required:
            return ""
        if not key or not key.isascii() or not key.isprintable() or any(c.isspace() for c in key):
            raise ProviderError(f"{self.name} has no usable API key", kind="auth", provider=self.name)
        return key

    # -- the shared call ------------------------------------------------------------------------

    def stream(self, text: str, *, voice: str | None = None, language: str | None = None) -> Iterator[bytes]:
        """PCM chunks at `sample_rate` as they arrive. Nothing is sent for blank text."""
        return self._chunks(text, voice, language, self.timeout)

    def synthesize(
        self, text: str, *, voice: str | None = None, language: str | None = None, timeout: float | None = None
    ) -> SpeechAudio:
        """All the audio at once. `timeout` bounds the whole call, not just each wait."""
        limit = self.timeout if timeout is None else timeout
        deadline = self._clock() + limit
        parts: list[bytes] = []
        with contextlib.closing(self._chunks(text, voice, language, limit)) as chunks:
            for pcm in chunks:
                parts.append(pcm)
                if self._clock() > deadline:
                    raise ProviderError(f"{self.name} timed out", kind="timeout", provider=self.name)
        return SpeechAudio(b"".join(parts), self.sample_rate)

    def _chunks(self, text: str, voice: str | None, language: str | None, timeout: float) -> Iterator[bytes]:
        if not text.strip():
            return
        request = self._request(text, voice, language)
        carry = b""
        produced = False
        call = interrupt.Call(self._http)
        with (
            _provider_errors(self.name),
            interrupt.closing(call.abort, self.name),  # until the first byte there is no response to close
            self._http.stream("POST", timeout=timeout, extensions=call.extensions, **request) as response,
            interrupt.closing(partial(interrupt.abort_response, response), self.name),
        ):
            if response.status_code != 200:
                raise self._status_error(response, text)
            for data in response.iter_bytes():
                data = carry + data
                keep = len(data) % SAMPLE_BYTES
                data, carry = (data[:-keep], data[-keep:]) if keep else (data, b"")
                if data:
                    produced = True
                    yield data
        interrupt.check(self.name)  # a body that a cancel cut short ends quietly: it is not a finished sentence
        if not produced:
            raise ProviderError(f"{self.name} returned no audio", kind="content", provider=self.name)

    def _status_error(self, response: httpx.Response, text: str) -> ProviderError:
        """The error for a non-200 reply. A streaming body is not read until asked, so read it first."""
        body = bytearray()
        for part in response.iter_bytes():
            body += part
            if len(body) >= ERROR_BODY_LIMIT:
                break
        raw = bytes(body[:ERROR_BODY_LIMIT]).decode("utf-8", "replace")
        return from_status(
            response.status_code,
            snippet(_describe(raw, (self._key, text if len(text) >= MIN_SCRUBBED_TEXT else ""))),
            provider=self.name,
            retry_after=_retry_after(response),
        )


# -- Adapters ---------------------------------------------------------------------------------------


class ElevenLabsTTS(_HTTPStreamTTS):
    """ElevenLabs streaming text to speech, as raw PCM.

    `output_format` is "pcm_<rate>" and fixes `sample_rate`. A voice is an ElevenLabs voice id; there is
    no sensible default, so one must come from the call, `voices` (language to voice id) or `voice`.
    A call with none is "unsupported", which lets the chain move to a voice that can speak.
    """

    def __init__(
        self,
        name: str,
        model: str,
        api_key: str,
        *,
        base_url: str = "",
        output_format: str = ELEVENLABS_DEFAULT_FORMAT,
        voice: str | None = None,
        voices: Mapping[str, str] | None = None,
        language_codes: Mapping[str, str] | None = None,
        voice_settings: Mapping | None = None,
        optimize_streaming_latency: int | None = None,
        timeout: float = DEFAULT_TIMEOUT_S,
        client: httpx.Client | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        match = ELEVENLABS_PCM_FORMAT.fullmatch(output_format or "")
        if not match:
            raise ValueError(f"output_format must be raw PCM such as {ELEVENLABS_DEFAULT_FORMAT!r}, not {output_format!r}")
        super().__init__(
            name, model, api_key, base_url=base_url or ELEVENLABS_BASE_URL, timeout=timeout, client=client, clock=clock
        )
        self.output_format = output_format
        self.sample_rate = int(match.group(1))
        self._voice = voice
        self._voices = _normalized(voices)
        self._language_codes = _normalized(language_codes)
        self._voice_settings = dict(voice_settings) if voice_settings else None
        self._latency = optimize_streaming_latency

    def _request(self, text: str, voice: str | None, language: str | None) -> dict:
        voice_id = _pick_voice(voice, language, self._voices, self._voice)
        if not voice_id:
            raise ProviderError(
                f"{self.name} has no voice configured for language {language or 'default'!r}",
                kind="unsupported",
                provider=self.name,
            )
        params: dict[str, str | int] = {"output_format": self.output_format}
        if self._latency is not None:
            params["optimize_streaming_latency"] = self._latency
        body: dict = {"text": text}
        if self.model:
            body["model_id"] = self.model
        code = _lookup(self._language_codes, language)
        if code:
            body["language_code"] = code
        if self._voice_settings:
            body["voice_settings"] = self._voice_settings
        return {
            "url": self._base_url + ELEVENLABS_STREAM_PATH.format(voice_id=quote(voice_id, safe="")),
            "params": params,
            "headers": {ELEVENLABS_KEY_HEADER: self._header_key(required=True)},
            "json": body,
        }


class OpenAICompatTTS(_HTTPStreamTTS):
    """Any server that answers OpenAI's `POST /audio/speech`, asked for raw PCM.

    OpenAI documents its PCM as 24 kHz, 16-bit signed little-endian, so `sample_rate` defaults to
    24000. A server that answers at another rate is told so with `sample_rate`. The key is optional:
    a local server may need none.
    """

    def __init__(
        self,
        name: str,
        model: str,
        api_key: str,
        *,
        base_url: str = "",
        sample_rate: int = OPENAI_PCM_RATE,
        voice: str | None = None,
        voices: Mapping[str, str] | None = None,
        instructions: str | None = None,
        speed: float | None = None,
        timeout: float = DEFAULT_TIMEOUT_S,
        client: httpx.Client | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        super().__init__(name, model, api_key, base_url=base_url or OPENAI_BASE_URL, timeout=timeout, client=client, clock=clock)
        self.sample_rate = _rate_from({"sample_rate": sample_rate}, OPENAI_PCM_RATE)
        self._voice = voice or OPENAI_DEFAULT_VOICE
        self._voices = _normalized(voices)
        self._instructions = instructions
        self._speed = speed

    def _request(self, text: str, voice: str | None, language: str | None) -> dict:
        body: dict = {
            "model": self.model,
            "input": text,
            "voice": _pick_voice(voice, language, self._voices, self._voice),
            "response_format": OPENAI_PCM_FORMAT,
        }
        if self._instructions:
            body["instructions"] = self._instructions
        if self._speed is not None:
            body["speed"] = self._speed
        key = self._header_key(required=False)
        return {
            "url": self._base_url + OPENAI_SPEECH_PATH,
            "headers": {"Authorization": f"Bearer {key}"} if key else {},
            "json": body,
        }


def parse_voice_list(listing: str) -> list[tuple[str, str]]:
    """(name, locale) for each line of `say -v ?`, such as ('Eddy (English (US))', 'en_US').

    A line is the voice name, spaces, the locale, then `# sample sentence`. Names hold spaces and
    parentheses, so the locale is taken from the end of what comes before the '#'.
    """
    voices = []
    for line in listing.splitlines():
        head = line.partition("#")[0].rstrip()
        parts = head.rsplit(None, 1)
        if len(parts) == 2 and re.fullmatch(r"[A-Za-z]{2,3}[_-][A-Za-z0-9]+", parts[1]):
            voices.append((parts[0].strip(), parts[1]))
    return voices


class MacSayTTS:
    """The macOS `say` command as a TTS client: synthesis to a temporary file, never playback.

    It needs no network and no key, which makes it the voice that is always there. Each call writes a
    16-bit mono WAVE file with `-o` into a temporary directory and reads the samples back. The voice
    comes from the call, else from `voices` for the language, else from `SAY_DEFAULT_VOICES`. If that
    voice is not installed the call is "unsupported": reading Cantonese in a Mandarin voice, or any
    text in a voice for another language, is wrong, so no other voice is substituted. A language with
    no voice at all is the same error. With no language given, `default_language` is used.

    `runner` is `subprocess.run` unless a test passes a fake.
    """

    def __init__(
        self,
        name: str = "macos_say",
        model: str = "",
        *,
        voices: Mapping[str, str] | None = None,
        default_language: str = "en",
        sample_rate: int = SAY_SAMPLE_RATE,
        rate_wpm: int | None = None,
        timeout: float = DEFAULT_TIMEOUT_S,
        chunk_bytes: int = 8192,
        runner: Callable[..., subprocess.CompletedProcess] | None = None,
    ) -> None:
        self.name = name
        self.model = model or SAY_BINARY
        self.sample_rate = _rate_from({"sample_rate": sample_rate}, SAY_SAMPLE_RATE)
        self.timeout = timeout
        self._voices = {**SAY_DEFAULT_VOICES, **_normalized(voices)}
        self._default_language = default_language
        self._rate_wpm = rate_wpm
        self._chunk_bytes = max(SAMPLE_BYTES, chunk_bytes - chunk_bytes % SAMPLE_BYTES)
        self._runner = runner
        self._installed: dict[str, str] | None = None  # lowercase name to the name as `say` lists it
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return f"MacSayTTS(name={self.name!r}, sample_rate={self.sample_rate})"

    def stream(self, text: str, *, voice: str | None = None, language: str | None = None) -> Iterator[bytes]:
        """The finished audio in chunks. `say` synthesizes faster than it speaks, so this is not delayed."""
        audio = self.synthesize(text, voice=voice, language=language)
        for start in range(0, len(audio.pcm), self._chunk_bytes):
            yield audio.pcm[start : start + self._chunk_bytes]

    def synthesize(
        self, text: str, *, voice: str | None = None, language: str | None = None, timeout: float | None = None
    ) -> SpeechAudio:
        if not text.strip():
            return SpeechAudio(b"", self.sample_rate)
        chosen = self._resolve(self._wanted_voice(voice, language), language)
        with tempfile.TemporaryDirectory(prefix="glide-say-") as directory:
            target = Path(directory) / "speech.wav"
            argv = [
                SAY_BINARY,
                "-v",
                chosen,
                "-o",
                str(target),
                "--file-format=WAVE",
                f"--data-format=LEI16@{self.sample_rate}",
            ]
            if self._rate_wpm:
                argv += ["-r", str(self._rate_wpm)]
            argv += ["-f", "-"]
            self._run(argv, text.encode("utf-8"), self.timeout if timeout is None else timeout)
            return SpeechAudio(self._read(target), self.sample_rate)

    # -- voices ---------------------------------------------------------------------------------

    def _wanted_voice(self, voice: str | None, language: str | None) -> str:
        wanted = voice or _lookup(self._voices, language, self._default_language)
        if not wanted:
            raise ProviderError(
                f"{self.name} has no voice configured for language {language or self._default_language!r}",
                kind="unsupported",
                provider=self.name,
            )
        return wanted

    def _resolve(self, wanted: str, language: str | None) -> str:
        """The installed voice named `wanted`, or an "unsupported" error. A miss on a list read some
        time ago is checked once more against a fresh list, in case the voice was installed since."""
        listed_now = self._installed is None
        hit = self._installed_voices().get(wanted.casefold())
        if hit is None and not listed_now:
            hit = self._installed_voices(refresh=True).get(wanted.casefold())
        if hit is None:
            raise ProviderError(
                f"the macOS voice {wanted!r} is not installed (language {language or self._default_language!r}); "
                "install it under System Settings, Spoken Content, or configure another voice",
                kind="unsupported",
                provider=self.name,
            )
        return hit

    def _installed_voices(self, *, refresh: bool = False) -> dict[str, str]:
        with self._lock:
            if self._installed is None or refresh:
                listing = self._run([SAY_BINARY, "-v", "?"], None, self.timeout).decode("utf-8", "replace")
                voices = parse_voice_list(listing)
                if not voices:
                    raise ProviderError(
                        f"{self.name} could not read the list of installed voices", kind="content", provider=self.name
                    )
                self._installed = {name.casefold(): name for name, _ in voices}
            return self._installed

    # -- the process and the file ---------------------------------------------------------------

    def _run(self, argv: list[str], stdin: bytes | None, timeout: float) -> bytes:
        """Run `say` and return its standard output. Bytes in and out: the locale must not touch Chinese text."""
        runner = self._runner or subprocess.run
        try:
            done = runner(argv, input=stdin, capture_output=True, timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            raise ProviderError(f"{self.name} timed out", kind="timeout", provider=self.name) from None
        except OSError as exc:  # no `say` on this machine, or it cannot be run
            raise ProviderError(
                f"{self.name} could not run say ({type(exc).__name__})", kind="unsupported", provider=self.name
            ) from None
        if done.returncode != 0:
            stderr = done.stderr.decode("utf-8", "replace") if isinstance(done.stderr, bytes) else str(done.stderr or "")
            raise ProviderError(
                f"say exited {done.returncode}: {snippet(stderr)}".rstrip(": "), kind="server", provider=self.name
            )
        return done.stdout if isinstance(done.stdout, bytes) else str(done.stdout or "").encode()

    def _read(self, path: Path) -> bytes:
        """The samples of the file `say` wrote, checked to be what was asked for.

        Playing 16 kHz audio as if it were 22.05 kHz is not a small error, so a file in any other
        shape is refused rather than passed on.
        """
        try:
            with wave.open(str(path), "rb") as wav:
                shape = (wav.getnchannels(), wav.getsampwidth(), wav.getcomptype(), wav.getframerate())
                if shape != (1, SAMPLE_BYTES, "NONE", self.sample_rate):
                    raise ProviderError(
                        f"say wrote {shape[0]} channel(s), {shape[1] * 8}-bit, {shape[3]} Hz; "
                        f"expected mono 16-bit at {self.sample_rate} Hz",
                        kind="content",
                        provider=self.name,
                    )
                return wav.readframes(wav.getnframes())
        except (wave.Error, EOFError, FileNotFoundError):
            raise ProviderError(f"{self.name} produced no readable audio file", kind="content", provider=self.name) from None


# -- Facade and factory -----------------------------------------------------------------------------


class TTS:
    """Text to speech over a chain of clients. Callers never see which vendor spoke.

    `stream` and `speak_sentences` yield `SpeechChunk(pcm, sample_rate)`, and `synthesize` returns a
    `SpeechAudio` that already carries its rate. A failover can change the rate, so a player must read
    it from each chunk. Which slot answered is on `chain.events` (a switch is never silent).
    """

    def __init__(self, chain: Chain[TTSClient]) -> None:
        self.chain = chain

    def stream(self, text: str, *, voice: str | None = None, language: str | None = None) -> Iterator[SpeechChunk]:
        """Audio chunks as the first working provider makes them.

        Failover happens only until the first chunk. The text is simply sent again to the next
        provider, since nothing has been played. After the first chunk a failure is raised to the
        caller as a ProviderError of kind "stream".
        """

        def attempt(slot) -> Iterator[SpeechChunk]:
            rate = slot.client.sample_rate
            for pcm in slot.client.stream(text, voice=voice, language=language):
                yield SpeechChunk(pcm, rate)

        return self.chain.stream(attempt)

    def synthesize(
        self, text: str, *, voice: str | None = None, language: str | None = None, timeout: float | None = None
    ) -> SpeechAudio:
        """All the audio at once, from the first provider that can make it."""
        return self.chain.call(lambda slot: slot.client.synthesize(text, voice=voice, language=language, timeout=timeout))

    def speak_sentences(
        self, sentences: Iterable[str], *, voice: str | None = None, language: str | None = None
    ) -> Iterator[SpeechChunk]:
        """Each sentence's audio in order, pulling the next sentence only when the last one's audio is used up.

        Pass the sentences as an iterator that fills while a model is still writing, and play the first
        chunks while the rest of the reply is being made. Pulling lazily is the point: nothing here
        reads ahead. Blank sentences are skipped. Each sentence is its own `stream` call, so a failover
        before a sentence's first chunk moves only that sentence, and a provider that dies in the
        middle of a sentence raises a ProviderError of kind "stream" to the caller, who has already
        played part of it and chooses whether to go on.
        """
        for sentence in sentences:
            if sentence.strip():
                yield from self.stream(sentence, voice=voice, language=language)


def build_client(spec: ProviderSpec, model: str, api_key: str, options: Mapping | None = None) -> TTSClient:
    """One TTS client for a `[providers.<name>]` entry. `options` override `spec.options`.

    Options by kind (anything else is ignored):
      elevenlabs: voice, voices {language: voice id}, language_codes {language: code}, voice_settings,
        optimize_streaming_latency, output_format ("pcm_<rate>"), timeout.
      openai_compat: voice, voices, instructions, speed, sample_rate, timeout.
      macos_say: voices {language: voice name}, default_language, sample_rate, rate (words per minute), timeout.
    A bad value raises ValueError here, at build time. `macos_say` ignores the key and base_url.
    """
    opts = {**spec.options, **(options or {})}
    timeout = float(opts.get("timeout", DEFAULT_TIMEOUT_S))
    if spec.kind == "elevenlabs":
        return ElevenLabsTTS(
            spec.name,
            model,
            api_key,
            base_url=spec.base_url,
            output_format=opts.get("output_format", ELEVENLABS_DEFAULT_FORMAT),
            voice=opts.get("voice"),
            voices=opts.get("voices"),
            language_codes=opts.get("language_codes"),
            voice_settings=opts.get("voice_settings"),
            optimize_streaming_latency=opts.get("optimize_streaming_latency"),
            timeout=timeout,
        )
    if spec.kind == "openai_compat":
        return OpenAICompatTTS(
            spec.name,
            model,
            api_key,
            base_url=spec.base_url,
            sample_rate=_rate_from(opts, OPENAI_PCM_RATE),
            voice=opts.get("voice"),
            voices=opts.get("voices"),
            instructions=opts.get("instructions"),
            speed=opts.get("speed"),
            timeout=timeout,
        )
    if spec.kind == "macos_say":
        return MacSayTTS(
            spec.name,
            model,
            voices=opts.get("voices"),
            default_language=opts.get("default_language", "en"),
            sample_rate=_rate_from(opts, SAY_SAMPLE_RATE),
            rate_wpm=opts.get("rate"),
            timeout=timeout,
        )
    raise ValueError(f"no text-to-speech adapter for kind {spec.kind!r}; use elevenlabs, openai_compat or macos_say")

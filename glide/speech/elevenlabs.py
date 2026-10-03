"""An optional realtime text-to-speech adapter: one websocket per sentence, raw PCM out, in the `TTSClient` shape.

This is a second ElevenLabs text-to-speech adapter beside the HTTP one in providers/tts.py, kept apart
because it is optional and because its wire protocol is the least certain part of the voice stack: it was
ported from a working-tree snapshot and has not been checked against the vendor's documentation in this
repository's history (see docs/live-checks/voice.md). Nothing here is a default: the base URL, model, voice
and sample rate all come from the `[providers.<name>]` entry, the key from the environment variable it
names, and the module is registered by `build_client` only when the providers owner adds the kind.

Speech to text needs no adapter here: `providers/stt.ElevenLabsSTT.stream` already speaks the realtime
transcription socket, and `Assistant.handle_audio` reaches it through `config.stt()`.

Adapters raise only `ProviderError`; a message carries a status or an exception's type, never the key, a
header or the text being spoken.
"""

from __future__ import annotations

import base64
import binascii
import contextlib
import json
import time
from collections.abc import Iterator, Mapping
from urllib.parse import urlencode

import websocket

from ..providers.base import SpeechAudio
from ..providers.errors import ProviderError, from_status

KEY_HEADER = "xi-api-key"
STREAM_PATH = "/v1/text-to-dialogue/stream-input"  # unverified against the vendor's docs; see the module docstring
PCM_RATES = (8000, 16000, 22050, 24000, 44100, 48000)
CONNECT_TIMEOUT_S = 15.0
EVENT_TIMEOUT_S = 15.0
TOTAL_TIMEOUT_S = 60.0
MAX_AUDIO_S = 120.0


class RealtimeElevenLabsTTS:
    def __init__(
        self,
        name: str,
        model: str,
        api_key: str,
        *,
        base_url: str,
        sample_rate: int = 16000,
        voice: str | None = None,
        voices: Mapping[str, str] | None = None,
        language_codes: Mapping[str, str] | None = None,
    ) -> None:
        if sample_rate not in PCM_RATES:
            raise ValueError(f"sample_rate must be one of {', '.join(map(str, PCM_RATES))}")
        if not base_url.startswith(("https://", "http://")):
            raise ValueError("a realtime text-to-speech provider needs a base_url")
        self.name, self.model, self.sample_rate = name, model, sample_rate
        self._key = (api_key or "").strip()
        scheme, rest = base_url.rstrip("/").split("://", 1)
        self._origin = f"{'wss' if scheme == 'https' else 'ws'}://{rest}"
        self._voice = voice
        self._voices = {k.lower(): v for k, v in (voices or {}).items()}
        self._codes = {k.lower(): v for k, v in (language_codes or {}).items()}

    def __repr__(self) -> str:
        return f"RealtimeElevenLabsTTS(name={self.name!r}, model={self.model!r}, sample_rate={self.sample_rate})"

    def _error(self, message: str, kind: str, status: int | None = None) -> ProviderError:
        return ProviderError(f"{self.name} {message}", kind=kind, provider=self.name, status=status)

    def _voice_for(self, voice: str | None, language: str | None) -> str:
        code = (language or "").lower().replace("_", "-")
        chosen = voice or self._voices.get(code) or self._voices.get(code.split("-")[0]) or self._voice
        if not chosen:
            raise self._error("has no voice configured for this language", "unsupported")
        return chosen

    def _url(self, language: str | None) -> str:
        params = {"model_id": self.model, "output_format": f"pcm_{self.sample_rate}"}
        code = (language or "").lower().replace("_", "-")
        if language and (mapped := self._codes.get(code) or self._codes.get(code.split("-")[0])):
            params["language_code"] = mapped
        return f"{self._origin}{STREAM_PATH}?{urlencode(params)}"

    def stream(self, text: str, *, voice: str | None = None, language: str | None = None) -> Iterator[bytes]:
        """PCM chunks at `sample_rate` as they arrive. Nothing is connected until the first chunk is asked for."""
        if not text.strip():
            return
        if not self._key or not self._key.isascii() or not self._key.isprintable():
            raise self._error("has no usable API key", "auth")
        voice_id = self._voice_for(voice, language)
        ws = self._connect(self._url(language))
        try:
            ws.send(json.dumps({"voices": [voice_id]}))
            ws.send(json.dumps({"inputs": [{"text": text, "voice_id": voice_id, "new_turn": True}]}))
            ws.send(json.dumps({"close_socket": True}))
            yield from self._audio(ws)
        except ProviderError:
            raise
        except Exception as exc:
            raise self._error(f"connection failed ({type(exc).__name__})", "stream") from exc
        finally:
            with contextlib.suppress(Exception):  # the socket is being thrown away: a failed close leaves nothing to do
                ws.close(timeout=0)

    def _connect(self, url: str):
        try:
            return websocket.create_connection(url, timeout=CONNECT_TIMEOUT_S, header=[f"{KEY_HEADER}: {self._key}"])
        except websocket.WebSocketBadStatusException as exc:
            raise from_status(exc.status_code, "", provider=self.name) from None  # never str(exc): it carries headers
        except (websocket.WebSocketTimeoutException, TimeoutError):
            raise self._error("timed out connecting", "timeout") from None
        except (websocket.WebSocketException, OSError, ValueError) as exc:
            raise self._error(f"could not be reached ({type(exc).__name__})", "transport") from None

    def _audio(self, ws) -> Iterator[bytes]:
        deadline = time.monotonic() + TOTAL_TIMEOUT_S
        limit = int(MAX_AUDIO_S * self.sample_rate * 2)
        total = 0
        carry = b""
        ws.settimeout(EVENT_TIMEOUT_S)
        while True:
            if time.monotonic() > deadline:
                raise self._error("timed out", "timeout")
            try:
                raw = ws.recv()
            except (websocket.WebSocketTimeoutException, TimeoutError):
                raise self._error("timed out", "timeout") from None
            if not raw:
                raise self._error("closed the connection before the audio was complete", "stream" if total else "transport")
            try:
                event = json.loads(raw)
            except ValueError:
                raise self._error("sent something that is not JSON", "content") from None
            if not isinstance(event, dict):
                continue
            if event.get("error"):
                raise self._error("reported an error", "server")  # its text may echo the request: not repeated
            if event.get("audio"):
                try:
                    data = carry + base64.b64decode(event["audio"], validate=True)
                except (binascii.Error, ValueError):
                    raise self._error("sent audio that is not base64", "content") from None
                keep = len(data) % 2
                data, carry = (data[:-keep], data[-keep:]) if keep else (data, b"")
                total += len(data)
                if total > limit:
                    raise self._error("sent more audio than a sentence can hold", "content")
                if data:
                    yield data
            if event.get("is_final"):
                if not total:
                    raise self._error("returned no audio", "content")
                return

    def synthesize(
        self, text: str, *, voice: str | None = None, language: str | None = None, timeout: float | None = None
    ) -> SpeechAudio:
        return SpeechAudio(b"".join(self.stream(text, voice=voice, language=language)), self.sample_rate)

    def close(self) -> None:
        return None

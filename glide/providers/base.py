"""The shapes every provider adapter speaks, one set per kind of job.

An adapter is a small client for one vendor. Each module (llm.py, stt.py, tts.py, classifier.py)
offers `build_client(spec, model, api_key, options)` returning one such client, and a facade class
that holds a `Chain` of them (chain.py) and exposes the same methods, so callers never see which
vendor answered. Adapters raise only `ProviderError` (errors.py).

Audio is raw PCM: 16-bit little-endian, mono. Nothing here knows about files or playback.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class ProviderSpec:
    """A vendor endpoint, as written under `[providers.<name>]` in glide.toml.

    `kind` picks the adapter: "openai_compat" (any OpenAI-style server), "elevenlabs", "typesafe"
    (any /v1/systemone server: TypeSafe Jev, Laya, Kev), "macos_say". `api_key_env` names the
    environment variable that holds the key; the key itself is never stored in a spec or a log.
    """

    name: str
    kind: str
    base_url: str = ""
    api_key_env: str = ""
    options: dict = field(default_factory=dict)


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0


@dataclass
class ChatResult:
    text: str
    usage: Usage
    provider: str  # the slot that answered, e.g. "openrouter:deepseek/deepseek-v4.1-flash"
    model: str
    latency_s: float
    ttft_s: float | None = None  # time to the first token, when the call streamed
    finish_reason: str | None = None
    logprobs: list[dict] | None = None  # OpenAI-style `content` entries, when asked for and given


class LLMClient(Protocol):
    name: str
    model: str

    def chat(
        self,
        messages: Sequence[dict],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        schema: dict | None = None,
        logprobs: bool = False,
        timeout: float | None = None,
    ) -> ChatResult:
        """`messages` are OpenAI chat messages. Content may be a string or a list of parts:
        {"type": "text", "text": ...} and {"type": "image_url", "image_url": {"url": "data:..."}}.
        `schema`, when given, is a JSON Schema the reply must satisfy; a provider without native
        support asks for JSON in the prompt instead, and the reply is still returned as text."""
        ...

    def stream(
        self,
        messages: Sequence[dict],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        timeout: float | None = None,
    ) -> Iterator[str]:
        """Text deltas as they arrive. Reasoning text is never yielded."""
        ...


@dataclass(frozen=True)
class Audio:
    pcm: bytes
    sample_rate: int = 16000


@dataclass
class Transcript:
    text: str
    language: str | None
    provider: str
    model: str
    latency_s: float
    partial: bool = False  # True for a streaming interim result that may still change


class STTClient(Protocol):
    name: str
    model: str

    def transcribe(
        self, audio: Audio, *, language: str | None = None, prompt: str | None = None, timeout: float | None = None
    ) -> Transcript:
        """One finished utterance in, one transcript out."""
        ...

    def stream(self, chunks: Iterable[bytes], *, sample_rate: int = 16000, language: str | None = None) -> Iterator[Transcript]:
        """PCM chunks in as they are captured; interim transcripts (partial=True) and then a final one out."""
        ...


@dataclass
class SpeechAudio:
    pcm: bytes
    sample_rate: int


class TTSClient(Protocol):
    name: str
    model: str
    sample_rate: int

    def synthesize(
        self, text: str, *, voice: str | None = None, language: str | None = None, timeout: float | None = None
    ) -> SpeechAudio: ...

    def stream(self, text: str, *, voice: str | None = None, language: str | None = None) -> Iterator[bytes]:
        """PCM chunks at `sample_rate`, starting as soon as the first audio exists."""
        ...

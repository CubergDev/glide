"""Provider-neutral text generation: the one request and result shape the writer speaks (`ModelProvider`).

The implementation is `glide.providers.writer_client.ChainWriter`, over the glide.toml chains. Nothing here knows a
vendor, a wire format or an environment variable.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Protocol

from .control import RunControl


class GenerationError(Exception):
    """A generation failed; its partial output must not be used."""

    halt = False


class GenerationUnavailable(GenerationError):
    """The run must stop; account/transport failures are not another action choice."""

    halt = True


@dataclass(frozen=True)
class GenerationRequest:
    model: str
    instructions: str
    text: str
    schema: dict
    image: bytes | None = None
    deadline_s: float = 30
    max_tokens: int = 1024  # used only by providers that support a token limit
    reasoning: str | None = None
    role: str = "writer"


@dataclass(frozen=True)
class TokenUsage:
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True)
class GenerationResult:
    text: str
    model: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    completed: bool = True


class ModelProvider(Protocol):
    def generate(self, request: GenerationRequest, cancel: RunControl | None = None) -> GenerationResult: ...


def image_url(data: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(data).decode()

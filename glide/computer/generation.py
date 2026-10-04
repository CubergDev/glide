"""Provider-neutral text generation. Only provider implementations know wire formats."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from typing import Protocol

from .control import RunControl, checkpoint


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
    stop_reason: str | None = None  # why the model stopped, when the provider says; a strict reader checks it


class ModelProvider(Protocol):
    def generate(self, request: GenerationRequest, cancel: RunControl | None = None) -> GenerationResult: ...


def image_url(data: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(data).decode()


class AnthropicProvider:
    """Explicit legacy provider; Anthropic shapes stop at this boundary."""

    def __init__(self, client, *, custom: bool = False):
        self.client = client
        self.messages = client.messages
        self.base_url = client.base_url
        self.custom = custom

    def generate(self, request: GenerationRequest, cancel: RunControl | None = None) -> GenerationResult:
        import anthropic

        checkpoint(cancel)
        content = []
        if request.image is not None:
            content.append(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/png", "data": base64.b64encode(request.image).decode()},
                }
            )
        content.append({"type": "text", "text": request.text})
        system = request.instructions
        extra = {}
        if getattr(self, "custom", False):
            system += "\n\nAnswer with a single JSON object and nothing else, matching this schema:\n" + json.dumps(
                request.schema
            )
            extra["thinking"] = {"type": "disabled"}
        try:
            reply = self.messages.create(
                model=request.model,
                timeout=request.deadline_s,
                max_tokens=request.max_tokens,
                system=system,
                messages=[{"role": "user", "content": content}],
                output_config={"format": {"type": "json_schema", "schema": request.schema}},
                **extra,
            )
        except anthropic.APIError as error:
            raise GenerationError("Anthropic request failed") from error
        checkpoint(cancel)
        usage = getattr(reply, "usage", None)
        return GenerationResult(
            "".join(b.text for b in reply.content if b.type == "text"),
            getattr(reply, "model", None) or request.model,
            TokenUsage(
                (getattr(usage, "input_tokens", 0) or 0) + (getattr(usage, "cache_creation_input_tokens", 0) or 0),
                getattr(usage, "cache_read_input_tokens", 0) or 0,
                getattr(usage, "output_tokens", 0) or 0,
            ),
            completed=getattr(reply, "stop_reason", None) not in {"max_tokens", "refusal"},
            stop_reason=getattr(reply, "stop_reason", None),
        )

"""The writer, served by the Glide LLM chains instead of one vendor's SDK.

glide/computer/writer.py talks to its model through one call shaped like the Anthropic SDK's,
`writer.messages.create(...)`, and reads text blocks back. `ChainWriter` answers that call from two
facades (anything with the `.chat` of `LLMClient`, see base.py), so the writer gets failover, pinning
and visible switches (chain.py) without a line of writer.py knowing which vendor replied.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace
from typing import Protocol

import httpx

from glide.computer.config import answer_model

from .base import ChatResult, Usage
from .errors import ProviderError

# A stand-in address: `provider(writer)` in writer.py reads `writer.base_url` for the startup line.
BASE_URL = "glide://writer-chain/"


class ChatFacade(Protocol):
    """What `ChainWriter` needs from an LLM facade: its `chat`, which fails over inside itself."""

    def chat(
        self,
        messages: Sequence[dict],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        schema: dict | None = None,
        timeout: float | None = None,
    ) -> ChatResult: ...


class ChainWriter:
    """The one call the writer makes, `messages.create`, routed through two LLM facades.

    Anthropic message blocks go in and come out as OpenAI chat messages for the facade, and the
    `ChatResult` comes back as a reply with text blocks and an Anthropic-style `usage`, so
    `MeteredWriter` (calls.py) counts it like any other writer.

    Routing: the writer picks its model with `answering=True` for the answer that reads the screen
    (config.answer_model()) and with the writer model (config.writer_model()) for everything else. A
    request goes to `smart` when its `model` equals config.answer_model() at the time of the call,
    and to `fast` otherwise. The model id itself is never sent anywhere, because the chain decides
    which model answers. One consequence: if CLICKER_WRITER_MODEL and CLICKER_ANSWER_MODEL name the
    same id, every request is taken for an answer and goes to `smart`.

    `thinking` and `reasoning` are accepted and dropped, as OpenAIWriter does: a facade has no such
    switch, and each provider's own options say how it thinks. A reply with no text becomes one empty
    text block, which writer.py then refuses as "no usable JSON". A `ProviderError` from a facade
    propagates untouched (writer.py turns it into `WriterError`), and so does any other exception,
    since that is a bug. This class holds no key and never logs: a failover is already a `SwitchEvent`
    on the facade's chain.

    Usage convention: `Usage.input_tokens` from a facade counts every prompt token, cached ones
    included, as OpenAI-style endpoints report it. The reply splits them the Messages way
    (`input_tokens` uncached, `cache_read_input_tokens` cached), which is what MeteredWriter reads.
    """

    # writer.py asks any endpoint that is not Anthropic's own to spell the schema out in the prompt
    # as well, since it may ignore output_config. A chain can reach any vendor, so it always asks.
    spells_out_schema = True

    def __init__(self, fast: ChatFacade, smart: ChatFacade, *, timeout: float | None = None):
        self._fast = fast
        self._smart = smart
        self._timeout = timeout
        self.base_url = httpx.URL(BASE_URL)
        self.messages = SimpleNamespace(create=self._create)

    def __repr__(self) -> str:
        # Never the facades: what one holds is its own business, and a repr ends up in logs.
        return f"<ChainWriter {self.base_url}>"

    def _create(
        self,
        *,
        model: str,
        max_tokens: int,
        system: str | Sequence[dict],
        messages: Sequence[dict],
        output_config: dict | None = None,
        thinking=None,
        reasoning: str | None = None,
    ):
        facade = self._smart if model == answer_model() else self._fast
        result = facade.chat(
            chat_messages(system, messages),
            max_tokens=max_tokens,
            temperature=0.0,
            schema=_schema(output_config),
            timeout=self._timeout,
        )
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=result.text or "")],
            usage=_usage(result.usage),
            model=result.model,
            provider=result.provider,  # the slot that answered, for anyone debugging a run
        )


def chat_messages(system: str | Sequence[dict], messages: Sequence[dict]) -> list[dict]:
    """A Messages request as OpenAI chat messages: the system prompt first, then each turn.

    A request that cannot be translated is our own fault, so it is a `bad_request`: the step is
    refused and no provider is tried with it.
    """
    system_text = system if isinstance(system, str) else "\n\n".join(b["text"] for b in system if b.get("type") == "text")
    out = [{"role": "system", "content": system_text}] if system_text else []
    return out + [_chat_message(m) for m in messages]


def _chat_message(message: dict) -> dict:
    """One Messages turn as a chat one: text parts stay text, an image becomes a data URL."""
    content = message["content"]
    if isinstance(content, str):
        return {"role": message["role"], "content": content}
    parts = []
    for block in content:
        if block.get("type") == "text":
            parts.append({"type": "text", "text": block["text"]})
        elif block.get("type") == "image":
            parts.append({"type": "image_url", "image_url": {"url": _image_url(block.get("source") or {})}})
        else:
            raise ProviderError(f"no chat form for a {block.get('type')!r} block", kind="bad_request")
    return {"role": message["role"], "content": parts}


def _image_url(source: dict) -> str:
    if source.get("type") == "url" or ("url" in source and "data" not in source):
        return source["url"]
    if "media_type" not in source or "data" not in source:
        raise ProviderError("an image block needs a base64 source with media_type and data, or a url", kind="bad_request")
    return f"data:{source['media_type']};base64,{source['data']}"


def _schema(output_config: dict | None) -> dict | None:
    """The JSON Schema the reply must satisfy, when the request asks for a JSON-schema format."""
    format_ = (output_config or {}).get("format") or {}
    return format_.get("schema") if format_.get("type") == "json_schema" else None


def _usage(usage: Usage | None) -> SimpleNamespace:
    """A facade's usage in the Messages shape, with cached tokens counted apart from the rest."""
    prompt = getattr(usage, "input_tokens", 0) or 0
    cached = getattr(usage, "cached_input_tokens", 0) or 0
    return SimpleNamespace(
        input_tokens=max(prompt - cached, 0),
        cache_read_input_tokens=cached,
        cache_creation_input_tokens=0,
        output_tokens=getattr(usage, "output_tokens", 0) or 0,
    )

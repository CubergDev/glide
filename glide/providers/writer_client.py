"""The writer, served by the Glide LLM chains instead of one vendor's SDK.

glide/computer/writer.py talks to its model through one neutral call, `writer.generate(GenerationRequest)`
(glide/computer/generation.py), and reads text back. `ChainWriter` answers it from LLM facades (anything with the
`.chat` of `LLMClient`, see base.py), so the writer gets failover, pinning and visible switches (chain.py) without
a line of writer.py knowing which vendor replied.

Every request is one non-streaming `chat`. A stream that dies half-way cannot be told from a finished answer
by the words it carries, so a reply is used whole or not at all: a connection that drops mid-body is a
`ProviderError` of kind `transport` inside the chain, the chain moves on to the next slot (a visible
`SwitchEvent`), and no partial text ever leaves this module.
"""

from __future__ import annotations

from collections.abc import Mapping

from glide.computer.control import RunControl, checkpoint, controlled, current_control
from glide.computer.generation import (
    GenerationError,
    GenerationRequest,
    GenerationResult,
    GenerationUnavailable,
    TokenUsage,
    image_url,
)

from .base import ChatFacade, Usage
from .errors import AllProvidersFailed, ProviderError

# Which LLM facade serves which `GenerationRequest.role`. A role is a job, never a model: the chain decides which
# model answers, from glide.toml. `research_*` is every stage of the research supervisor.
FAST_ROLES = frozenset({"writer", "task_routing"})
SMART_ROLES = frozenset({"recovery"})
PLANNER_ROLES = frozenset({"planner"})
RESEARCH_PREFIX = "research"

# A reply that ended for one of these reasons is cut short, not finished. (An ordinary stop, or a provider
# that names no reason, is finished.)
CUT_SHORT = frozenset({"length", "content_filter", "error"})


class ChainWriter:
    """The writer's one call, `generate`, routed through the LLM facades by the request's role.

    Routing: `writer` and `task_routing` go to `fast`, `recovery` (the answer that reads the screen) to `smart`,
    `planner` to the planner facade and every `research*` role to the research facade. Those two default to
    `smart` when glide.toml configures no chain for them (config.py). A role nobody serves is a `bad_request`:
    our own mistake, so the step is refused and no provider is tried.

    `reasoning` and `model` on the request are not sent: a facade has no such switch, the model of a slot is in
    glide.toml, and each slot's own options say how it thinks. `deadline_s` bounds one request, in total;
    `timeout` (and `deadlines`, per facade: fast, smart, planner, research) cap it from configuration.

    Errors: `bad_request` stays a `GenerationError` (this step is refused). Every other `ProviderError` is
    `GenerationUnavailable`, which halts the run: by then the chain has tried every slot it had, an account is
    spent or a key refused, and what is left for the user to do is outside the run. The messages are the chain's
    own, which carry slot names and error kinds and never a key, a header or a request.

    This class holds no key and never logs: a failover is already a `SwitchEvent` on the facade's chain.

    Usage: a facade's `Usage` already keeps cached tokens apart from `input_tokens`, so it passes through as is.
    """

    def __init__(
        self,
        fast: ChatFacade,
        smart: ChatFacade,
        *,
        planner: ChatFacade | None = None,
        research: ChatFacade | None = None,
        timeout: float | None = None,
        deadlines: Mapping[str, float] | None = None,
    ):
        self._fast = fast
        self._smart = smart
        self._planner = planner or smart
        self._research = research or smart
        self._timeout = timeout
        self._deadlines = dict(deadlines or {})

    def __repr__(self) -> str:
        # Never the facades: what one holds is its own business, and a repr ends up in logs.
        return "<ChainWriter>"

    def describe(self) -> str:
        """The slots behind each facade, by name, for the startup line. Names only: no key, no URL."""
        parts = []
        for label, facade in (
            ("fast", self._fast),
            ("smart", self._smart),
            ("planner", self._planner),
            ("research", self._research),
        ):
            names = getattr(getattr(facade, "chain", None), "names", None)
            parts.append(f"{label}: {', '.join(names) if names else '?'}")
        return "; ".join(parts)

    def generate(self, request: GenerationRequest, cancel: RunControl | None = None) -> GenerationResult:
        checkpoint(cancel)
        label, facade = self._route(request.role)
        try:
            with controlled(cancel if cancel is not None else current_control()):  # the chains cancel what is in flight
                result = facade.chat(
                    _messages(request),
                    max_tokens=request.max_tokens,
                    temperature=0.0,
                    schema=request.schema,
                    timeout=self._deadline(request, label),
                )
        except ProviderError as error:
            checkpoint(cancel, wait=False)  # a call a cancel cut short is an `Abort`, not an unavailable writer
            raise _generation_error(error) from error
        checkpoint(cancel)  # an answer that arrives after a cancel is dropped, never used
        return GenerationResult(
            result.text or "",
            result.model,
            _usage(result.usage),
            completed=result.finish_reason not in CUT_SHORT,
        )

    def _route(self, role: str) -> tuple[str, ChatFacade]:
        if role in FAST_ROLES:
            return "fast", self._fast
        if role in SMART_ROLES:
            return "smart", self._smart
        if role in PLANNER_ROLES:
            return "planner", self._planner
        if role.startswith(RESEARCH_PREFIX):
            return "research", self._research
        raise GenerationError(f"no provider chain serves the {role!r} role")

    def _deadline(self, request: GenerationRequest, label: str) -> float:
        limit = request.deadline_s
        for cap in (self._timeout, self._deadlines.get(label)):
            if cap is not None:
                limit = min(limit, cap)
        return limit


def _generation_error(error: ProviderError) -> GenerationError:
    if error.kind == "bad_request":
        return GenerationError(str(error))
    message = str(error)
    if isinstance(error, AllProvidersFailed) and error.errors:
        message += f"; the last said: {error.errors[-1][1]}"  # why, in the adapter's own safe words (a spent account, say)
    return GenerationUnavailable(message)


def _messages(request: GenerationRequest) -> list[dict]:
    """The request as OpenAI chat messages: the instructions as the system prompt, then the user's turn.

    The user's turn is the request's data, fenced (`_fenced`). The schema is not written into the prompt here: a facade given a `schema` enforces it where it can, and asks
    for JSON in the prompt where it cannot (llm.py).
    """
    content: list[dict] = [{"type": "text", "text": _fenced(request.text)}]
    if request.image is not None:
        content.insert(0, {"type": "image_url", "image_url": {"url": image_url(request.image)}})
    system = [{"role": "system", "content": request.instructions}] if request.instructions else []
    return [*system, {"role": "user", "content": content}]


def _fenced(text: str) -> str:
    """The request's JSON between <data> tags, which its instructions call untrusted data (writer.UNTRUSTED).

    `</` becomes `<\\/`, which JSON reads as the same two characters, so nothing in the data can write the closing tag.
    """
    return "<data>\n" + text.replace("</", "<\\/") + "\n</data>"


def _usage(usage: Usage | None) -> TokenUsage:
    """A facade's usage as the neutral shape. `Usage.input_tokens` is already the uncached count (llm.py splits the
    cached ones out), so nothing is subtracted here."""
    return TokenUsage(
        getattr(usage, "input_tokens", 0) or 0,
        getattr(usage, "cached_input_tokens", 0) or 0,
        getattr(usage, "output_tokens", 0) or 0,
    )

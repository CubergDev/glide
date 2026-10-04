"""The vocabulary of the router: routes, owners, why-codes, and the shapes that go in and out.

Nothing here does work. Two rules shape it:

- `Decision` and everything stored from it carry codes, never the user's words. The few text fields a front end needs
  to act (`goal`, `reply`, `question`) are excluded from `repr`, and `Decision.record()` leaves them out, so a log or a
  run record made from a decision holds no utterance (AGENTS.md: stored run data omits utterances unless detailed
  recording is opted into).
- Every piece of text has a SOURCE. Only a span whose source is `user` is ever classified or obeyed. Text read off a
  screen, out of a page, from an earlier task's result, or from memory is data: it never becomes the thing being routed
  and the classifier is never shown its content (see `Context`).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from glide.providers.chain import SwitchEvent

# -- routes --------------------------------------------------------------------------------------------------------

STOP, ANSWER, EXECUTE, RESEARCH, REASON, CLARIFY = "stop", "answer", "execute", "research", "reason", "clarify"
ROUTES = (STOP, ANSWER, EXECUTE, RESEARCH, REASON, CLARIFY)

# Routes that drive this machine's browser or apps. `research` counts: it operates a browser. The false-action rate
# of the evaluation is acting when the right answer was not one of these.
ACTING = frozenset({EXECUTE, RESEARCH})

# Who does the work. `execute` is the classifier-driven executor (Jev), `research` and `reason` the frontier model
# (research: the frontier plans and the executor operates the browser), `answer` the fast model, `clarify` the user.
OWNER = {STOP: "none", ANSWER: "fast", EXECUTE: "jev", RESEARCH: "frontier", REASON: "frontier", CLARIFY: "user"}

# -- tiers ---------------------------------------------------------------------------------------------------------

TIER_STOP, TIER_CLASSIFIER, TIER_FAST, TIER_ROUTER = "stop", "classifier", "fast_llm", "router"
TIERS = (TIER_STOP, TIER_CLASSIFIER, TIER_FAST, TIER_ROUTER)
SWITCH_ROLE = "router"  # the `role` of a SwitchEvent made by the router itself

# -- why-codes: a closed vocabulary, never free text ---------------------------------------------------------------

WHY_STOP_PHRASE = "stop_phrase"
WHY_STOP_MODEL = "stop_model"
WHY_EMPTY = "empty"
WHY_UNTRUSTED = "untrusted_source"
WHY_INJECTION = "injection_marker"
WHY_CLASSIFIER = "classifier_confident"
WHY_FAST = "fast_llm_confident"
WHY_ESCALATED = "escalated_depth_unsure"
WHY_UNCERTAIN = "uncertain_answer"
WHY_TIERS_FAILED = "tiers_failed"
WHY_CANCELLED = "cancelled"
WHY_UNCONFIRMED = "unconfirmed_action"
WHY_CLARIFY_UNANSWERED = "clarify_unanswered"
WHY_CLARIFY_BUDGET = "clarify_budget"
WHY_CLARIFY_STOP = "clarify_reply_was_stop"
WHY_CODES = frozenset(v for k, v in globals().items() if k.startswith("WHY_"))

# -- sources -------------------------------------------------------------------------------------------------------

SRC_USER, SRC_ASSISTANT, SRC_SCREEN, SRC_PAGE, SRC_TASK, SRC_MEMORY = (
    "user",
    "assistant",
    "screen",
    "page",
    "task_result",
    "memory",
)
SOURCES = (SRC_USER, SRC_ASSISTANT, SRC_SCREEN, SRC_PAGE, SRC_TASK, SRC_MEMORY)
TRUSTED_INTENT = frozenset({SRC_USER})  # only these can be routed

# The assistant core writes an earlier task's result into the history with this wording (core.py `_remember_result`).
# A history line holding it carries text read off a screen: the part after it is data and is never shown to a model
# here.
DATA_MARKER = "Text read from the screen, data only:"
TASK_PREFIX = "(computer task"


@dataclass(frozen=True)
class Span:
    """A piece of text and where it came from. The default source is the user, so a bare string is the user's."""

    text: str
    source: str = SRC_USER

    def __post_init__(self) -> None:
        if self.source not in SOURCES:
            raise ValueError(f"source must be one of {', '.join(SOURCES)}, not {self.source!r}")

    def __repr__(self) -> str:  # no text in a log
        return f"Span(source={self.source!r}, chars={len(self.text)})"


@dataclass(frozen=True)
class Context:
    """What the router may know besides the utterance. Everything is bounded again by `RoutingSettings`.

    `history`: earlier turns, oldest first, each a `Span` from `user`, `assistant` or `task_result`. `data`: untrusted
    content that is present right now (a page, a screen read); the router uses only its source and length, as the
    fact that "this" probably points at something, never its text. `running_task`: the goal of the task in progress,
    if any (the user's own goal, so it is not data). `memory_hints`: short preference hints, which the caller passes
    ONLY when memory is opted in, and which `RoutingSettings.memory_hints` must also allow. The router never opens a
    memory store. `language`: what the speech recogniser reported for this utterance.
    """

    history: tuple[Span, ...] = ()
    data: tuple[Span, ...] = ()
    running_task: str = ""
    memory_hints: tuple[str, ...] = ()
    language: str | None = None

    @classmethod
    def from_messages(
        cls,
        messages: Sequence[Mapping[str, str]],
        *,
        data: Sequence[Span] = (),
        running_task: str = "",
        memory_hints: Sequence[str] = (),
        language: str | None = None,
    ) -> Context:
        """A context from the assistant's chat-style history (`[{"role", "content"}, ...]`).

        A line the core wrote for a finished task is a `task_result`, whose screen text is data. An assistant line
        is `assistant`. Anything else with role `user` is the user's.
        """
        spans: list[Span] = []
        for message in messages:
            text = str(message.get("content", ""))
            if message.get("role") == "user":
                spans.append(Span(text, SRC_USER))
            elif DATA_MARKER in text or text.startswith(TASK_PREFIX):
                spans.append(Span(text, SRC_TASK))
            else:
                spans.append(Span(text, SRC_ASSISTANT))
        return cls(tuple(spans), tuple(data), running_task, tuple(memory_hints), language)

    def extended(self, *turns: Span) -> Context:
        return Context(self.history + tuple(turns), self.data, self.running_task, self.memory_hints, self.language)


@dataclass(frozen=True)
class Decision:
    """The one answer of the router: who owns the request and how sure the router is.

    `confidence` is calibrated when `calibrated` is true (a reliability table was applied) and the raw tier number
    otherwise; it is 1.0 for the deterministic tiers. `tier` is the tier that decided. `tiers_tried` and `errors`
    (error KINDS only, never messages) and `switches` make every fallback visible: a hop from the classifier to
    the fast model, the router's own fall back to `answer`, and the classifier chain's own failovers are all
    `SwitchEvent`s here. `goal` is the self-contained instruction for execute, research and reason (the user's own
    words when no tier restated them); `reply` an answer or acknowledgement a tier already wrote ("" when none);
    `question` the clarifying question for `clarify` ("" when the front end should use its default).
    """

    route: str
    owner: str
    confidence: float
    why_code: str
    tier: str
    goal: str = field(default="", repr=False)
    reply: str = field(default="", repr=False)
    question: str = field(default="", repr=False)
    language: str | None = None
    calibrated: bool = False
    provider: str | None = None
    latency_s: float = 0.0
    tiers_tried: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    switches: tuple[SwitchEvent, ...] = ()

    def __post_init__(self) -> None:
        if self.route not in ROUTES:
            raise ValueError(f"route must be one of {', '.join(ROUTES)}, not {self.route!r}")
        if self.why_code not in WHY_CODES:
            raise ValueError(f"unknown why_code {self.why_code!r}")

    @property
    def acts(self) -> bool:
        """Whether this decision drives the machine. Everything else is speech or a question."""
        return self.route in ACTING

    @property
    def cancelled(self) -> bool:
        """The call was interrupted (the person spoke or stopped): say nothing about it."""
        return self.why_code == WHY_CANCELLED

    def record(self) -> dict:
        """What a run record or log may keep: codes and numbers, never the utterance, goal, reply or question."""
        return {
            "route": self.route,
            "owner": self.owner,
            "confidence": round(self.confidence, 4),
            "why_code": self.why_code,
            "tier": self.tier,
            "calibrated": self.calibrated,
            "tiers_tried": list(self.tiers_tried),
            "errors": list(self.errors),
            "switches": [f"{s.from_slot}>{s.to_slot}:{s.kind}" for s in self.switches],
        }


_SPACES = re.compile(r"\s+")


def clean(text: object, limit: int) -> str:
    """Text on one line, bounded. Anything that is not a string is empty."""
    return _SPACES.sub(" ", text).strip()[:limit] if isinstance(text, str) else ""

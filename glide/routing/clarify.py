"""Clarify needs a channel: the small interface a front end implements, and what happens when there is no answer.

```python
class Clarifier(Protocol):
    def ask(self, question: str) -> str | None: ...
```

`ask` shows or speaks the question and returns what the user said back, or None when it cannot be asked or the user
did not answer (no channel, a timeout, a stop, a new request that arrived instead). The front end owns which of its
inputs is the reply: a NEW request is never taken for the answer (the assistant's `answer_pending` rule), so it makes
`ask` return None and is routed on its own.

`resolve` is the whole loop around one request:

1. route the request;
2. while the decision is `clarify`: ask. With no clarifier, no answer, or the question budget spent, nothing is done: the
   result is a `clarify` decision whose `said` is one sentence naming what is needed. No action, no guess;
3. an answer that is itself a stop phrase is a `stop`. Any other answer is the USER's words (it came through the pending
   question): it is added to the history as an assistant question and a user reply, and the original request is routed
   again with that context.

The question is the tier's own (the fast model writes one) or, when no tier did, a fixed sentence in the user's language.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Protocol

from .decision import (
    CLARIFY,
    SRC_ASSISTANT,
    SRC_USER,
    STOP,
    TIER_ROUTER,
    WHY_CLARIFY_BUDGET,
    WHY_CLARIFY_STOP,
    WHY_CLARIFY_UNANSWERED,
    Context,
    Decision,
    Span,
    clean,
)
from .phrases import say
from .router import Router

MAX_REPLY_CHARS = 500


class Clarifier(Protocol):
    """What a front end implements so a question can reach the user and an answer come back."""

    def ask(self, question: str) -> str | None: ...


@dataclass(frozen=True)
class Resolution:
    """The end of a request's routing. `decision` is final: act on it only if it is not `clarify`.

    `said` is a sentence to tell the user when nothing was done for want of an answer ("" otherwise). `asked` counts
    the questions put. `unresolved` is true when the decision is still `clarify`: take no action.
    """

    decision: Decision
    asked: int = 0
    said: str = ""

    @property
    def unresolved(self) -> bool:
        return self.decision.route == CLARIFY


def resolve(
    router: Router,
    utterance: str | Span,
    context: Context | None = None,
    clarifier: Clarifier | None = None,
    *,
    language: str | None = None,
    on_question: Callable[[str], None] | None = None,
) -> Resolution:
    """Route `utterance`, asking the user as many questions as `router.settings.max_clarifications` allows."""
    ctx = context or Context()
    span = utterance if isinstance(utterance, Span) else Span(utterance)
    decision = router.route(span, ctx)
    asked, exchanges = 0, []
    while decision.route == CLARIFY:
        lang = decision.language or language or ctx.language
        question = decision.question or say("clarify_default", lang)
        if clarifier is None:
            return _unresolved(decision, WHY_CLARIFY_UNANSWERED, question, lang, asked)
        if asked >= router.settings.max_clarifications:
            return _unresolved(decision, WHY_CLARIFY_BUDGET, question, lang, asked)
        if on_question is not None:
            on_question(question)
        reply = clarifier.ask(question)
        asked += 1
        reply = clean(reply, MAX_REPLY_CHARS)
        if not reply:
            return _unresolved(decision, WHY_CLARIFY_UNANSWERED, question, lang, asked)
        if router.is_stop(reply):
            return Resolution(replace(decision, route=STOP, owner="none", why_code=WHY_CLARIFY_STOP, tier=TIER_ROUTER), asked)
        exchanges.append(f"{question}\nUser: {reply}")
        ctx = ctx.extended(Span(question, SRC_ASSISTANT), Span(reply, SRC_USER))
        decision = router.route(span, ctx)
    if exchanges and decision.goal:  # the executor's own format: the goal, then what was asked and answered
        decision = replace(decision, goal=decision.goal + "\n\nClarifications:\n" + "\n\n".join(exchanges))
    return Resolution(decision, asked)


def _unresolved(decision: Decision, why: str, question: str, language: str | None, asked: int) -> Resolution:
    final = replace(decision, why_code=why, tier=TIER_ROUTER)
    return Resolution(final, asked, say("clarify_needed", language, question=question))

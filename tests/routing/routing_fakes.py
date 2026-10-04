"""Scripted stand-ins for the two model tiers of the router. Nothing here reaches a network, a key or a machine.

`FakeClassifier` has the one method of a classifier slot, `system_one`, and answers with the SDK's own `ChoiceAnswer`
wrapped in the real `ClassifierReply`, so the router reads exactly what it reads in production. `FakeLLM` has the one
method of the fast chain, `chat`, and returns a `ChatResult`. Each records everything it was shown (deep-copied), so a test
can prove what a model was NOT told.

A script is a value, an exception to raise, or a callable that gets what the fake was asked and returns one of those.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Callable
from dataclasses import dataclass

from typesafe_sdk import ChoiceAnswer

from glide.providers.base import ChatResult, Usage
from glide.providers.classifier import ClassifierReply
from glide.providers.errors import ProviderError

ROUTES = ("answer", "execute", "research", "reason", "clarify", "stop")


@dataclass(frozen=True)
class Pick:
    """A classifier answer: the choice, its confidence, and the rest of the distribution (None: spread what is left)."""

    choice: str
    p: float
    probs: dict[str, float] | None = None
    runner_up: str | None = None  # where the rest of the mass goes when `probs` is None

    def distribution(self) -> dict[str, float]:
        if self.probs is not None:
            return dict(self.probs)
        rest = max(0.0, 1.0 - self.p)
        second = self.runner_up or next(r for r in ROUTES if r != self.choice)
        return {self.choice: self.p, second: rest}


@dataclass(frozen=True)
class Garbage:
    """A classifier reply that is not usable: `what` is one of the cases below."""

    what: str  # "no_answer", "unknown_choice", "nan_confidence", "percent", "string_confidence"


def failing(kind: str) -> ProviderError:
    return ProviderError(f"fake {kind}", kind=kind, provider="fake")


class FakeClassifier:
    name = "fake:classifier"
    model = "fake"

    def __init__(self, script) -> None:
        self.script = script
        self.calls: list[dict] = []

    def system_one(self, state, questions, *, model=None, **options):
        self.calls.append({"state": copy.deepcopy(state), "questions": dict(questions)})
        outcome = self.script(state) if callable(self.script) else self.script
        if isinstance(outcome, BaseException):
            raise outcome
        if isinstance(outcome, Garbage):
            return _garbage(outcome.what)
        answer = ChoiceAnswer(choice=outcome.choice, confidence=outcome.p, probabilities=outcome.distribution())
        return ClassifierReply({"route": answer}, Usage(), self.model)


def _garbage(what: str) -> ClassifierReply:
    if what == "no_answer":
        return ClassifierReply({}, Usage(), "fake")
    choice, confidence = "execute", 0.95
    if what == "unknown_choice":
        choice = "delete_everything"
    elif what == "nan_confidence":
        confidence = math.nan
    elif what == "percent":
        confidence = 95
    elif what == "string_confidence":
        confidence = "0.95"  # type: ignore[assignment]
    answer = ChoiceAnswer(choice=choice, confidence=confidence, probabilities={})  # a Struct does not validate itself
    return ClassifierReply({"route": answer}, Usage(), "fake")


def fast_json(route="answer", confidence="high", reply="", goal="", question="", language="en", **extra) -> str:
    """The fast model's reply, in the shape its schema demands."""
    body = {
        "route": route,
        "confidence": confidence,
        "reply": reply,
        "goal": goal,
        "question": question,
        "language": language,
    }
    body.update(extra)
    return json.dumps(body, ensure_ascii=False)


class FakeLLM:
    def __init__(self, script=None) -> None:
        self.script = script if script is not None else fast_json("answer", reply="Hello.")
        self.calls: list[dict] = []

    def chat(self, messages, *, hedge=None, **kw) -> ChatResult:
        self.calls.append({"messages": copy.deepcopy(list(messages)), **kw})
        outcome = self.script(messages) if callable(self.script) else self.script
        if isinstance(outcome, BaseException):
            raise outcome
        return ChatResult(outcome, Usage(), "fake:fast", "fake", 0.0)


def everything_shown(fake) -> str:
    """All the text a fake model was shown, as one string, for a test that looks for a canary in it."""
    if isinstance(fake, FakeClassifier):
        return json.dumps([c["state"] for c in fake.calls], ensure_ascii=False, default=str)
    return json.dumps([c["messages"] for c in fake.calls], ensure_ascii=False, default=str)


Script = Callable[..., object]

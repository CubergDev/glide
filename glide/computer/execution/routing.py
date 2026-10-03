"""Decide who owns a task before anything touches a browser: `execute`, `research`, `reason` or `clarify`.

`decide` is a pure function of the classifier, the writer and the goal: it keeps no state and touches no machine. It
is the engine's own scope decision. It is not the final router: a later phase merges it with
`glide.assistant.router` (stop fast-path, this classifier question, then the fast-LLM JSON call), so keep it
importable and free of engine state.

One cheap classifier question (Jev) decides; only an uncertain answer (`InvalidAction`: a missing, unknown or
low-confidence choice) goes to the ordinary writer. A classifier chain that is down raises its own `ProviderError`,
which is not a reason to guess: the run ends there and says so.
"""

from dataclasses import dataclass

from .. import diagnostics
from ..writer import route_task
from .contracts import InvalidAction
from .policy import choose

WORKFLOWS = ("execute", "research", "reason", "clarify")


@dataclass(frozen=True)
class Scope:
    workflow: str  # one of WORKFLOWS, or "plan" when the writer answered an uncertain classifier
    source: str  # who decided: "jev" (the classifier) or "writer"
    question: str = ""

    @property
    def owner(self) -> str:
        """Who does the work: the frontier model for an answer, the classifier-driven executor for browser steps."""
        return "frontier" if self.workflow in {"research", "reason"} else "jev"


def decide(client, writer, goal, *, minimum, clarifications=()):
    """The `Scope` of the whole request. `clarifications` are earlier questions and the user's replies."""
    context = {"goal": goal, **({"clarifications": list(clarifications)} if clarifications else {})}
    try:
        workflow, _ = choose(
            client,
            {"task_scope": True, **context},
            {
                "execute": "Perform browser or application operations, including searches; no composed answer is requested.",
                "research": "Gather external or on-screen information and return a factual answer, summary, comparison, review or recommendation; may also include browser actions.",
                "reason": "Reason, calculate, explain, draft or write code from the supplied requirements or text; no current external information or browser interaction is needed.",
                "clarify": "The user's intended outcome is unclear.",
            },
            minimum=minimum,
            instructions="Route the WHOLE requested deliverable, in any language. Complexity means reasoning or synthesis, "
            "not prompt length or the number of UI steps. Opening tabs, typing supplied text and submitting a search "
            "remain execute, even if the search topic mentions reviews or comparisons. If the user wants you to give "
            "an answer from the results, use research, preserving all requested browser work. Pure analysis or code "
            "generation is reason; entering already supplied code is execute. Do not drop an answer requirement.",
        )
        return Scope(workflow, "jev")
    except InvalidAction:
        workflow, question = route_task(writer, context)
        return Scope(workflow, "writer", question)


def record(scope):
    """The decision as run.json and the event log keep it: workflow, who decided and who owns the work."""
    item = {"workflow": scope.workflow, "source": scope.source, "owner": scope.owner}
    diagnostics.event("task_route", **item)
    return item

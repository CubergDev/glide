"""Who owns a task once it is a computer task: `execute`, `research`, `reason` or `clarify`.

The ONE router is `glide.routing` (docs/ROUTER.md): the assistant asks it first and hands the result to the engine as
`RunConfig.route`, and the engine then takes it as the scope (`Scope(route, "router")`) instead of asking again. What is
left here is only what the engine still needs: the scope question for a run nothing routed first (`glide-computer` run
directly, an empty `route`) or one routed `clarify`, which the engine asks about on its own channel (`ctx.ask`).

`decide` is a pure function of the classifier, the writer and the goal: it keeps no state and touches no machine. One
cheap classifier question (Jev) decides; only an uncertain answer (`InvalidAction`: a missing, unknown or low-confidence
choice) goes to the ordinary writer. A classifier chain that is down raises its own `ProviderError`, which is not a
reason to guess: the run ends there and says so.
"""

from dataclasses import dataclass

from .. import diagnostics
from ..writer import route_task
from .contracts import InvalidAction
from .policy import choose

WORKFLOWS = ("execute", "research", "reason", "clarify")
# What a router decision can hand the engine as final. "clarify" is not here: the router resolves it (asks, or does
# nothing) before any task starts, and the engine's own question loop stays the way a task asks mid-run.
ROUTED = frozenset({"execute", "research", "reason"})


@dataclass(frozen=True)
class Scope:
    workflow: str  # one of WORKFLOWS, or "plan" when the writer answered an uncertain classifier
    source: str  # who decided: "router" (glide/routing, passed in), "jev" (the classifier) or "writer"
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

"""Plans constrain Jev; only observed effects advance the durable task ledger.

`run_execution` takes the goal through four stages, each bounded:

1. Scope (routing.py): the request is `execute`, `research`, `reason` or `clarify`. A question to the user is asked
   at most `MAX_QUESTIONS` times in a run.
2. Plan: milestones (contracts.Milestone), each an effect to be observed and not an instruction to follow. A direct
   binding, a query intent or the planner produces them; research and reasoning hand the work to a supervisor.
3. Loop: observe, wait for readiness, let Jev bind one offered action to the milestone, re-observe, dispatch the
   action through `control.dispatch`, and count the milestone only when a fresh observation shows its effect. An
   action whose outcome is unknown is never replayed: the run stops, reads the screen once and says so.
4. Report: run.json (counts, timings, the outcome; goal, plan and history only when content recording is on, D3).

Nothing here talks to a vendor. The classifier is the one the caller's `classifier_factory` returns (a provider
chain, D7), model calls go through `writer.generate` (D1), and every input action passes `RunControl` (D2). Browser
and desktop adapters arrive through `make_backend`, and the research supervisor through `make_supervisor`; tests
replace those two names.
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
import uuid
from dataclasses import asdict, replace

from ...providers.config import ConfigError
from ...providers.errors import AllProvidersFailed, ProviderError
from .. import config, diagnostics
from ..control import CANCELLED, checkpoint, controlled, current_control, dispatch
from ..models import Abort, BrowserConnectionError, DesktopError
from ..runner import FAILURE_CHARS, MAX_QUESTIONS, WOULD_DO_CHARS, RunState, metered
from ..writer import Answer
from . import planning, policy, query, routing
from .contracts import InvalidAction, Observation, UnsupportedCapability, effect, primitive_effect, rebind, validate
from .progress import Ledger, observe, wait_effect, wait_ready

# Bounds a caller may set (RunConfig); outside them nothing is dispatched.
MAX_STEPS = 500
MAX_READINESS_S = 30
MAX_RECOVERIES = 2  # replans a run may use, whatever `handoffs` allows: a plan that failed twice will not work a third time
MEDIA_VERIFICATION_S = 10.0  # a started video may take this long to show a playing state (unless readiness is 0)
TAB_EFFECTS = {"url", "tab_created", "tab_active", "tab_closed"}  # these do not wait on the source document
NO_FRESH_VERIFICATION_NEEDED = {"scroll", "tab_created", "tab_closed"}  # a repeat is a new operation, not the same state
DEFAULT_QUESTION = "What outcome should this task reach?"
UNSUPPORTED_RECORDED = "A requested capability is not available; its name is kept only when content recording is on."
UNSUPPORTED_SEARCH = "I couldn't find an available search field on this page, so I didn't submit the search."

# What a user is told when a provider failed, by the error's kind and status and never by its text (a provider's
# message can carry what the server said). One sentence per cause the user can act on.
PROVIDER_SENTENCES = {
    "no_credit": "A model provider reports no credit or quota left. Add credit to that account or use another provider.",
    "auth": "A model provider refused its API key. Check the key named for that provider in glide.toml.",
    "rate_limit": "A model provider is limiting requests. Try again in a little while.",
    "timeout": "A model provider did not answer in time.",
    "transport": "A model provider could not be reached.",
    "stream": "A model provider's connection dropped before it finished.",
    "server": "A model provider had an internal error.",
    "unsupported": "A configured model or feature is not available at its provider. Check glide.toml.",
    "content": "A model provider answered with something Glide could not use.",
    "bad_request": "A model request was refused as malformed. This is a fault in Glide, not an outage.",
}
GENERIC_PROVIDER_SENTENCE = "A model provider failed."
NO_REPLAY = "No action was repeated."


def make_backend(browser: str):
    """The browser or desktop adapter for this run (contracts.Backend); `native.make_backend` picks the provider."""
    from . import native

    return native.make_backend(browser)


def make_supervisor(goal: str, route: str):
    """The research supervisor for the `research` and `reason` routes (research.Supervisor)."""
    from . import research

    return research.Supervisor(goal, route, config.research_calls())


def provider_error(error: BaseException) -> ProviderError | None:
    """The `ProviderError` behind `error`: itself, or the first one on its cause chain.

    The writer boundary wraps a chain's failure (ChainWriter, then writer._generate), so a model failure reaches the
    engine as a WriterUnavailable whose `__cause__` is the ProviderError. The classifier chain raises it directly.
    """
    for _ in range(8):
        if isinstance(error, ProviderError) or error is None:
            return error
        error = error.__cause__
    return None


def provider_sentence(error: ProviderError) -> str:
    """A fixed sentence for what happened to a provider, from its kind and status; never any of its words."""
    if isinstance(error, AllProvidersFailed):
        last = error.errors[-1][1] if error.errors else None
        return f"Every provider for this job failed. {provider_sentence(last) if last else ''}".strip()
    spent = error.status == 402 or (error.kind == "auth" and error.status in (400, 429))  # providers.errors.from_status
    return PROVIDER_SENTENCES.get("no_credit" if spent else error.kind, GENERIC_PROVIDER_SENTENCE)


def no_browser() -> Observation:
    """What the engine sees when there is no browser: nothing, and nothing it may do."""
    return Observation("", "", capabilities=set(), browser_front=False)


class Phases:
    """Times and traces each stage of a run. Calling it runs one stage; `stage` is the one running or the one that failed.

    Stage names are the vocabulary of the report (`phases` in run.json) and of the events; the research supervisor is
    handed this callable to run its own stages the same way.
    """

    def __init__(self):
        self.stage = "setup"
        self.totals: dict[str, dict] = {}

    def __call__(self, name, function, *args, **kwargs):
        self.stage = name
        began = time.perf_counter()
        diagnostics.event("phase_started", stage=name)
        try:
            result = function(*args, **kwargs)
        except BaseException as error:
            diagnostics.event("phase_failed", stage=name, error_type=type(error).__name__)
            raise
        else:
            self._trace(name, result)
            return result
        finally:
            seconds = time.perf_counter() - began
            metric = self.totals.setdefault(name, {"calls": 0, "seconds": 0.0})
            metric["calls"] += 1
            metric["seconds"] += seconds
            diagnostics.event("phase_finished", stage=name, elapsed_s=seconds)

    @staticmethod
    def _trace(name, result):
        """Event for an observation (the first of a verification's `(after, verified)`), with page content only if recorded."""
        view = result[0] if isinstance(result, tuple) and result else result
        if not isinstance(view, Observation):
            return
        details = {
            "app": view.app,
            "owner": view.owner,
            "url": view.url,
            "active_tab": view.active_tab,
            "capabilities": sorted(view.capabilities),
            "ready": view.ready,
            "element_count": len(view.elements),
            "form_count": len(view.forms),
            "focus": view.focus,
        }
        if diagnostics.record_content():
            details["page_content"] = view.packet()
        diagnostics.event("observation", stage=name, observation=details)
        if name == "verification":
            diagnostics.event("effect_checked", verified=bool(result[1]), stage=name)

    def summary(self):
        return {name: {**values, "seconds": round(values["seconds"], 6)} for name, values in self.totals.items()}


class Execution:
    """One structured run. `run()` reports every way out in `state` and writes run.json; it raises only `ConfigError`."""

    def __init__(self, cfg, ctx_factory, classifier_factory):
        self.cfg, self.ctx_factory, self.classifier_factory = cfg, ctx_factory, classifier_factory
        self.control = current_control()
        self.state = RunState()
        self.phases = Phases()
        self.started = time.perf_counter()
        self.ledger = self.backend = self.supervisor = None
        # The plan and what has been used of the budgets.
        self.steps, self.revision, self.recoveries, self.questions = [], 0, 0, 0
        self.clarifications, self.routes, self.bindings = [], [], {}
        # The operation in progress, kept for the read-only reconciliation after a stop.
        self.operation = self.step = self.action = self.before = None
        self.receipt, self.verification_step, self.op_started = "", None, 0.0
        self.first_action = None
        # The loop's own bookkeeping.
        self.action_source = self.cached = None
        self.failures = self.stale = 0
        self.transitions: set = set()
        self.selection_error = ""
        # How the run failed, for the report.
        self.failure_stage = self.error_type = self.failure_code = ""
        self.connection_error = None

    # -- entry ------------------------------------------------------------------------------------------------

    def run(self) -> RunState:
        cfg, state = self.cfg, self.state
        if (
            not math.isfinite(cfg.readiness_timeout)
            or not 0 <= cfg.readiness_timeout <= MAX_READINESS_S
            or type(cfg.steps) is not int
            or not 1 <= cfg.steps <= MAX_STEPS
            or cfg.handoffs < 0
            or not 0 <= cfg.min_confidence <= 1
        ):
            return self._blocked_early("Invalid execution budget or readiness deadline; no action was issued.")
        try:
            cfg.out.mkdir(parents=True, exist_ok=True)
            self.ledger = Ledger(cfg.out / "progress.sqlite3", self.control.task_id)
        except (OSError, sqlite3.Error):
            return self._blocked_early("The task journal is unavailable; no action was issued.")
        try:
            self._execute()
        except (Abort, KeyboardInterrupt) as error:
            self._stopped(error)
        except UnsupportedCapability as error:
            self._unsupported(error)
        except ConfigError:
            raise  # the caller words a missing configuration, the same as for the legacy loop
        except Exception as error:
            self._failed(error)
        finally:
            self._report()
        return state

    def _blocked_early(self, failure) -> RunState:
        self.state.outcome, self.state.failure = "blocked", failure
        return self.state

    # -- the run ----------------------------------------------------------------------------------------------

    def _execute(self):
        cfg, state, phases = self.cfg, self.state, self.phases
        if self.ledger.unresolved():
            raise InvalidAction("An unresolved operation exists for this task; review its result before restarting")
        with self.classifier_factory() as classifier:
            ctx = metered(self.ctx_factory(classifier, state.history), state.calls)
            scope = self._scope(ctx)
            goal = cfg.goal + ("\n\nClarifications:\n" + "\n\n".join(self.clarifications) if self.clarifications else "")
            # Analysis and authoring need no browser, CDP or desktop capture.
            observed = no_browser()
            if scope.workflow != "reason":
                self.backend = make_backend(cfg.execution_browser or ctx.browser)
                if hasattr(self.backend, "configure_permissions"):  # optional: a session that asks before it writes
                    self.backend.configure_permissions(cfg.act)
                observed = phases("initial_observation", self.backend.inspect)
            self.action_source = observed
            checkpoint()
            self._plan(ctx, scope, goal, observed)
            self._loop(ctx, goal)

    def _ask(self, ctx, question) -> str:
        """Put one question to the user and keep the exchange. A reply that arrives after a stop is dropped, unrecorded."""
        if not ctx.ask or self.questions >= MAX_QUESTIONS:
            raise InvalidAction(question)
        self.questions += 1
        self.control.event("question", question, question_id=str(uuid.uuid4()))
        reply = ctx.ask(question).strip()
        checkpoint()
        if not reply:
            raise InvalidAction("Clarification was declined")
        self.clarifications.append(question + "\nUser: " + reply)
        return reply

    def _scope(self, ctx):
        cfg = self.cfg
        scope = self.phases("task_scope", routing.decide, ctx.typesafe, ctx.writer, cfg.goal, minimum=cfg.min_confidence)
        self.routes.append(routing.record(scope))
        while scope.workflow == "clarify":
            self._ask(ctx, scope.question or DEFAULT_QUESTION)
            scope = self.phases(
                "task_scope",
                routing.decide,
                ctx.typesafe,
                ctx.writer,
                cfg.goal,
                minimum=cfg.min_confidence,
                clarifications=self.clarifications,
            )
            self.routes.append(routing.record(scope))
        return scope

    def _plan(self, ctx, scope, goal, observed):
        """Turn the scope into registered milestones (or a supervisor, for research and reasoning)."""
        cfg, phases = self.cfg, self.phases
        route, direct = scope.workflow, None
        if route == "execute":
            try:
                route, direct = phases("routing", policy.route, ctx.typesafe, goal, observed, minimum=cfg.min_confidence)
                source = "jev"
            except InvalidAction:
                # Ownership was already decided. An action Jev could not bind needs the ordinary planner, not another
                # decision about frontier reasoning.
                route, source = "plan", "jev_uncertain"
            self.routes.append(routing.record(routing.Scope("direct" if direct else route, source, scope.question)))
        quick, question = None, scope.question
        if route == "query":
            if "query_form" not in observed.capabilities:
                raise UnsupportedCapability(["query_form (connect the approved browser in pet settings)"], observed.capabilities)
            quick, question = phases("query_extraction", query.intent, ctx.writer, goal, observed, config.search_url())
        if route in {"research", "reason"}:
            self.supervisor = make_supervisor(cfg.goal, route)
            self.supervisor.replies = list(self.clarifications)
            self.supervisor.questions = self.questions
        elif quick and not question:
            self.steps = quick
        elif direct:
            self.step, self.action = direct
            self.steps = [self.step]
        else:
            question = question or (DEFAULT_QUESTION if route == "clarify" else "")
            reply = ""
            while True:
                if not question:
                    self.steps, question = phases("planning", planning.plan, ctx.writer, goal, observed, reply=reply)
                if not question:
                    break
                reply = self._ask(ctx, question)
                observed = phases("observation", self.backend.inspect)
                question = ""
        self._register()

    def _register(self):
        self.ledger.register(self.steps, self.revision)
        diagnostics.event(
            "plan_registered", revision=self.revision, step_count=len(self.steps), plan=[asdict(s) for s in self.steps]
        )

    def _loop(self, ctx, goal):
        cfg, state, ledger, supervisor = self.cfg, self.state, self.ledger, self.supervisor
        for _ in range(cfg.steps):
            checkpoint()
            self.step = next((s for s in self.steps if ledger.count(s.id) < s.quantity), None)
            if self.step is None:
                if not supervisor:
                    break
                if self._supervise(ctx):
                    return
                continue
            if self._advance(ctx, goal):
                return
        else:
            state.outcome = "step limit"
            state.failure = "The execution budget ended before all requested effects were verified."
        # Completing the final effect on the last allowed step is still completion.
        checkpoint()
        if not supervisor and self.steps and all(ledger.count(s.id) >= s.quantity for s in self.steps):
            state.outcome, state.failure = "done", ""
            queries = [s.effect == "query_submitted" for s in self.steps]
            only_queries = all(s.effect in {"url", "query_submitted"} for s in self.steps) and any(queries)
            state.answer = Answer(
                "Submitted your query and verified the loaded search destination."
                if only_queries
                else "Done. Your task is complete.",
                True,
            )

    def _supervise(self, ctx) -> bool:
        """All planned milestones are verified: let the supervisor read, ask, answer or plan the next browser batch."""
        state, supervisor = self.state, self.supervisor
        observed = self.phases("observation", self.backend.inspect) if self.backend is not None else no_browser()
        batch = supervisor.advance(ctx, self.backend, observed, self.steps, self.ledger, self.phases, self.cfg.readiness_timeout)
        if supervisor.answer:
            state.outcome, state.answer = "done", supervisor.answer
            return True
        if batch:
            self.steps.extend(batch)
            self.revision += 1
            self._register()
        self.action = self.cached = None
        self.failures = 0
        return False

    # -- one pass of the loop -----------------------------------------------------------------------------------

    def _advance(self, ctx, goal) -> bool:
        """Work on the first unverified milestone once. True when the run is over (a dry run's proposal)."""
        cfg, state, phases, step = self.cfg, self.state, self.phases, self.step
        observed = phases("observation", observe, self.backend, step, self.action)
        if step.effect not in TAB_EFFECTS:  # tab identity and URL operations do not depend on the source document loading
            condition = (lambda view, text=step.value: query.available(view, text)) if step.effect == "query_submitted" else None
            observed = phases(
                "readiness",
                wait_ready,
                self.backend,
                observed,
                cfg.readiness_timeout,
                step=step,
                action=self.action,
                condition=condition,
            )
        checkpoint()
        # The state as it is may already satisfy a value or URL milestone: counted without a write.
        self.verification_step = self.bindings.get(step.id, step)
        existing = effect(self.verification_step, None, observed, observed)
        if existing and step.effect not in NO_FRESH_VERIFICATION_NEEDED:
            self.ledger.observe_effect(step, existing)
            self.action = None
            return False
        if self.action is not None:
            try:
                self.action = rebind(self.action, self.action_source, observed)
            except InvalidAction:
                self.action = None
        if self.action is None:
            self._choose(ctx, goal, observed)
        self.action_source = observed
        if self.action is None or self.failures >= 2:
            self._recover(ctx, goal)
            return False
        # A fresh preflight after the model call: a stale observation is rejected before any side effect.
        before = self.before = phases("preflight", observe, self.backend, step, self.action)
        checkpoint()
        try:
            self.action = rebind(self.action, self.action_source, before)
        except InvalidAction:
            self.stale += 1
            self.action = self.cached = None
            if self.stale >= 3:
                raise InvalidAction("Targets kept changing before dispatch; no further action was issued") from None
            return False
        self.stale = 0
        action = self.action
        if self.supervisor:
            self.supervisor.validate_action(action, before)
        diagnostics.event("action_selected", stage="preflight", step_id=step.id, kind=action.kind, action_params=asdict(action))
        if diagnostics.record_content():
            target = before.elements.get(action.target)
            diagnostics.event(
                "selected_target",
                step_id=step.id,
                observation_id=before.identity,
                selected_content={
                    "milestone": asdict(step),
                    "element": target.public() if target else None,
                    "tab_url": before.tabs.get(action.target),
                },
            )
        if not cfg.act:
            state.outcome = "dry run"
            state.would_do = self._would_do(action, before)
            return True
        self._dispatch(before)
        return False

    def _choose(self, ctx, goal, observed):
        """Let Jev bind one offered action to the milestone, or reuse the last one for a stable repeat."""
        step = self.step
        if self.cached and self.cached[0] == step.id:
            old_action, old_owner = self.cached[1:]
            candidate = replace(old_action, observation=observed.identity)
            # Amortize a Jev-selected repeat only for stable, fully specified primitives.
            if old_action.kind == "tab_create" or observed.owner == old_owner:
                try:
                    validate(candidate, observed)
                    if candidate in policy.candidates(step, observed):
                        self.action = candidate
                except InvalidAction:
                    pass
        if self.action is None:
            try:
                self.action = self.phases(
                    "selection",
                    policy.select,
                    ctx.typesafe,
                    goal,
                    step,
                    observed,
                    self.ledger.summary(),
                    minimum=self.cfg.min_confidence,
                )
            except UnsupportedCapability:
                raise
            except InvalidAction as error:
                self.action, self.selection_error = None, str(error)

    def _recover(self, ctx, goal):
        """The milestone cannot be reached as planned: ask the planner for a corrected plan, within the replan budget."""
        cfg, phases, supervisor = self.cfg, self.phases, self.supervisor
        if self.recoveries >= min(cfg.handoffs, MAX_RECOVERIES):
            raise InvalidAction(
                "Recovery budget exhausted; " + (self.selection_error or "the remaining effects could not be verified")
            )
        self.recoveries += 1
        reason = (
            "Selected operations produced no new verified effect"
            if self.failures
            else self.selection_error or "No offered action could reach the milestone"
        )
        checkpoint()
        original = [s for s in self.steps if s.id in supervisor.batch_ids] if supervisor else self.steps
        prior = [s for s in self.steps if s.id not in supervisor.batch_ids] if supervisor else []
        planning_goal = supervisor.browser_goal if supervisor else goal
        observed = phases("observation", self.backend.inspect)  # recovery needs the full surrounding UI
        context = supervisor.planning_context(observed) if supervisor else None
        steps, question = phases(
            "planning",
            planning.plan,
            ctx.writer,
            planning_goal,
            observed,
            original,
            self.ledger.summary(),
            reason,
            context=context,
        )
        if question:
            reply = supervisor.ask(ctx, question) if supervisor else self._ask(ctx, question)
            steps, question = phases(
                "planning",
                planning.plan,
                ctx.writer,
                planning_goal,
                observed,
                original,
                self.ledger.summary(),
                reason,
                reply,
                context=context,
            )
            if question:
                raise InvalidAction(question)
        if supervisor:
            supervisor.validate_plan(steps, observed)
            old_ids = {s.id for s in original}
            steps = [s if s.id in old_ids else replace(s, id=f"research{supervisor.batch}_{s.id}") for s in steps]
            supervisor.batch_ids = {s.id for s in steps}
            steps = prior + steps
        self.steps = steps
        self.revision += 1
        self._register()
        self.action = self.cached = None
        self.failures = 0
        self.selection_error = ""
        self.transitions.clear()

    def _dispatch(self, before):
        """Send the action and decide from fresh observations what it did. Never repeats an action of unknown outcome."""
        cfg, state, phases, step, action, ledger = self.cfg, self.state, self.phases, self.step, self.action, self.ledger
        is_query = step.effect == "query_submitted" and action.kind != "inspect"
        self.verification_step = query.bind(step, action, before) if is_query else self.bindings.get(step.id, step)
        if is_query:
            self.bindings[step.id] = self.verification_step
        self.operation = ledger.begin(step, action)
        self.receipt = ""
        self.op_started = time.perf_counter()
        if self.first_action is None and action.kind != "inspect":
            self.first_action = self.op_started - self.started
        if action.kind == "inspect":
            self.receipt = phases("dispatch", self.backend.execute, action, before)
        else:
            self.receipt = phases("dispatch", dispatch, self.backend.execute, action, before)
        media = step.effect == "media_playing" and action.kind in {"click", "key"}
        timeout = MEDIA_VERIFICATION_S if media and cfg.readiness_timeout > 0 else cfg.readiness_timeout
        if media:
            diagnostics.event("media_verification_pending", step_id=step.id, timeout_s=timeout)
        after, verified = phases(
            "media_verification" if media else "verification",
            wait_effect,
            self.backend,
            self.verification_step,
            action,
            before,
            self.receipt,
            timeout=timeout,
        )
        elapsed = time.perf_counter() - self.op_started
        intermediate = primitive_effect(action, before, after, self.receipt)
        if media and not verified:
            diagnostics.event(
                "media_verification_timed_out",
                step_id=step.id,
                timeout_s=timeout,
                media_state=[{k: v for k, v in asdict(m).items() if k != "label"} for m in after.media.values()],
            )
            raise InvalidAction(
                "Playback could not be verified. The selected page may be open; review the player before retrying."
            )
        if step.effect == "query_submitted" and action.kind == "key" and not verified:
            raise InvalidAction("Search submission was not verified; review the result before retrying")
        if not verified and not intermediate and action.kind in {"click", "type", "key", "tab_create", "tab_close"}:
            raise InvalidAction("Write outcome was not verified; review the effect before retrying")
        previous = ledger.count(step.id)
        ledger.finish(self.operation, step, verified, elapsed, intermediate)
        self.operation = None
        self.control.in_flight = False
        advanced = ledger.count(step.id) > previous
        transition = (step.id, action.identity, intermediate)
        novel = bool(intermediate) and transition not in self.transitions
        self.transitions.add(transition)
        self.failures = 0 if advanced or novel else self.failures + 1
        if advanced:
            self.transitions.clear()
        state.history.append(action.kind)
        state.timings.append({"act_and_verify": elapsed})
        self.control.event(
            "action_checked",
            "Progress verified."
            if advanced
            else "Intermediate operation verified."
            if novel
            else "The intended effect was not observed.",
        )
        self.cached = (step.id, action, after.owner) if advanced and action.kind in {"tab_create", "scroll"} else None
        self.action = None

    @staticmethod
    def _would_do(action, observed) -> str:
        """A dry run's proposal in words. It names page content, so it stays in memory (D3)."""
        target = observed.elements.get(action.target) or observed.containers.get(action.target)
        label = target.label if target else observed.tabs.get(action.target, "")
        what = f"{action.kind} {action.value[:WOULD_DO_CHARS]!r}" if action.value else action.kind
        return what + (f" {'in' if action.kind == 'type' else 'on'} {label[:WOULD_DO_CHARS]!r}" if label else "")

    # -- every way out ------------------------------------------------------------------------------------------

    def _stopped(self, error):
        """A stop: the user's, the mouse corner's or Ctrl-C. Read once, read-only, to say what the last write did."""
        state, control = self.state, self.control
        self.failure_stage, self.error_type, self.failure_code = self.phases.stage, type(error).__name__, "cancelled"
        state.outcome = "aborted"
        state.cancel_reason = str(error) if str(error) in {CANCELLED, "mouse in top-left corner"} else "interrupted"
        state.uncertain = control.in_flight
        if not (self.operation and self.backend and self.step and self.action and self.before):
            return
        try:
            with controlled(None):  # cancellation is off for this one observation only
                after = self.phases("reconciliation", observe, self.backend, self.step, self.action)
            verified = effect(self.verification_step or self.step, self.action, self.before, after, self.receipt)
            intermediate = primitive_effect(self.action, self.before, after, self.receipt)
            if verified or intermediate:
                self.ledger.finish(self.operation, self.step, verified, time.perf_counter() - self.op_started, intermediate)
                state.uncertain = control.in_flight = False
            state.readback = (
                "effect verified"
                if verified
                else "intermediate operation verified; task incomplete"
                if intermediate
                else "observed; completion unknown"
            )
        except Exception:
            state.readback = "unavailable; completion unknown"

    def _unsupported(self, error):
        diagnostics.exception(error, stage=self.phases.stage)
        self.failure_stage, self.error_type, self.failure_code = self.phases.stage, type(error).__name__, "unsupported_capability"
        self.state.outcome = "unsupported"
        self.state.failure = UNSUPPORTED_SEARCH if error.missing == (query.FORM_CAPABILITY,) else str(error)
        self.state.unsupported_capabilities = list(error.missing)

    def _failed(self, error):
        state = self.state
        diagnostics.exception(error, stage=self.phases.stage)
        self.failure_stage, self.error_type = self.phases.stage, type(error).__name__
        provider = provider_error(error)
        if isinstance(error, BrowserConnectionError):
            self.failure_code, self.connection_error = "browser_unavailable", error.details
        elif provider:
            self.failure_code = "model_unavailable"
        elif isinstance(error, InvalidAction):
            self.failure_code = "invalid_query_contract" if "Query verification" in str(error) else "invalid_action"
        else:
            self.failure_code = "execution_error"
        state.outcome = "provider failure" if provider else "blocked"
        if provider:
            state.failure = f"{provider_sentence(provider)} {NO_REPLAY}"
        elif isinstance(error, InvalidAction | DesktopError):
            state.failure = str(error)
        else:
            state.failure = f"Execution stopped ({type(error).__name__}); no automatic replay."
        state.uncertain = self.control.in_flight
        if state.uncertain:
            state.readback = "completion unknown; review before a fresh task"

    # -- the report -----------------------------------------------------------------------------------------------

    def _report(self):
        """Close the journal and the backend and write run.json, whichever way the run ended."""
        cfg, state, ledger, supervisor = self.cfg, self.state, self.ledger, self.supervisor
        try:
            try:
                state.progress = ledger.summary()
            except sqlite3.Error:
                state.failure += " Task progress could not be read; review before retrying."
            state.plan_revisions, state.first_action_s = self.revision, self.first_action
            if supervisor:
                answered = supervisor.answer is not None
                state.progress.append(
                    {"id": "research_answer", "requested": 1, "verified": int(answered), "remaining": int(not answered)}
                )
            if state.failure and state.progress:
                verified, remaining = sum(p["verified"] for p in state.progress), sum(p["remaining"] for p in state.progress)
                state.failure += f" Verified {verified} effect(s); {remaining} remain."
            summary = self._summary()
            state.summary = summary
            try:
                diagnostics.write_private(cfg.out / "run.json", json.dumps(diagnostics.redact(summary), indent=2))
            except OSError:
                state.failure += " Diagnostic output could not be saved."
        finally:
            ledger.close()
            if self.backend:
                self.backend.close()

    def _summary(self) -> dict:
        cfg, state, supervisor = self.cfg, self.state, self.supervisor
        connection, failure = self.connection_error, state.failure
        if not cfg.record_content:
            # An address, and a capability the planner named (it can echo the request), are content (D3).
            connection = {k: v for k, v in connection.items() if k != "endpoint"} if connection else connection
            failure = UNSUPPORTED_RECORDED if self.failure_code == "unsupported_capability" else failure
        summary = {
            "engine": "structured",
            "transport": getattr(self.backend, "transport", "scripted") if self.backend is not None else "none",
            "routing": self.routes,
            "task_id": self.control.task_id,
            "outcome": state.outcome,
            "seconds": time.perf_counter() - self.started,
            "first_action_s": self.first_action,
            "steps_taken": len(state.history),
            "progress": state.progress,
            "plan_revisions": self.revision,
            "recoveries": self.recoveries,
            "calls": state.calls.summary(),
            "usage": state.calls.tokens(),
            "uncertain": state.uncertain,
            "unsupported_capability_count": len(state.unsupported_capabilities),
            "cancel_reason": state.cancel_reason,
            "interrupted_readback": state.readback,
            "failure_stage": self.failure_stage,
            "error_type": self.error_type,
            "failure_code": self.failure_code,
            "connection_error": connection,
            "phases": self.phases.summary(),
            "goal_achieved": bool(state.answer and state.answer.achieved),
            "failure": diagnostics.scrub_text(failure, limit=FAILURE_CHARS) if failure else None,
            **({"research": supervisor.summary()} if supervisor else {}),
        }
        if cfg.record_content:
            summary.update(
                goal=cfg.goal,
                plan=[asdict(s) for s in self.steps],
                history=state.history,
                unsupported_capabilities=state.unsupported_capabilities,
            )
            if supervisor:
                summary.update(evidence=supervisor.sources, answer_content=state.answer.text if state.answer else None)
        return summary


def run_execution(cfg, ctx_factory, classifier_factory) -> RunState:
    """Run `cfg.goal` with the structured engine. `classifier_factory()` is the classifier as a context manager."""
    return Execution(cfg, ctx_factory, classifier_factory).run()

"""The one router: `route(utterance, context) -> Decision`.

Tiers, cheapest first, and the first one that is sure decides:

1. **Not the user, empty, or a stop phrase.** Text whose source is not the user is never routed (it is data): the answer
   is `answer`. An exact stop phrase (`glide.routing.stop`) is `stop`, with no model and no network.
2. **The classifier chain** (`GlideConfig.classifier()`): one Choice question over answer, execute, research, reason,
   clarify and stop, with the executor's own criteria. Its confidence is calibrated, and compared to a threshold.
3. **The fast-LLM JSON call** (`GlideConfig.llm("fast")`): asked only when the classifier failed or was not sure.

Acceptance, per route, after calibration: a non-acting route needs `min_confidence`; an acting route (execute,
research) needs the higher `act_min_confidence` and, from the classifier, a margin over the runner-up; a stop the model
heard needs `stop_min_confidence`.

When no tier is sure, this is the whole fallback table, top row first:

| situation                                                        | route    | why_code                |
|------------------------------------------------------------------|----------|-------------------------|
| the call was cancelled (the person spoke or stopped)             | answer   | cancelled (say nothing) |
| any tier is sure of an acting route, and the utterance carries an override attempt | answer | injection_marker |
| a tier is sure                                                   | its route| classifier_confident / fast_llm_confident / stop_model |
| nobody is sure, but a tier puts at least `min_confidence` of its mass on answer or reason (sure it is not an action, unsure how deep) and `escalate_to_reason` | reason | escalated_depth_unsure |
| nobody is sure and doubt includes whether to act                 | answer   | uncertain_answer        |
| every tier failed, was missing or unparseable                    | answer   | tiers_failed            |
| `confirm_acting` is on and the other tier does not read an acting route the same way | answer | unconfirmed_action |

Doubt about acting never acts, and doubt between a quick answer and the frontier model goes to the frontier model.
`answer` is also what "I could not decide" has always meant here: it cannot touch the machine.

Every hop is visible: a tier that fails or is unsure adds an error kind (or "uncertain") to `Decision.errors` and a
`SwitchEvent` (role "router") to `Decision.switches` and to `on_event`; the classifier chain's own failovers are added to
`Decision.switches` too. `Decision.record()` is all a run record keeps: codes, never the request.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from contextlib import ExitStack
from dataclasses import dataclass, field, replace

from glide.computer.control import current_control, linked, spawn
from glide.computer.models import Abort
from glide.providers.chain import SwitchEvent
from glide.providers.errors import CANCELLED, ProviderError

from .calibration import Calibration
from .decision import (
    ACTING,
    ANSWER,
    OWNER,
    REASON,
    STOP,
    TIER_CLASSIFIER,
    TIER_FAST,
    TIER_ROUTER,
    TIER_STOP,
    TRUSTED_INTENT,
    WHY_CANCELLED,
    WHY_CLASSIFIER,
    WHY_EMPTY,
    WHY_ESCALATED,
    WHY_FAST,
    WHY_INJECTION,
    WHY_STOP_MODEL,
    WHY_STOP_PHRASE,
    WHY_TIERS_FAILED,
    WHY_UNCERTAIN,
    WHY_UNCONFIRMED,
    WHY_UNTRUSTED,
    Context,
    Decision,
    Span,
    clean,
)
from .features import Features, features
from .settings import RoutingSettings
from .stop import is_stop
from .stop import stop_phrases as stop_phrases_of
from .tiers import TierFailure, TierResult, ask_classifier, ask_fast, view

UNCERTAIN = "uncertain"  # the `kind` of a SwitchEvent for a tier that answered but was not sure
FAILED, BELOW = "tier_failed", "below_threshold"  # the `reason` of those events: fixed codes, never text
ANSWER_SLOT = "answer"


class _Cancelled(Exception):
    """The person cancelled while a tier was being asked."""


@dataclass
class _Trace:
    tried: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    switches: list[SwitchEvent] = field(default_factory=list)


class Router:
    """Decides who owns a request. Holds no conversation state: everything it needs comes in as `Context`.

    `classifier` is a `ChainedClassifier` (or anything with `system_one`) and `fast_llm` an `LLM` (anything with `chat`);
    either may be None, and the router then works with what it has. `stop_phrases` are the configured extras. `on_event`
    receives the router's own `SwitchEvent`s as they happen. `extra_markers` are further override phrases.
    """

    def __init__(
        self,
        classifier=None,
        fast_llm=None,
        *,
        settings: RoutingSettings | None = None,
        calibration: Calibration | None = None,
        stop_phrases: Iterable[str] = (),
        on_event: Callable[[SwitchEvent], None] | None = None,
        extra_markers: Iterable[str] = (),
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.classifier, self.fast_llm = classifier, fast_llm
        self.settings = settings or RoutingSettings()
        self.calibration = calibration or Calibration.identity()
        self._stops = stop_phrases_of(stop_phrases)
        self._on_event = on_event
        self._markers = tuple(extra_markers)
        self._clock = clock

    # -- the entry point --------------------------------------------------------------------------------------------

    def route(self, utterance: str | Span, context: Context | None = None) -> Decision:
        started = self._clock()
        ctx = context or Context()
        span = utterance if isinstance(utterance, Span) else Span(utterance)
        text = clean(span.text, self.settings.utterance_chars)
        trace = _Trace()

        def done(decision: Decision) -> Decision:
            return replace(
                decision,
                language=decision.language or ctx.language,
                latency_s=self._clock() - started,
                tiers_tried=tuple(trace.tried),
                errors=tuple(trace.errors),
                switches=tuple(trace.switches),
            )

        if not text:
            return done(self._plain(ANSWER, WHY_EMPTY, TIER_ROUTER, 1.0))
        if span.source not in TRUSTED_INTENT:  # what a screen, a page or a task said is never what the user said
            return done(self._plain(ANSWER, WHY_UNTRUSTED, TIER_ROUTER, 1.0))
        if is_stop(text, self._stops):
            return done(self._plain(STOP, WHY_STOP_PHRASE, TIER_STOP, 1.0, goal=""))
        feats = features(text, ctx, extra_markers=self._markers)
        state = view(text, feats, ctx, self.settings)
        tiers = self._tiers()
        results: list[TierResult] = []

        def attempt(tier: str, following: str, speculative) -> TierResult | None:
            """Run one tier. None when it failed (recorded as a visible hop); `_Cancelled` when the person cancelled."""
            if tier not in trace.tried:
                trace.tried.append(tier)
            try:
                return self._run(tier, text, state, ctx, trace, speculative)
            except TierFailure as failure:
                self._hop(trace, tier, following, failure.kind, FAILED)
            except ProviderError as error:
                if error.kind == CANCELLED:
                    raise _Cancelled from None
                self._hop(trace, tier, following, error.kind, FAILED)
            return None

        with ExitStack() as stack:
            try:
                speculative = self._start_speculative(stack, text, ctx) if len(tiers) == 2 else None
                for index, tier in enumerate(tiers):
                    following = tiers[index + 1] if index + 1 < len(tiers) else ANSWER_SLOT
                    result = attempt(tier, following, speculative)
                    if result is None:
                        continue
                    results.append(result)
                    accepted, confidence, calibrated = self._verdict(result)
                    if not accepted:
                        self._hop(trace, tier, following, UNCERTAIN, BELOW)
                        continue
                    decision = self._accepted(result, confidence, calibrated, feats)
                    if decision.acts and self.settings.confirm_acting:
                        if not self._confirmed(result, results, attempt, tiers, speculative):
                            return done(self._plain(ANSWER, WHY_UNCONFIRMED, result.tier, confidence, calibrated=calibrated))
                    elif speculative is not None and tier == TIER_CLASSIFIER:
                        speculative[1].cancel("the classifier was sure")
                    return done(decision)
                return done(self._fallback(results))
            except (Abort, _Cancelled):
                return done(self._plain(ANSWER, WHY_CANCELLED, TIER_ROUTER, 0.0))

    def _confirmed(self, result: TierResult, results: list[TierResult], attempt, tiers: list[str], speculative) -> bool:
        """Two keys to act (`confirm_acting`): the OTHER tier must read the request the same way, with at least the plain floor.

        The classifier's decision is confirmed by the fast model (asked now if it has not been), and the fast model's
        by the classifier's own reading. A missing, failed or disagreeing second reader is no confirmation.
        """
        other = TIER_FAST if result.tier == TIER_CLASSIFIER else TIER_CLASSIFIER
        if other not in tiers:
            return False
        second = next((r for r in results if r.tier == other), None)
        if second is None and other == TIER_FAST:
            second = attempt(TIER_FAST, ANSWER_SLOT, speculative)
        if second is None:
            return False
        confidence, _ = self.calibration.apply(second.tier, second.route, second.raw)
        return second.route == result.route and confidence >= self.settings.min_confidence

    def is_stop(self, text: str) -> bool:
        """Whether `text` is, whole, a built-in or configured stop phrase."""
        return is_stop(text, self._stops)

    # -- the tiers --------------------------------------------------------------------------------------------------

    def _tiers(self) -> list[str]:
        return [t for t, client in ((TIER_CLASSIFIER, self.classifier), (TIER_FAST, self.fast_llm)) if client is not None]

    def _start_speculative(self, stack: ExitStack, text: str, ctx: Context):
        """Start the fast call now, beside the classifier, when asked to: both are read-only and safe to run twice.

        The child control is cancelled by the person's cancel (it is linked) and by this router when the classifier
        is sure, so the unused call's connection is closed instead of left to finish. Returns (future, control).
        """
        if not self.settings.speculative_fast:
            return None
        child = stack.enter_context(linked(current_control()))
        future = spawn(lambda: ask_fast(self.fast_llm, text, ctx, self.settings, clock=self._clock), child)
        return future, child

    def _run(self, tier, text, state, ctx, trace: _Trace, speculative) -> TierResult:
        if tier == TIER_FAST:
            if speculative is not None:
                return speculative[0].result()
            return ask_fast(self.fast_llm, text, ctx, self.settings, clock=self._clock)
        before = {id(e) for e in _events(self.classifier)}
        try:
            return ask_classifier(self.classifier, state, clock=self._clock)
        finally:  # the chain's own failovers are part of the same visible story
            trace.switches.extend(e for e in _events(self.classifier) if id(e) not in before)

    # -- judging a result -------------------------------------------------------------------------------------------

    def _verdict(self, result: TierResult) -> tuple[bool, float, bool]:
        s = self.settings
        confidence, calibrated = self.calibration.apply(result.tier, result.route, result.raw)
        if result.route == STOP:
            floor = s.stop_min_confidence
        elif result.route in ACTING:
            floor = s.act_min_confidence
        else:
            floor = s.min_confidence
        accepted = confidence >= floor
        if accepted and result.route in ACTING and result.margin is not None and result.margin < s.act_min_margin:
            accepted = False  # two readings are close: that is doubt about acting
        return accepted, confidence, calibrated

    def _accepted(self, result: TierResult, confidence: float, calibrated: bool, feats: Features) -> Decision:
        if result.route in ACTING and feats.injection_marker:
            # Whatever the tier thinks, a request that carries an override attempt does not act. It is answered.
            return self._plain(ANSWER, WHY_INJECTION, result.tier, confidence, calibrated=calibrated, provider=result.provider)
        why = WHY_STOP_MODEL if result.route == STOP else WHY_CLASSIFIER if result.tier == TIER_CLASSIFIER else WHY_FAST
        return Decision(
            result.route,
            OWNER[result.route],
            confidence,
            why,
            result.tier,
            goal=result.goal if result.route in ACTING | {REASON} else "",
            reply=result.reply,
            question=result.question,
            language=result.language,
            calibrated=calibrated,
            provider=result.provider,
        )

    def _fallback(self, results: list[TierResult]) -> Decision:
        if not results:
            return self._plain(ANSWER, WHY_TIERS_FAILED, TIER_ROUTER, 0.0)
        if self.settings.escalate_to_reason:
            sure_not_acting = [(self._not_acting_mass(r), r) for r in results]
            mass, best = max(sure_not_acting, key=lambda pair: pair[0])
            if best.route in {ANSWER, REASON} and mass >= self.settings.min_confidence:
                return Decision(
                    REASON,
                    OWNER[REASON],
                    mass,
                    WHY_ESCALATED,
                    best.tier,
                    goal=best.goal,
                    language=best.language,
                    provider=best.provider,
                )
        top = max(self.calibration.apply(r.tier, r.route, r.raw)[0] for r in results)
        return self._plain(ANSWER, WHY_UNCERTAIN, TIER_ROUTER, top)

    @staticmethod
    def _not_acting_mass(result: TierResult) -> float:
        """How much of what a tier believes is on answer or reason: sure it is not an action, whatever the depth."""
        if result.probs:
            return sum(result.probs.get(r, 0.0) for r in (ANSWER, REASON))
        return result.raw if result.route in {ANSWER, REASON} else 0.0

    # -- plumbing ---------------------------------------------------------------------------------------------------

    def _plain(self, route: str, why: str, tier: str, confidence: float, **kw) -> Decision:
        return Decision(route, OWNER[route], confidence, why, tier, **kw)

    def _hop(self, trace: _Trace, tier: str, following: str, kind: str, reason: str) -> None:
        trace.errors.append(f"{tier}:{kind}")
        event = SwitchEvent(role="router", from_slot=tier, to_slot=following, kind=kind, reason=reason)
        trace.switches.append(event)
        if self._on_event is not None:
            self._on_event(event)


def _events(client) -> list[SwitchEvent]:
    """The switches a classifier chain has recorded so far (none for a client that is not a chain)."""
    return list(getattr(getattr(client, "chain", None), "events", ()))

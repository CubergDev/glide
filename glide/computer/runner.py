"""The step loop and the run folder."""

from __future__ import annotations

import json
import re
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from ..providers.errors import AllProvidersFailed, ProviderError
from . import diagnostics
from .actions import Context, perform
from .calls import Calls, MeteredClassifier, MeteredWriter
from .config import (
    DEFAULT_DELAY,
    DEFAULT_HANDOFFS,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_READINESS_TIMEOUT,
    DEFAULT_RESEARCH_CALLS,
    DEFAULT_STEPS,
    MAX_OPTIONS,
)
from .control import RunControl, checkpoint, controlled, current_control
from .decide import OFFSCREEN_PREFIX, Decision, decide, offscreen_records
from .generation import GenerationUnavailable
from .models import Abort, DesktopError, Guidance, Item, Screen, Signature, same_screen, signature
from .perception import OcrCache, capture, perceive
from .platform_adapter import desktop
from .report import Log, annotate, ax_count, render_payload, top
from .timing import format_timing, phase, summarize
from .writer import Answer, WriterError, compose_answer

# Two ways a run stalls, both read off the screen rather than off the history line, because an
# action's description says what was attempted and only the next capture says what came of it.
MAX_IDLE = 3  # consecutive actions that left the screen as it was: refusals, waits on a spinner, scrolls at the bottom
MAX_REPEATS = 2  # consecutive actions already taken on the same screen earlier in the run: a cycle, or a click that does nothing
MAX_STALLS = 3  # stalls with no new page between them; the last is final, since two focuses from the writer did not free the run
WOULD_DO_CHARS = 60  # the most of an item's text quoted when describing a dry run's move
FAILURE_CHARS = 400  # the most of a failure's words kept in run.json
EARLIER_LINES = 600  # lines of text from the screens before the last one that the answer may also be read from
MAX_QUESTIONS = 3  # questions the writer may put to the user in one run; an empty reply ends the asking sooner

# The outcomes the writer is handed, each in words it can pass on. A dry run took no action and
# an abort is the user's own stop, so neither has anything to report.
STOPPED = {
    "done": "the classifier judged the goal already achieved on this screen",
    "nothing helps": "the classifier found nothing on this screen that helps with the goal",
    "low confidence": "the classifier was not confident enough in any next action",
    "stalled": "the last actions changed nothing",
    "stuck": (
        "the agent is stuck: its last actions changed nothing, as they did twice before on these same pages, "
        "and the focus given each time did not help, so the run ends with this answer"
    ),
    "step limit": "the run used every step it was allowed",
}


@dataclass
class RunConfig:
    goal: str
    out: Path
    act: bool = False
    steps: int = DEFAULT_STEPS
    min_confidence: float = DEFAULT_MIN_CONFIDENCE
    delay: float = DEFAULT_DELAY
    handoffs: int = DEFAULT_HANDOFFS  # stops the writer may send the classifier back from; 0 makes every stop final
    image: Path | None = None  # replay a saved capture (never acts)
    app: str | None = None  # frontmost app to report during replay
    url: str | None = None  # browser URL to report during replay
    # Start each capture's OCR on a thread of its own while the capture asks for the app, window,
    # field, and URL (see `OcrCache.read_ahead`). Worth it where those questions are slow, as over an
    # OSWorld VM; off for this machine, whose own are quick.
    read_ahead: bool = False
    # D3: the goal, answers, history, screenshots and step files are written only when this is set. Off by default.
    record_content: bool = False
    engine: str = "legacy"  # "structured" is the execution engine (execution/engine.py)
    execution_browser: str = ""
    readiness_timeout: float = DEFAULT_READINESS_TIMEOUT
    research_calls: int = DEFAULT_RESEARCH_CALLS  # structured engine: the most model calls a research task may use
    # The folder every run of one user shares (the parent of each run folder). A write that never reported its result
    # leaves a marker there that blocks the next acting run until an operator removes it. None: no cross-run guard.
    journal: Path | None = None

    @property
    def replay(self) -> bool:
        return self.image is not None


@dataclass(frozen=True)
class Handoff:
    """One stop the writer sent the classifier back from."""

    step: int
    outcome: str  # why the classifier stopped
    focus: str  # what the writer sent it back to do
    actions: int  # how many actions the run had taken by then, so a focus that led to none can be told


@dataclass
class RunState:
    history: list[str] = field(default_factory=list)
    timings: list[dict[str, float]] = field(default_factory=list)
    idle: int = 0  # actions in a row that changed nothing on screen
    repeats: int = 0  # actions in a row already taken on the same screen
    stalls: int = 0  # stalls since the run last reached a page it had not been on
    pages: set[tuple[str, str | None]] = field(default_factory=set)  # every (app, URL) the run has shown
    last: Signature | None = None  # the screen the last action was taken on
    seen: list[tuple[Signature, str | None]] = field(
        default_factory=list
    )  # every screen acted on, with the action; None for a wait
    outcome: str = "crashed"  # every way out of the loop names its own; only an unexpected exception leaves this
    ocr_cache: OcrCache = field(default_factory=OcrCache)  # carries one step's OCR into the next
    view: tuple[Screen, list[Item]] | None = None  # the latest capture, until an action makes it stale
    uncertain: bool = False
    readback: str = "not needed"
    failure: str = ""  # why the run ended, when it did not end with an answer; scrubbed before it is stored
    answer: Answer | None = None  # the writer's latest; the last one is the run's answer
    guidance: Guidance = field(default_factory=Guidance)  # the writer's focus and the user's replies, as they stand
    handoffs: list[Handoff] = field(default_factory=list)
    calls: Calls = field(default_factory=Calls)  # requests to each model, over the whole run
    progress: list[dict] = field(default_factory=list)
    plan_revisions: int = 0
    first_action_s: float | None = None
    unsupported_capabilities: list[str] = field(default_factory=list)
    cancel_reason: str = ""
    would_do: str | None = None  # a dry run's first move in words; kept in memory only, never written (D3)
    summary: dict = field(default_factory=dict)  # the structured engine's run.json, for the recorder's report


def run(cfg: RunConfig, ctx_factory, classifier_factory=None, control: RunControl | None = None) -> RunState:
    """Drive the run under a `RunControl` and report how it ended. `_run` is the legacy loop; `cfg.engine == "structured"`
    hands the run to `execution.engine.run_execution`, with the same classifier, control and recorder.

    `classifier_factory()` returns the classifier as a context manager with the `system_one` of
    `TypeSafeClient`: a `ChainedClassifier`, an `LLMClassifier`, or any stand-in. It is required by both engines: the
    classifier always comes from the provider chains (`glide_config.classifier`), so a failover is never skipped and
    every switch is visible (D7). A run with none is refused with a ValueError before anything is made.

    `control` is the run's stop gate (see `control.py`). Left out, a private one is made, so a caller that
    never stops the run needs nothing. Its events are delivered to its own `emit`; nothing is printed.

    A provider that fails is not a crash: the run ends with the outcome "provider failure" and a run.json as for any
    other stop (the legacy loop logs each provider and why; the structured engine says it in one fixed sentence).

    Content (goal, answers, history, page text, raw URLs, screenshots, step files) is written only when
    `cfg.record_content` is set. Otherwise the run folder holds run.json with counts, timings, the outcome
    and a scrubbed `failure`.
    """
    if classifier_factory is None:
        raise ValueError("a run needs classifier_factory: pass the provider chains' classifier (glide_config.classifier)")
    if cfg.replay:
        cfg = replace(cfg, engine="legacy")  # a replay must never inspect or act on the live machine
    control = control or RunControl(str(uuid.uuid4()))
    recorder = diagnostics.Diagnostics(cfg.out, record_content=cfg.record_content, task_id=control.task_id)
    with controlled(control), recorder.activate():
        recorder.event("task_started", engine=cfg.engine, act=cfg.act, goal=cfg.goal, configuration=asdict(cfg))
        control.event("accepted", "Working on your request.")
        try:
            state = _run(cfg, ctx_factory, classifier_factory)
        except BaseException as error:
            cancelled = isinstance(error, Abort | KeyboardInterrupt)
            recorder.finish({"outcome": "aborted" if cancelled else "crashed", "engine": cfg.engine}, error=error)
            raise
        recorder.finish(
            {
                "outcome": state.outcome,
                "failure": state.failure,
                "engine": cfg.engine,
                "act": cfg.act,
                "steps_taken": len(state.history),
                "goal_achieved": state.answer.achieved if state.answer else None,
                "uncertain": state.uncertain,
                **state.summary,
            }
        )
        if state.outcome.startswith("aborted"):
            hint = (
                " Move the pointer out of the top-left abort corner before retrying."
                if state.cancel_reason == "mouse in top-left corner"
                else ""
            )
            control.event("cancelled", "Stopped." + hint, outcome=state.outcome)
        elif state.outcome == "dry run":
            control.event("dry_run", state.would_do or "", outcome=state.outcome)
        elif state.failure:
            control.event("blocked", spoken_failure(state), outcome=state.outcome)
        elif state.answer is not None:
            control.event(
                "completed" if state.answer.achieved else "blocked",
                state.answer.text,
                outcome=state.outcome,
                spoken_text=getattr(state.answer, "spoken_text", ""),  # set by the writer boundary
            )
        else:
            control.event("blocked", "The task ended without a verified result.", outcome=state.outcome)
        return state


def spoken_failure(state: RunState) -> str:
    """The failure as a person is told it. The structured engine's own report keeps the capability list and the
    effect counts; neither belongs in a spoken sentence (the counts are in run.json's `progress`)."""
    if state.unsupported_capabilities and "Available capabilities:" in state.failure:
        return "I couldn't complete the requested task with the available browser capabilities. Details are in the task report."
    return re.sub(r"\s+Verified \d+ effect\(s\); \d+ remain\.$", "", state.failure)


def _run(cfg: RunConfig, ctx_factory, classifier_factory=None) -> RunState:
    """Drive the loop. ctx_factory(typesafe, history) builds the action Context."""
    if cfg.engine not in ("legacy", "structured"):
        raise ValueError("Unknown execution engine")
    if cfg.engine == "structured":
        from .execution.engine import run_execution

        return run_execution(cfg, ctx_factory, classifier_factory)
    cfg.out.mkdir(parents=True, exist_ok=True)
    log = Log(cfg.out / "run.log" if cfg.record_content else None, enabled=cfg.record_content)
    log(f"run folder: {cfg.out}")
    if cfg.act:
        log(f"driving the machine. abort: {desktop.abort_hint()}.")

    state = RunState()
    started = time.time()
    try:
        with classifier_factory() as typesafe:
            ctx = metered(ctx_factory(typesafe, state.history), state.calls)
            for step in range(1, cfg.steps + 1):
                if run_step(cfg, ctx, state, step, log):
                    continue
                if not hand_off(cfg, ctx, state, step, log):
                    break
                ctx = replace(ctx, guidance=state.guidance)
            else:
                log(f"\nstopped after {cfg.steps} steps")
                state.outcome = "step limit"
                hand_off(cfg, ctx, state, cfg.steps, log)
    except GenerationUnavailable as error:
        diagnostics.exception(error, stage="generation")
        state.outcome = "generation unavailable"
        state.failure = str(error)
        _write_unobserved(state)
    except (DesktopError, subprocess.CalledProcessError) as error:
        diagnostics.exception(error, stage="desktop_operation")
        state.outcome = "desktop unavailable"
        state.failure = (
            str(error) if isinstance(error, DesktopError) else "A desktop operation failed; completion could not be verified."
        )
        _write_unobserved(state)
    except (KeyboardInterrupt, Abort) as e:
        state.uncertain = bool(current_control() and current_control().in_flight)
        if state.uncertain:
            # Read only, on the execution thread. A changed screen cannot prove that a
            # submit did or did not reach the server, so replacement remains gated.
            try:
                with controlled(None):
                    observed = capture(cfg.image, cfg.app, cfg.url, ctx.browser)
                    state.view = (observed, perceive(observed, MAX_OPTIONS, cfg.goal))
                state.readback = "captured; completion unknown"
            except Exception:
                state.readback = "unavailable; completion unknown"
        state.cancel_reason = str(e)
        state.outcome = f"aborted ({e or 'Ctrl-C'})"
        log(f"\n{state.outcome} after {len(state.history)} actions")
    except ProviderError as e:
        diagnostics.exception(e, stage="provider")
        state.outcome = "provider failure"
        state.failure = provider_failure(e)
        _write_unobserved(state)
        log(f"\nprovider failure after {len(state.history)} actions: {state.failure}")
    finally:
        summary = {
            "goal": cfg.goal,
            "act": cfg.act,
            "steps_taken": len(state.history),
            "outcome": state.outcome,
            "failure": state.failure,
            "answer": state.answer.text if state.answer else None,
            "goal_achieved": state.answer.achieved if state.answer else None,
            "seconds": round(time.time() - started, 1),
            "calls": state.calls.summary(),
            "usage": state.calls.tokens(),
            "handoffs": [asdict(h) for h in state.handoffs],
            "questions": [asdict(e) for e in state.guidance.exchanges],
            "timing": summarize(state.timings),
            "history": state.history,
            "config": {k: str(v) for k, v in asdict(cfg).items()},
        }
        if not cfg.record_content:
            summary = {k: summary[k] for k in ("steps_taken", "seconds", "calls", "usage", "timing")}
            summary["outcome"] = "aborted" if state.outcome.startswith("aborted") else state.outcome
        # Named providers and error kinds, without URLs or anything key-like, in both modes.
        summary["failure"] = diagnostics.scrub_text(state.failure, limit=FAILURE_CHARS) if state.failure else None
        summary["interrupted_readback"] = state.readback
        summary["task_id"] = current_control().task_id if current_control() else ""
        diagnostics.write_private(cfg.out / "run.json", json.dumps(summary, indent=2))
        log(f"{state.calls.line()}  handoffs {len(state.handoffs)}  questions {len(state.guidance.exchanges)}")
        log(f"run folder: {cfg.out}")
    return state


def _write_unobserved(state: RunState) -> None:
    """A run that ends on a failure right after a write was sent never saw its effect: say so, so it is not retried.

    The model that checks a typed value fails after the keystrokes went out, for one. Only a fresh observation
    clears `in_flight` (see control.py), so it still being set is the evidence.
    """
    state.uncertain = bool(current_control() and current_control().in_flight)
    if state.uncertain:
        state.readback = "unavailable; completion unknown"


def provider_failure(error: ProviderError) -> str:
    """Which provider failed and why, on one line: every slot a chain tried, or the one that raised."""
    if isinstance(error, AllProvidersFailed):
        tried = "; ".join(f"{name} [{err.kind}] {err}" for name, err in error.errors)
        return f"every {error.role} provider failed: {tried}" if tried else str(error)
    return f"{error.provider or 'a provider'} [{error.kind}] {error}"


def metered(ctx: Context, calls: Calls) -> Context:
    """The same context, with every request to either model counted."""
    return replace(
        ctx,
        typesafe=MeteredClassifier(ctx.typesafe, calls),
        writer=MeteredWriter(ctx.writer, calls) if ctx.writer is not None else None,
    )


def hand_off(cfg: RunConfig, ctx: Context, state: RunState, step: int, log: Log) -> bool:
    """The classifier stopped: hand the run to the writer. True when the writer handed it back.

    The classifier can stop on the right page but cannot say what the page says, and it can stop
    short because one sentence of goal does not say which of two good moves comes first. The writer
    reads the screen and answers for the user either way. When the goal is not reached it may also
    name a focus, which sends the classifier back to work with `state.guidance` saying what on, or
    a question, which goes to the user first; the writer then reads the same screen again with the
    reply. Each reply is new information and each focus must lead to an action, so the exchange
    cannot go round on itself: a focus the classifier could not act on leaves the answer that came
    with it standing, and a stall with no new page since the last two is final (see `stuck`).
    """
    stuck(state, log)
    stopped = STOPPED.get(state.outcome)
    if stopped is None:
        return False
    if ctx.writer is None:
        log("\nno answer: the writer is disabled (no writer provider is configured; see glide.toml)")
        return False
    if state.handoffs and state.handoffs[-1].actions == len(state.history) and state.answer is not None:
        log(f"\nanswer ({verdict(state.answer)}; the focus led to no action, so the last answer stands):\n  {state.answer.text}")
        return False

    may_resume = step < cfg.steps and len(state.handoffs) < cfg.handoffs and state.outcome != "stuck"
    reviews: list[dict] = []
    while True:
        can_ask = may_resume and ctx.ask is not None and len(state.guidance.exchanges) < MAX_QUESTIONS
        started = time.perf_counter()
        try:
            answer = review(cfg, ctx, state, stopped, can_ask)
        except WriterError as e:
            if e.halt:  # an account spent or every slot down ends the run, as it does anywhere else the writer is asked
                raise
            state.failure = str(e)
            log(f"\nno answer: the writer failed ({e})")
            return False
        state.answer = answer
        seconds = time.perf_counter() - started
        record = {"outcome": state.outcome, "seconds": round(seconds, 3), **asdict(answer), "reply": None, "handed_back": False}
        reviews.append(record)
        try:
            resuming = may_resume and not answer.achieved
            if resuming and can_ask and answer.question:
                log(f"\nreview ({verdict(answer)}, {seconds:.1f}s):\n  {answer.text}\n  the writer asks: {answer.question}")
                checkpoint()
                active = current_control()
                if active:
                    active.event("question", answer.question, question_id=str(uuid.uuid4()))
                reply = ctx.ask(answer.question).strip()
                checkpoint()
                record["reply"] = reply
                log(f"  > {reply}", echo=False)  # the terminal already shows what was typed
                if not reply:
                    log("  no reply, so the answer stands")
                    return False
                state.guidance = state.guidance.heard(answer.question, reply)
                if cfg.act and not cfg.replay and state.view is not None:
                    desktop.activate(state.view[0].app)  # answering took the terminal to the front; put the work back there
                continue
            if resuming and answer.focus:
                log(
                    f"\nreview ({verdict(answer)}, {seconds:.1f}s):\n  {answer.text}\n  back to the classifier, focus: {answer.focus}"
                )
                record["handed_back"] = True
                state.handoffs.append(Handoff(step, state.outcome, answer.focus, len(state.history)))
                state.guidance = state.guidance.focused(answer.focus)
                state.idle = state.repeats = 0  # the stall was under the old focus; the new one starts clean
                return True
            log(f"\nanswer ({verdict(answer)}, {seconds:.1f}s):\n  {answer.text}")
            return False
        finally:
            if cfg.record_content:
                (cfg.out / f"step-{step:03d}-review.json").write_text(json.dumps(reviews, indent=2), encoding="utf-8")


def stuck(state: RunState, log: Log) -> None:
    """Count a stall, and call the run stuck once it is the third with no new page between them.

    A stall hands the run to the writer like any stop, and its focus usually frees the classifier.
    When the classifier stalls again on the pages it was already on, the focus did not help, and
    each further round costs a writer call and a few futile actions. On OSWorld's Chrome tasks every
    run that stalled a third time on the same pages failed all the same, up to 210 s later, and no
    solved run stalled more than twice. A new page is the progress that starts the count again, so
    a long task that stalls now and then on its way is not cut short.
    """
    if state.outcome != "stalled":
        return
    state.stalls += 1
    if state.stalls >= MAX_STALLS:
        log(f"  stuck: {state.stalls} stalls without reaching a new page; the run ends here")
        state.outcome = "stuck"


def verdict(answer: Answer) -> str:
    return "goal achieved" if answer.achieved else "goal not achieved"


def review(cfg: RunConfig, ctx: Context, state: RunState, stopped: str, can_ask: bool) -> Answer:
    """Have the writer read the screen the classifier stopped on.

    The last step's capture serves when nothing acted after it. An action makes it stale, so the
    screen is captured again, and saved so the answer can be checked against what it was read from.
    """
    if state.view is None:
        desktop.check_abort()
        screen = capture(cfg.image, cfg.app, cfg.url, ctx.browser)
        if cfg.record_content:
            screen.image.save(cfg.out / "answer-raw.png")
        state.view = (screen, perceive(screen, MAX_OPTIONS, cfg.goal))
    active = current_control()
    if active and active.in_flight:
        active.in_flight = False
        active.event("action_checked")
    screen, items = state.view
    earlier = earlier_screens(state, signature(screen, items))
    earlier_stops = [{"after_action": h.actions, "why": STOPPED[h.outcome], "focus_given": h.focus} for h in state.handoffs]
    return compose_answer(
        ctx.writer, cfg.goal, screen, items, state.history, stopped, earlier, state.guidance, earlier_stops, can_ask
    )


def earlier_screens(state: RunState, final: Signature, budget: int = EARLIER_LINES) -> list[dict]:
    """The distinct screens the run passed through before the one it ended on, oldest first.

    The goal may ask for something that was on the way (a price on the listing, not on the checkout),
    and the run's own captures are the only place the answer may come from. A screen seen twice
    is sent once. The newest screens are kept whole and the oldest dropped once the line budget
    is spent, since the writer reads all of it in one call.
    """
    out: list[Signature] = []
    lines_left = budget
    for seen, _ in reversed(state.seen):
        if same_screen(seen, final) or any(same_screen(seen, kept) for kept in out):
            continue
        lines_left -= len(seen[3])
        if lines_left < 0:
            break
        out.append(seen)
    return [{"app": app, "url": url, "text": [text for text, _ in lines]} for app, url, _, lines in reversed(out)]


def run_step(cfg: RunConfig, ctx: Context, state: RunState, step: int, log: Log) -> bool:
    checkpoint()
    desktop.check_abort()
    timing: dict[str, float] = {}
    started = time.perf_counter()
    cache = None if cfg.replay else state.ocr_cache
    with phase(timing, "capture"):
        screen = capture(cfg.image, cfg.app, cfg.url, ctx.browser, timing, ahead=cache if cfg.read_ahead else None)
    items = perceive(screen, MAX_OPTIONS, cfg.goal, timing, cache)
    state.view = (screen, items)
    active = current_control()
    if active and state.history:
        active.in_flight = False
        active.event("action_checked")
    if not screen_moved(state, screen, items, log):
        return False
    tried = tried_here(state)
    prefix = cfg.out / f"step-{step:03d}"  # three digits, so a run of 100 steps still lists in order
    if cfg.record_content:
        screen.image.save(prefix.with_name(prefix.name + "-raw.png"))
        prefix.with_name(prefix.name + "-payload.txt").write_text(
            render_payload(cfg.goal, screen, items, state.history, ctx.browser, ctx.email, tried, ctx.guidance), encoding="utf-8"
        )

    with phase(timing, "decide"):
        decision = decide(ctx.typesafe, cfg.goal, screen, items, state.history, ctx.browser, ctx.email, tried, ctx.guidance)
    by_index = {str(it.index): it for it in items}
    checkpoint()
    if cfg.record_content:
        annotate(screen, items, decision.chosen, prefix.with_suffix(".png"))

    field_desc = f" field={screen.field.role}:{screen.field.label!r}" if screen.field else ""
    log(
        f"\nstep {step}: app={screen.app!r}{field_desc} url={screen.url!r} items={len(items)} ax={ax_count(items)} "
        f"offscreen={len(screen.offscreen)} kind={decision.kind.choice} ({decision.kind.confidence:.2f}) "
        f"site={decision.site.choice}"
    )
    for key, p in top(decision.kind, 4):
        log(f"  {p:5.2f}  {key}")
    if decision.item is not None:
        log(f"  item ({decision.item.confidence:.2f}):")
        for key, p in top(decision.item, 4):
            log(f"  {p:5.2f}  [{key}] {by_index[key].text!r}")
    if decision.offscreen is not None:
        log(f"  offscreen ({decision.offscreen.confidence:.2f}):")
        for key, p in top(decision.offscreen, 3):
            log(f"  {p:5.2f}  [{key}] {screen.offscreen[int(key)].label!r}")

    keep_going = resolve(cfg, ctx, state, screen, items, decision, timing, log)
    timing.setdefault("act", 0.0)
    timing["total"] = round(time.perf_counter() - started, 3)
    state.timings.append(timing)

    if cfg.record_content:
        prefix.with_name(prefix.name + "-answers.json").write_text(
            json.dumps(answers(decision, screen, items, timing, tried, state), indent=2), encoding="utf-8"
        )
        log(f"  files: {prefix.name}-raw.png, {prefix.name}.png, {prefix.name}-payload.txt, {prefix.name}-answers.json")
    log(format_timing(timing))

    if state.view is None:  # an action ran: let the screen settle before the next step, or the answer, reads it
        desktop.sleep_watching(cfg.delay)
    return keep_going


def resolve(
    cfg: RunConfig,
    ctx: Context,
    state: RunState,
    screen: Screen,
    items: list[Item],
    decision: Decision,
    timing: dict[str, float],
    log: Log,
) -> bool:
    """Apply the stop rules, then the action. True to keep looping."""
    if decision.stops:
        log(f"  model says {decision.kind.choice!r}; stopping")
        state.outcome = "done" if decision.kind.choice == "done" else "nothing helps"
        return False
    if decision.confidence < cfg.min_confidence:
        log(f"  confidence {decision.confidence:.2f} below {cfg.min_confidence}; stopping")
        state.outcome = "low confidence"
        return False
    if not cfg.act or cfg.replay:
        log(f"  would do: {decision.chosen}. dry run (pass --act without --image to drive the machine)")
        state.outcome = "dry run"
        state.would_do = describe_move(decision, items)
        return False

    checkpoint()
    active = current_control()
    with phase(timing, "act"):
        what = perform(decision, screen, items, ctx)
    state.view = None
    state.history.append(what)
    active = current_control()
    if active:
        active.event("progress", "Checking the result of the action.")
    log(f"  did: {what}")
    return not repeating(state, what, decision.kind.choice == "wait", log)


def screen_moved(state: RunState, screen: Screen, items: list[Item], log: Log) -> bool:
    """Count the actions that left the screen as it was, and stop once too many did in a row.

    The capture is the only witness to what an action did. Compared with the one the action was
    taken on, an unchanged screen means a refused action, a wait on a page still loading, or a
    scroll that has run out of page; any of them is fine a couple of times. A page the run has not
    been on before starts the count of stalls again (see `stuck`).
    """
    now = signature(screen, items)
    if now[:2] not in state.pages:
        state.pages.add(now[:2])
        state.stalls = 0
    if state.last is not None:
        state.idle = state.idle + 1 if same_screen(now, state.last) else 0
    state.last = now
    if state.idle >= MAX_IDLE:
        log(f"  the last {MAX_IDLE} actions changed nothing on screen; stopping")
        state.outcome = "stalled"
        return False
    return True


def tried_here(state: RunState) -> list[str]:
    """The actions already taken on the screen now showing, oldest first, for the classifier to steer around."""
    return [what for seen, what in state.seen if what is not None and state.last is not None and same_screen(seen, state.last)]


def repeating(state: RunState, what: str, waiting: bool, log: Log) -> bool:
    """Count the actions already taken on the same screen earlier, and stop once too many run in a row.

    The same action on the same screen led somewhere once, and this is where it led: back here.
    That is a cycle through two pages as much as a button that does nothing. A wait is recorded
    with no action, so the screen it was taken on still reaches the answer, but it is never a
    repeat and never listed as tried: waiting is repeating by design, and the classifier must
    stay free to wait again. The idle count bounds it instead.
    """
    if waiting:
        state.seen.append((state.last, None))
        return False
    state.repeats = state.repeats + 1 if what in tried_here(state) else 0
    state.seen.append((state.last, what))
    if state.repeats >= MAX_REPEATS:
        log(f"  {MAX_REPEATS} actions in a row already taken on the same screen; stopping")
        state.outcome = "stalled"
        return True
    return False


def describe_move(decision: Decision, items: list[Item]) -> str:
    """The move a dry run stopped at, in words. Item text is screen content, so this stays in memory (D3)."""
    chosen = str(decision.chosen)
    if chosen.isdigit():
        text = next((it.text for it in items if str(it.index) == chosen), "")
        return f"click {text[:WOULD_DO_CHARS]!r}" if text else "click an item"
    if chosen.startswith(OFFSCREEN_PREFIX):
        return "press an off-screen control"
    return chosen.replace("_", " ")


def answers(
    decision: Decision, screen: Screen, items: list[Item], timing: dict[str, float], tried: list[str], state: RunState
) -> dict:
    """What the classifier returned for this step, plus what it cost and where the stop rules stand."""
    return {
        "kind": decision.kind.choice,
        "kind_confidence": decision.kind.confidence,
        "kind_probabilities": decision.kind.probabilities,
        "item": decision.item.choice if decision.item else None,
        "item_confidence": decision.item.confidence if decision.item else None,
        "item_probabilities": decision.item.probabilities if decision.item else None,
        "site": decision.site.choice,
        "site_probabilities": decision.site.probabilities,
        "offscreen": decision.offscreen.choice if decision.offscreen else None,
        "offscreen_probabilities": decision.offscreen.probabilities if decision.offscreen else None,
        "offscreen_controls": offscreen_records(screen.offscreen),
        "chosen": decision.chosen,
        "confidence": decision.confidence,
        "already_tried_on_this_screen": tried,
        "idle_actions": state.idle,
        "repeated_actions": state.repeats,
        "timing": timing,
        "items": [asdict(it) for it in items],
        "field": screen.field.record() if screen.field else None,
        "app": screen.app,
        "url": screen.url,
    }

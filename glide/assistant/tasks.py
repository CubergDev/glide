"""Running a computer task: the bridge from the assistant to the screen-driving loop (glide/computer/).

A task is `runner.run` on a worker thread, so the assistant stays free to hear "stop". Three promises:

- A dry run is the default at every layer. `act=True` has to be passed in by the caller, and it is checked
  against the macOS Accessibility permission first, as `glide-computer --act` does.
- A stop reaches the loop by two routes that raise the same `Abort` with the same reason. `stop()` cancels the
  task's `RunControl` (glide/computer/control.py): every step, every action (`dispatch`) and every platform
  input call takes a checkpoint, and connections registered with `closing_on_cancel` are closed so a blocked
  request ends at once instead of being waited out. It also sets the stop event that `abort_on` watches:
  while a task runs, `desktop.check_abort()` is replaced by one that also raises when that event is set,
  which keeps working for code that has no control in scope. After `stop()`, no further action is taken.
  An action that was already sent cannot be taken back; the run records it as "completion unknown", reads the
  screen once and never replays it, and the `TaskResult` carries that (`uncertain`, `readback`) so the assistant tells
  the person in a fixed sentence instead of reporting a clean stop. The replacement is process-wide, so only one task
  runs at a time.
- The loop's own words are data. What the writer read off the screen is spoken and printed, never routed.
- A question the writer puts to the user is answered by `ComputerTask.answer` and by nothing else. It is opt-in
  (`on_question`): without it the loop never asks (`ask=None`), as before. A stop, or anything that stops the task,
  ends the wait for the answer.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from ..computer.control import RunControl
from .phrases import say

DEFAULT_RUNS_DIR = Path("runs")
STOPPED_BY_USER = "stopped by the user"

_ACTIVE = threading.Lock()  # held while a task runs: the abort hook is process-wide, so one task at a time
_NOTHING = object()


class TaskBusy(RuntimeError):
    """A task is already running in this process."""


@contextmanager
def abort_on(stop: threading.Event, control=None) -> Iterator[None]:
    """While the block runs, `desktop.check_abort()` also raises `Abort` once `stop` is set.

    With a `RunControl`, a cancel of it counts as a stop too, with the reason it was cancelled for, so
    whichever of the two fires first gives the same outcome.

    The replacement is assigned through `desktop`, which forwards to the platform adapter, so the calls the
    adapter makes to its own `check_abort` (between the characters of `type_text`, in `click_at`, while
    `sleep_watching`) reach it too: stopping lands mid-typing, not at the end of the word. The original
    is put back afterwards, also when the block raises.
    """
    from ..computer.models import Abort
    from ..computer.platform_adapter import desktop

    try:
        original = desktop.check_abort
    except RuntimeError:  # no desktop at all: the run fails on its own
        original = _NOTHING

    def check() -> None:
        if control is not None and control.cancelled.is_set():
            raise Abort(control.reason)
        if stop.is_set():
            raise Abort(STOPPED_BY_USER)
        if original is not _NOTHING:
            original()

    desktop.check_abort = check
    try:
        yield
    finally:
        if original is _NOTHING:
            del desktop.check_abort
        else:
            desktop.check_abort = original


@dataclass
class TaskResult:
    """How a task ended. `outcome` is the runner's own word ("dry run", "done", "stalled", "aborted (...)", ...)."""

    goal: str
    act: bool
    outcome: str
    answer: str | None = None  # the writer's words about what the screen shows
    achieved: bool | None = None
    failure: str | None = None
    folder: Path | None = None
    steps: int = 0
    seconds: float = 0.0
    would_do: str | None = None  # a dry run's first move, in words
    stopped: bool = False  # the user stopped it: nothing is said
    uncertain: bool = False  # the run ended with a write whose effect was never observed (RunState.uncertain)
    readback: str = "not needed"  # what the run saw afterwards, in the runner's own words (RunState.readback)

    def spoken(self, language: str | None = None) -> str:
        """What to say about the result: the writer's answer when there is one, else the outcome in a sentence.

        A run that may have left a write half done says so, in a fixed sentence, whatever else the result holds: no answer,
        no "done", and not the silence of an ordinary stop.
        """
        if self.uncertain:
            return say("uncertain", language)
        if self.stopped:
            return ""
        if self.answer and not self.failure and self.outcome not in UNANSWERED:  # an answer kept from an earlier stop is stale
            return self.answer
        if self.outcome == "dry run":
            return say("dry_run", language, what=self.would_do) if self.would_do else say("dry_run_plain", language)
        key = OUTCOME_PHRASES.get(self.outcome, "crashed")
        return say(key, language)

    def summary(self) -> str:
        """One or two lines for a terminal: what happened, why, and where the run folder is."""
        lines = [f"task {self.outcome}: {self.goal}"]
        if self.uncertain:
            lines.append(say("uncertain"))
            lines.append(f"what was seen afterwards: {self.readback}")
        elif self.answer:
            lines.append(self.answer)
        if self.would_do:
            lines.append(f"would do: {self.would_do}")
        if self.failure:
            lines.append(self.failure)
        if self.folder is not None:
            lines.append(f"run folder: {self.folder}")
        return "\n".join(lines)


# Outcomes that end a run on a failure: whatever answer an earlier stop of the same run left is not what happened.
UNANSWERED = ("provider failure", "generation unavailable", "desktop unavailable", "crashed")

OUTCOME_PHRASES = {
    "done": "done",
    "nothing helps": "nothing",
    "low confidence": "unsure",
    "stalled": "stalled",
    "stuck": "stalled",
    "step limit": "limit",
    "provider failure": "provider",
    "generation unavailable": "provider",
    "desktop unavailable": "desktop",
    "crashed": "crashed",
    "not permitted": "no_permission",
    "not configured": "not_configured",
}


class _Question:
    """A question put to the user, and the answer once there is one."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.reply = ""
        self.settled = threading.Event()


class ComputerTask:
    """One run of the screen-driving loop on a worker thread. Start it once; stop it any time.

    With `on_question(task, text)` the loop may ask the user something in the middle of a run: the callback shows and
    says the question, and the run waits until `answer(text)` is called, or until the task is stopped.
    """

    def __init__(
        self,
        goal: str,
        *,
        act: bool,
        config,
        folder: Path,
        on_question: Callable[[ComputerTask, str], None] | None = None,
        route: str = "",
    ) -> None:
        self.goal = goal
        self.route = route  # what the router decided ("execute" or "research"); "" when nothing did
        self.act = act
        self.folder = folder
        self.stop_event = threading.Event()
        self.events: list = []  # the run's TaskEvents, in order; nothing is printed
        self.control = RunControl(str(uuid.uuid4()), self.events.append)
        self.result: TaskResult | None = None
        self._config = config
        self._on_question = on_question
        self._pending: _Question | None = None
        self._question_lock = threading.Lock()
        self._finished = threading.Event()
        self._thread: threading.Thread | None = None

    def __repr__(self) -> str:
        state = "finished" if self._finished.is_set() else "running" if self._thread else "new"
        return f"<ComputerTask {state} act={self.act} {self.goal[:40]!r}>"

    @property
    def running(self) -> bool:
        return self._thread is not None and not self._finished.is_set()

    @property
    def stop_requested(self) -> bool:
        return self.stop_event.is_set()

    @property
    def pending_question(self) -> str | None:
        """The question the run is waiting on, or None."""
        with self._question_lock:
            pending = self._pending
        return pending.text if pending is not None and not pending.settled.is_set() else None

    def answer(self, text: str) -> bool:
        """Give the run the answer to its question. False when it is not waiting on one."""
        with self._question_lock:
            pending = self._pending
            if pending is None or pending.settled.is_set():
                return False
            pending.reply = text
            pending.settled.set()
        return True

    def _ask(self, text: str) -> str:
        """The loop's `ask`, on the task's thread: say the question, then wait for the answer or the stop."""
        pending = _Question(text)
        with self._question_lock:
            self._pending = pending
        try:
            with self.control.closing_on_cancel(pending.settled.set):  # a stop wakes the wait (and refuses a late question)
                self._on_question(self, text)
                pending.settled.wait()
            self.control.check(wait=False)  # stopped, not answered: the run ends here
            return pending.reply
        finally:
            with self._question_lock:
                self._pending = None

    def start(self, on_done: Callable[[ComputerTask], None] | None = None, release: Callable[[], None] | None = None) -> None:
        self._thread = threading.Thread(target=self._work, args=(on_done, release), name="glide-task", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Ask the loop to stop. It does so at its next check, which is before the next action."""
        self.stop_event.set()
        self.control.cancel(STOPPED_BY_USER)

    def wait(self, timeout: float | None = None) -> bool:
        """Block until the task has finished and `on_done(task)` has returned. False if `timeout` ran out first."""
        return self._finished.wait(timeout)

    # -- the worker -----------------------------------------------------------------------------

    def _work(self, on_done: Callable[[ComputerTask], None] | None, release: Callable[[], None] | None) -> None:
        started = time.monotonic()
        try:
            try:
                result = self._execute()
            except Exception as exc:  # a bug in the loop or its inputs: the user hears that it failed, the log has why
                unknown = self.control.in_flight  # a write was sent and nothing has observed its effect since
                result = TaskResult(
                    self.goal,
                    self.act,
                    "crashed",
                    failure=self._scrub(f"{type(exc).__name__}: {exc}"),
                    folder=self.folder,
                    uncertain=unknown,
                    readback="unavailable; completion unknown" if unknown else "not needed",
                )
            result.seconds = round(time.monotonic() - started, 1)
            result.stopped = result.stopped or self.stop_event.is_set()
            self.result = result
        finally:
            if release is not None:
                release()
            try:
                if self.result is not None and on_done is not None:
                    on_done(self)
            finally:
                self._finished.set()

    def _execute(self) -> TaskResult:
        from ..computer import config as computer_config
        from ..computer import runner
        from ..computer.actions import Context
        from ..computer.platform_adapter import desktop
        from ..providers.config import ConfigError

        if self.stop_event.is_set():
            return TaskResult(self.goal, self.act, f"aborted ({STOPPED_BY_USER})", stopped=True)
        if self.act and not desktop.accessibility_trusted():
            return TaskResult(
                self.goal,
                self.act,
                "not permitted",
                failure="this terminal lacks Accessibility permission; grant it in System Settings > Privacy & Security",
            )
        try:
            engine_fields = self._engine_fields()
        except ValueError as exc:  # a bad [computer], [browser] or [research] setting: nothing was started
            return TaskResult(self.goal, self.act, "not configured", failure=self._scrub(str(exc)), folder=self.folder)
        try:
            writer = self._config.writer()
            cfg = runner.RunConfig(
                goal=self.goal,
                out=self.folder,
                act=self.act,
                record_content=bool(getattr(self._config, "record_content", False)),
                **engine_fields,
                route=self.route,
                journal=self.folder.parent,  # one journal for every run folder under the runs directory
            )

            def ctx_factory(typesafe, history):
                # ask=None unless the assistant opted in: an input() here would race the terminal the assistant is
                # reading, and a voice user has no keyboard to answer on. The answer comes from `answer`, never from
                # whatever the user says next.
                return Context(
                    goal=self.goal,
                    browser=computer_config.browser(),
                    email=computer_config.email(),
                    typesafe=typesafe,
                    writer=writer,
                    history=history,
                    ask=self._ask if self._on_question is not None else None,
                )

            with abort_on(self.stop_event, self.control):
                state = runner.run(cfg, ctx_factory, classifier_factory=self._config.classifier, control=self.control)
        except ConfigError as exc:
            return TaskResult(self.goal, self.act, "not configured", failure=self._scrub(str(exc)), folder=self.folder)
        return TaskResult(
            self.goal,
            self.act,
            state.outcome,
            answer=state.answer.text if state.answer else None,
            achieved=state.answer.achieved if state.answer else None,
            failure=self._scrub(state.failure) if state.failure else None,
            folder=self.folder,
            steps=len(state.history),
            would_do=state.would_do if state.outcome == "dry run" else None,
            stopped=state.outcome.startswith("aborted"),
            uncertain=state.uncertain,
            readback=state.readback,
        )

    def _engine_fields(self) -> dict:
        """The engine in force (`features.engine_for`: the one resolver every front end shares) and, for the structured one, its
        browser and research settings from glide.toml. A bad setting raises ValueError, which `_execute` reports as not configured."""
        from .. import features
        from ..computer import browser_settings
        from ..computer import config as computer_config

        engine = features.engine_for(self._config)
        if engine != "structured":
            return {}
        browser_settings.use(features.table(self._config, "browser"))
        return {
            "engine": engine,
            "execution_browser": computer_config.browser(),
            "research_calls": computer_config.research_budget(features.table(self._config, "research")),
        }

    def _scrub(self, text: str) -> str:
        scrub = getattr(self._config, "scrub", None)
        return scrub(text) if callable(scrub) else text


class TaskRunner:
    """Starts tasks, one at a time, each in a run folder of its own under `runs_dir`."""

    def __init__(self, config, runs_dir: Path = DEFAULT_RUNS_DIR) -> None:
        self._config = config
        self._runs_dir = Path(runs_dir)
        self.current: ComputerTask | None = None

    def start(
        self,
        goal: str,
        *,
        act: bool = False,
        route: str = "",
        on_done: Callable[[ComputerTask], None] | None = None,
        on_question: Callable[[ComputerTask, str], None] | None = None,
    ) -> ComputerTask:
        """Begin a task. Raises `TaskBusy` when one is running (here or in any other assistant of this process)."""
        if not _ACTIVE.acquire(blocking=False):
            raise TaskBusy("a task is already running")
        try:
            task = ComputerTask(
                goal, act=act, config=self._config, folder=self._fresh_folder(), on_question=on_question, route=route
            )
            self.current = task
            task.start(on_done, release=_ACTIVE.release)
        except BaseException:
            _ACTIVE.release()
            raise
        return task

    def stop(self) -> bool:
        """Stop the running task, if any. True if there was one."""
        task = self.current
        if task is not None and task.running:
            task.stop()
            return True
        return False

    @property
    def running(self) -> bool:
        return self.current is not None and self.current.running

    def _fresh_folder(self) -> Path:
        """A run folder no other run has, made here: another Glide process may start in the same second, and `mkdir`
        is the one step that tells two of them apart."""
        base = self._runs_dir / time.strftime("%Y%m%d-%H%M%S")
        folder, n = base, 1
        while True:
            try:
                folder.mkdir(parents=True)
                return folder
            except FileExistsError:
                n += 1
                folder = base.with_name(f"{base.name}-{n}")

"""Running a computer task: the bridge from the assistant to the screen-driving loop (glide/computer/).

A task is `runner.run` on a worker thread, so the assistant stays free to hear "stop". Three promises:

- A dry run is the default at every layer. `act=True` has to be passed in by the caller, and it is checked
  against the macOS Accessibility permission first, as `glide-computer --act` does.
- A stop reaches the loop through the abort path it already has. Every step, every action and every wait of
  the loop calls `desktop.check_abort()`, which raises `Abort`; while a task runs that call is replaced by one
  that also raises when the task's stop event is set. After `stop()`, no further action can be taken, but the
  step already in flight (a classifier or writer request) finishes first, since a network call cannot be
  interrupted. The replacement is process-wide, so only one task runs at a time.
- The loop's own words are data. What the writer read off the screen is spoken and printed, never routed.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from .phrases import say

DEFAULT_RUNS_DIR = Path("runs")
STOPPED_BY_USER = "stopped by the user"
WOULD_DO_CHARS = 60

_ACTIVE = threading.Lock()  # held while a task runs: the abort hook is process-wide, so one task at a time
_NOTHING = object()


class TaskBusy(RuntimeError):
    """A task is already running in this process."""


@contextmanager
def abort_on(stop: threading.Event) -> Iterator[None]:
    """While the block runs, `desktop.check_abort()` also raises `Abort` once `stop` is set.

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

    def spoken(self, language: str | None = None) -> str:
        """What to say about the result: the writer's answer when there is one, else the outcome in a sentence."""
        if self.stopped:
            return ""
        if self.answer and self.outcome not in ("provider failure", "crashed"):
            return self.answer
        if self.outcome == "dry run":
            return say("dry_run", language, what=self.would_do) if self.would_do else say("dry_run_plain", language)
        key = OUTCOME_PHRASES.get(self.outcome, "crashed")
        return say(key, language)

    def summary(self) -> str:
        """One or two lines for a terminal: what happened, why, and where the run folder is."""
        lines = [f"task {self.outcome}: {self.goal}"]
        if self.answer:
            lines.append(self.answer)
        if self.would_do:
            lines.append(f"would do: {self.would_do}")
        if self.failure:
            lines.append(self.failure)
        if self.folder is not None:
            lines.append(f"run folder: {self.folder}")
        return "\n".join(lines)


OUTCOME_PHRASES = {
    "done": "done",
    "nothing helps": "nothing",
    "low confidence": "unsure",
    "stalled": "stalled",
    "stuck": "stalled",
    "step limit": "limit",
    "provider failure": "provider",
    "crashed": "crashed",
    "not permitted": "no_permission",
    "not configured": "not_configured",
}


def would_do(folder: Path) -> str | None:
    """The first move of a dry run, in words, read back from the run folder's first step. None if unreadable."""
    try:
        first = sorted(folder.glob("step-*-answers.json"))[0]
        data = json.loads(first.read_text(encoding="utf-8"))
        chosen = str(data["chosen"])
        if chosen.isdigit():
            text = next((str(it["text"]) for it in data.get("items", []) if str(it.get("index")) == chosen), "")
            return f"click {text[:WOULD_DO_CHARS]!r}" if text else "click an item"
        if chosen.startswith("offscreen:"):
            return "press an off-screen control"
        return chosen.replace("_", " ")
    except (IndexError, OSError, ValueError, KeyError, TypeError):
        return None


class ComputerTask:
    """One run of the screen-driving loop on a worker thread. Start it once; stop it any time."""

    def __init__(self, goal: str, *, act: bool, config, folder: Path) -> None:
        self.goal = goal
        self.act = act
        self.folder = folder
        self.stop_event = threading.Event()
        self.result: TaskResult | None = None
        self._config = config
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

    def start(self, on_done: Callable[[ComputerTask], None] | None = None, release: Callable[[], None] | None = None) -> None:
        self._thread = threading.Thread(target=self._work, args=(on_done, release), name="glide-task", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Ask the loop to stop. It does so at its next check, which is before the next action."""
        self.stop_event.set()

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
                result = TaskResult(
                    self.goal, self.act, "crashed", failure=self._scrub(f"{type(exc).__name__}: {exc}"), folder=self.folder
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
            writer = self._config.writer()
            cfg = runner.RunConfig(goal=self.goal, out=self.folder, act=self.act)

            def ctx_factory(typesafe, history):
                # ask=None: nobody can be asked a question in the middle of a run. An input() here would race the
                # terminal the assistant is reading, and a voice user has no keyboard to answer on.
                return Context(
                    goal=self.goal,
                    browser=computer_config.browser(),
                    email=computer_config.email(),
                    typesafe=typesafe,
                    writer=writer,
                    history=history,
                    ask=None,
                )

            with abort_on(self.stop_event):
                state = runner.run(cfg, ctx_factory, classifier_factory=self._config.classifier)
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
            would_do=would_do(self.folder) if state.outcome == "dry run" else None,
            stopped=state.outcome.startswith("aborted"),
        )

    def _scrub(self, text: str) -> str:
        scrub = getattr(self._config, "scrub", None)
        return scrub(text) if callable(scrub) else text


class TaskRunner:
    """Starts tasks, one at a time, each in a run folder of its own under `runs_dir`."""

    def __init__(self, config, runs_dir: Path = DEFAULT_RUNS_DIR) -> None:
        self._config = config
        self._runs_dir = Path(runs_dir)
        self.current: ComputerTask | None = None

    def start(self, goal: str, *, act: bool = False, on_done: Callable[[ComputerTask], None] | None = None) -> ComputerTask:
        """Begin a task. Raises `TaskBusy` when one is running (here or in any other assistant of this process)."""
        if not _ACTIVE.acquire(blocking=False):
            raise TaskBusy("a task is already running")
        try:
            task = ComputerTask(goal, act=act, config=self._config, folder=self._fresh_folder())
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
        base = self._runs_dir / time.strftime("%Y%m%d-%H%M%S")
        folder, n = base, 1
        while folder.exists():
            n += 1
            folder = base.with_name(f"{base.name}-{n}")
        return folder

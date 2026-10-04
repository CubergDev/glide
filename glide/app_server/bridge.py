"""Between the app protocol and the assistant: commands in, state and events out.

The bridge is the server's `Backend`. It turns the app's commands into calls on the assistant (`handle_text`, `stop`,
`interrupt_speech`) and turns what the assistant and the provider chains already report into messages, without adding
a thing to the assistant beyond one hook (`IO.approve`):

- transcripts from `IO.partial`, `IO.heard` and `IO.show`;
- provider switches from `config.on_switch`: every `SwitchEvent`, as it happens (a switch while no app is connected is
  still printed by the command line's own listener);
- task phases from the events a running `ComputerTask` collects (`task.events`), read by a small thread;
- state and speech phases from the same thread, from what the assistant and its speaker say they are doing;
- approvals from `IO.approve`, which the assistant asks before ANY computer task starts, a dry run too, since a dry run
  looks at the screen. The answer is the app's explicit approve, or no (see `server.Approvals`).

What is sent without recording content on: state; speech phases; task phases whose `summary` is the runner's own one-word
outcome; switches (provider, slot and a bounded, scrubbed reason); and transcripts that carry no words, only that one was
said and how many characters it had (`redacted: true`, `chars`). The one thing that carries words regardless is the approval
request, whose `command` has to say what the person is being asked to allow: it is shown, never logged and never stored.

Everything that is not the core's own text goes through `wire.clean_text` before it is sent.
"""

from __future__ import annotations

import contextlib
import importlib.metadata
import itertools
import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..assistant.core import IO, Assistant
from ..assistant.tasks import DEFAULT_RUNS_DIR
from ..computer.control import TaskEvent
from . import wire
from .runtime import Runtime, VoiceFactory
from .server import AppServer, Session
from .settings import SettingsPanel, SettingsState, VoiceUnavailable

log = logging.getLogger("glide.app_server")

MAX_TEXT_REQUESTS = 4  # typed requests being answered at once; the assistant itself supersedes the older ones
PARTIAL_INTERVAL_S = 0.2  # partial transcripts are sent at most this often
THINKING_S = 20.0  # a heard utterance counts as being thought about this long if nothing else shows it is
PUMP_S = 0.1
WATCH_S = 1.5  # after a sentence is shown, look for its speech this long: it is queued just after it is shown

GOAL_PREFIX_ACT = "Control this Mac (clicks and typing) to: "
GOAL_PREFIX_LOOK = "Look at this Mac's screen, without clicking or typing, to: "


def core_version() -> str:
    try:
        return importlib.metadata.version("glide")
    except importlib.metadata.PackageNotFoundError:
        return "0"


class _TaskTrace:
    """Where reading one task's events has got to, and what has been said about it."""

    def __init__(self, task: Any) -> None:
        self.task = task
        self.cursor = 0
        self.attempts = 0


class AppBridge:
    """The `Backend` of an `AppServer` over a loaded configuration. Make it, make the server from it, then `bind` the server."""

    def __init__(
        self,
        config: Any,
        *,
        record_content: bool = False,
        runs_dir: Path = DEFAULT_RUNS_DIR,
        voice_factory: VoiceFactory | None = None,
        assistant_factory: Callable[..., Any] = Assistant,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self._clock = clock
        self._lock = threading.RLock()
        self._server: AppServer | None = None
        self._counter = itertools.count(1)
        self._user_utterance: str | None = None  # the utterance being said (partials replace each other under one id)
        self._assistant_utterance: str | None = None
        self._assistant_text = ""
        self._last_partial = 0.0
        self._thinking_until = 0.0
        self._requests = threading.BoundedSemaphore(MAX_TEXT_REQUESTS)
        self._running_requests = 0
        self._speaking = False
        self._watch_until = 0.0
        self._speech_id: str | None = None
        self._interrupted = False
        self._last_state: tuple | None = None
        self._traces: list[_TaskTrace] = []
        self._stop = threading.Event()
        self._pump: threading.Thread | None = None
        self._unsubscribe: Callable[[], None] | None = None
        self.runtime = Runtime(
            config, self._make_io, runs_dir=runs_dir, voice_factory=voice_factory, assistant_factory=assistant_factory
        )
        self.panel = SettingsPanel(
            config,
            SettingsState.from_config(config, record_content=record_content),
            apply_voice=self.runtime.apply_voice,
            set_record_content=self._set_record_content,
        )
        if record_content:
            self._set_record_content(True)

    # -- wiring -------------------------------------------------------------------------------------

    def _make_io(self) -> IO:
        return IO(show=self._show, partial=self._partial, heard=self._heard, warn=self._warn, approve=self._approve)

    def bind(self, server: AppServer) -> None:
        """Connect to the server and start reading what the assistant is doing."""
        self._server = server
        self._unsubscribe = self.config.on_switch(self._on_switch)  # before any chain is built: no switch is missed
        self._pump = threading.Thread(target=self._run_pump, name="glide-app-pump", daemon=True)
        self._pump.start()

    def close(self) -> None:
        self._stop.set()
        if self._unsubscribe is not None:
            self._unsubscribe()
        if self._pump is not None and self._pump is not threading.current_thread():
            self._pump.join(timeout=2.0)
        self.runtime.close()

    def _emit(self, message: wire.Outgoing) -> None:
        server = self._server
        if server is not None:
            server.emit(message)

    @property
    def _recording(self) -> bool:
        return self.panel.state.record_content

    def _set_record_content(self, on: bool) -> None:
        """The one switch for storing content, here and in the run folders (`ComputerTask` reads it from the configuration)."""
        with contextlib.suppress(AttributeError):
            self.config.record_content = on
        if on:
            log.warning("detailed recording of content is ON for this session")

    # -- Backend ------------------------------------------------------------------------------------

    def capabilities(self) -> list[str]:
        caps = ["text_input", "approvals", "settings", "switch_events", "task_events"]
        if self.runtime.voice_available:
            caps.append("voice")
        return caps

    def recording_content(self) -> bool:
        return self._recording

    def on_ready(self, session: Session) -> None:
        self._publish_state(force=True)

    def on_closed(self, session: Session) -> None:
        """The app went away. Nothing it asked for is replayed when it returns; an open approval has already become a no."""
        with self._lock:
            self._last_state = None

    def handle(self, session: Session, command: wire.Command) -> None:
        match command.type:
            case "text_input":
                self._text_input(session, command)
            case "interrupt":
                self._interrupt()
            case "stop":
                self._stop_task(session, command)
            case "voice_control":
                self.runtime.mute(command.data["action"] == "mute")
                self._publish_state()
            case "settings_get":
                revision, payload = self.panel.snapshot()
                session.send(wire.settings(revision, payload, reply_to=command.id))
            case "settings_set":
                ok, revision, errors = self.panel.apply(command.data["base_revision"], command.data["changes"])
                session.send(wire.settings_result(ok, revision, errors, reply_to=command.id))
                self._publish_state()

    # -- commands -----------------------------------------------------------------------------------

    def _text_input(self, session: Session, command: wire.Command) -> None:
        text = command.data["text"]
        assistant = self.runtime.assistant
        if assistant.pending_question is not None:  # a task asked something: this line is the answer, and only a typed line is
            self._say_user(text)
            assistant.answer_pending(text)
            return
        if not self._requests.acquire(blocking=False):
            session.send(wire.error("busy", "too many requests are being answered", reply_to=command.id))
            return
        threading.Thread(target=self._run_text, args=(assistant, text, command.id), name="glide-app-request", daemon=True).start()

    def _run_text(self, assistant: Any, text: str, request_id: str | None) -> None:
        with self._lock:
            self._running_requests += 1
        try:
            self._say_user(text)
            assistant.handle_text(text, act=self.panel.state.act_enabled, wait=False)
        except Exception as exc:  # the type only: what was asked is not repeated
            log.exception("a typed request failed")
            self._emit(wire.error("request_failed", f"the request failed ({type(exc).__name__})", reply_to=request_id))
        finally:
            with self._lock:
                self._running_requests -= 1
            self._requests.release()

    def _interrupt(self) -> None:
        if self._speaking:
            self._interrupted = True
        self.runtime.interrupt_speech()

    def _stop_task(self, session: Session, command: wire.Command) -> None:
        wanted = command.data.get("task_id")
        if wanted is not None:
            current = [a.task for a in self.runtime.assistants() if a.task is not None and a.task.running]
            if not any(getattr(t.control, "task_id", None) == wanted for t in current):
                session.send(wire.error("unknown_task", "no running task has that id", reply_to=command.id))
                return
        if self._speaking:
            self._interrupted = True
        self.runtime.stop()

    # -- what the assistant says --------------------------------------------------------------------

    def _utterance(self, prefix: str) -> str:
        return f"{prefix}{next(self._counter)}"

    def _partial(self, text: str) -> None:
        now = self._clock()
        with self._lock:
            if self._user_utterance is None:
                self._user_utterance = self._utterance("u")
            if now - self._last_partial < PARTIAL_INTERVAL_S:
                return
            self._last_partial = now
            ident = self._user_utterance
        self._emit(wire.transcript(ident, "user", text, partial=True, record_content=self._recording))

    def _heard(self, text: str) -> None:
        with self._lock:
            ident = self._user_utterance or self._utterance("u")
            self._user_utterance = None
            self._begin_answer()
            self._thinking_until = self._clock() + THINKING_S
        self._emit(wire.transcript(ident, "user", text, partial=False, record_content=self._recording))

    def _say_user(self, text: str) -> None:
        """A typed line is a heard utterance that needed no hearing."""
        self._heard(text)

    def _begin_answer(self) -> None:
        self._assistant_utterance = self._utterance("a")
        self._assistant_text = ""

    def _show(self, sentence: str) -> None:
        with self._lock:
            if self._assistant_utterance is None:
                self._begin_answer()
            self._assistant_text = f"{self._assistant_text} {sentence}".strip()
            ident, text = self._assistant_utterance, self._assistant_text
            self._thinking_until = 0.0
            self._watch_until = self._clock() + WATCH_S
        self._emit(wire.transcript(ident or "a0", "assistant", self._scrub(text), partial=False, record_content=self._recording))

    def _warn(self, message: str) -> None:
        with self._lock:
            self._thinking_until = 0.0
        self._emit(wire.error("warning", self._scrub(message)))

    def _scrub(self, text: str) -> str:
        scrub = getattr(self.config, "scrub", None)
        return scrub(text) if callable(scrub) else text

    def _on_switch(self, event: Any) -> None:
        self._emit(
            wire.switch(
                role=event.role,
                from_slot=event.from_slot,
                to_slot=event.to_slot,
                kind=event.kind,
                reason=self._scrub(str(event.reason)),
            )
        )

    # -- approvals ----------------------------------------------------------------------------------

    def _approve(self, goal: str, act: bool) -> bool:
        """The assistant is about to start a computer task. Ask the app, and take nothing but an explicit yes for one."""
        server = self._server
        if server is None:
            return False
        with self._lock:
            self._thinking_until = 0.0
        command = (GOAL_PREFIX_ACT if act else GOAL_PREFIX_LOOK) + goal
        if len(command) > wire.MAX_COMMAND_CHARS:  # the person approves what they can read: never a cut-off of what runs
            self._warn("That request is too long to approve in full, so it was not run. Say it shorter.")
            return False
        return server.approvals.ask("input" if act else "screen", command)

    # -- reading the assistant ----------------------------------------------------------------------

    def _run_pump(self) -> None:
        while not self._stop.wait(PUMP_S):
            try:
                self._poll_tasks()
                self._poll_speech()
                self._publish_state()
            except Exception:
                log.exception("reading the assistant's state failed")

    def _poll_tasks(self) -> None:
        for assistant in self.runtime.assistants():
            task = assistant.task
            if task is not None and all(t.task is not task for t in self._traces):
                self._traces.append(_TaskTrace(task))
                with self._lock:
                    self._thinking_until = 0.0
        for trace in list(self._traces):
            events = trace.task.events
            fresh = events[trace.cursor :]
            trace.cursor += len(fresh)
            for event in fresh:
                for message in self._task_messages(trace, event):
                    self._emit(message)
            if not fresh and trace.task.wait(0) and trace.cursor >= len(trace.task.events):
                self._traces.remove(trace)

    def _task_messages(self, trace: _TaskTrace, event: TaskEvent) -> list[wire.Outgoing]:
        """The protocol's task phases for one event of the run. Only what the run reported is claimed: an action was sent
        (`attempted`), a fresh observation followed (`step`), and the end says what the final reading found. Nothing here
        decides that an action had its effect."""
        tid = event.task_id or "task"
        recording = self._recording
        summary = clean_for(event.text) if recording and event.text else None
        outcome = event.outcome or None
        match event.kind:
            case "accepted":
                return [wire.task(tid, "started")]
            case "progress":
                trace.attempts += 1
                return [wire.task(tid, "attempted", step=trace.attempts)]
            case "action_checked":
                return [wire.task(tid, "step", step=trace.attempts)]
            case "completed":
                return [
                    wire.task(tid, "verified", step=trace.attempts, verified=True),
                    wire.task(tid, "completed", summary=summary or outcome),
                ]
            case "dry_run":
                return [wire.task(tid, "completed", summary=summary or "dry run: nothing was done")]
            case "blocked":
                out = []
                if getattr(trace.task.control, "in_flight", False):
                    out.append(wire.task(tid, "reconcile_required"))  # a write whose outcome is unknown: nothing is replayed
                out.append(wire.task(tid, "failed", summary=summary or outcome))
                return out
            case "cancelled":
                out = []
                if getattr(trace.task.control, "in_flight", False):
                    out.append(wire.task(tid, "reconcile_required"))
                out.append(wire.task(tid, "stopped", summary=outcome))
                return out
        return []

    def _poll_speech(self) -> None:
        if not (self._speaking or self._clock() < self._watch_until):
            return
        speaking = any(not a.wait_idle(0) for a in self.runtime.assistants())
        if speaking and not self._speaking:
            self._speech_id = self._assistant_utterance
            self._interrupted = False
            self._speaking = True
            self._emit(wire.speech("started", self._speech_id))
        elif not speaking and self._speaking:
            self._speaking = False
            self._emit(wire.speech("interrupted" if self._interrupted else "finished", self._speech_id))
            self._interrupted = False

    def _assistant_state(self) -> tuple[str, str | None]:
        server = self._server
        assistants = self.runtime.assistants()
        if server is not None and server.approvals.waiting:
            return "awaiting_approval", None
        if any(a.pending_question is not None for a in assistants):
            return "asking", None
        if self._speaking:
            return "speaking", None
        if any(a.busy for a in assistants):
            return "acting", None
        with self._lock:
            thinking = self._running_requests > 0 or self._clock() < self._thinking_until
        if thinking:
            return "thinking", None
        failure = self.runtime.voice_failure
        if failure:
            return "error", "voice_failed"
        return ("listening" if self.runtime.hands_free and not self.runtime.muted else "idle"), None

    def _publish_state(self, *, force: bool = False) -> None:
        assistant, detail = self._assistant_state()
        key = (assistant, self.runtime.hands_free, self.runtime.muted, detail)
        with self._lock:
            if key == self._last_state and not force:
                return
            self._last_state = key
        self._emit(wire.state(assistant, hands_free=key[1], muted=key[2], detail=detail))


def clean_for(text: str) -> str:
    return wire.clean_text(text, wire.MAX_SUMMARY_CHARS)


__all__ = ["AppBridge", "VoiceUnavailable", "core_version"]

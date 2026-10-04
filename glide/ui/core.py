"""The pet's one adapter to the core: an in-process `Assistant`, and a plain event stream out. No Qt in this module.

Why in-process. variant-6 ran each request in a child process (`glide.cli_voice`, `glide.cli --control-stdin`),
sent it `stop`, `finish`, `resume` and JSON lines on stdin, and parsed the child's stdout for prefixes such as
"Listening." and "completed:". None of those entry points exist in this tree, the socket protocol of app/PROTOCOL.md
has no core-side server yet, and a third protocol would be one too many. So the pet calls the same objects the
command line does (`Assistant`, `build_voice`) and the window reads events from `PetCore.drain()`. The event names
follow PROTOCOL.md where it has one (`state`, `transcript`, `task`, `switch`), so a later swap to the socket is a
change in this file alone. The subprocess boundary is dropped: stopping is `Assistant.stop()`, which is safe from
any thread, and the pet never has stdout to scrape or persist.

What the pet stores: nothing. Typed text, transcripts and answers live in memory, bounded, and go to the window.
Whether the core's own run folders hold content is `config.record_content` (decision D3), off unless the pet is
started with it or the person ticks the box, and every view shows it while it is on.

Threads. Every callback of the assistant, the voice loop and the provider chains fires on a worker thread, so none
of them touches a widget: each puts a `PetEvent` on a queue, and the window drains it on a timer.
"""

from __future__ import annotations

import contextlib
import dataclasses
import queue
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from ..assistant.core import IO, Assistant, Reply
from ..assistant.point_ask import capture_point
from ..assistant.tasks import DEFAULT_RUNS_DIR
from ..providers.chain import SwitchEvent
from ..providers.config import ConfigError
from .point_core import POINT_DELAY_S, PointMode

MAX_TEXT = 8192  # the longest typed request taken
REASON_CHARS = 160  # how much of a switch's reason is shown
LINES = 80  # lines of activity kept in memory
ANSWER_CHARS = 4096  # the answer card
LABEL_CHARS = 1024
POINT_IDLE = "Choose Point & ask, then press Ask: the point you aim at stays pinned for follow-up questions."
PROVIDER_ROLES = ("llm.fast", "llm.smart", "stt", "tts", "classifier")


def task_phase(result) -> str:
    """How a finished task looks to the person. Only a run that said "done" (or a dry run, which is meant to end so)
    and was not found wanting is `completed`: every other outcome, those not yet invented included, is `failed`, and
    a write whose effect was never seen is `uncertain`, never a success."""
    if result.stopped:
        return "stopped"
    if result.uncertain:
        return "uncertain"
    return "completed" if result.outcome in ("done", "dry run") and result.achieved is not False else "failed"


@dataclass(frozen=True)
class PetEvent:
    """One thing the window should know. `type` and the `data` keys follow app/PROTOCOL.md where it has the message.

    - `state`: `assistant` is `idle`, `listening`, `thinking`, `acting`, `speaking` or `error`.
    - `transcript`: `role` (`user` or `assistant`), `text`, `partial`. For display only; never stored.
    - `task`: `phase` (`started`, `completed`, `failed`, `stopped`, `uncertain`), `outcome`, `act`.
    - `switch`: `role`, `from_slot`, `to_slot`, `kind`, `reason`. One per provider fallback; none is silent.
    - `mic`: `open`, and `detail` (a short machine reason, never user content).
    - `recording`: `on`. Whether the core keeps content (D3).
    - `point`: `kind` (`status`, `selecting`, `selected`, `preview`, `thinking`, `answer`, `error`, `stopped`, `closed`)
      and that kind's fields (point_core.py, point_session.py). Point and ask, read only.
    - `notice`: `message`, something the person should see that is not an answer.
    """

    type: str
    data: dict = field(default_factory=dict)


def _ignore(*_: object) -> None:
    return None


class PetAssistant(Assistant):
    """The assistant, reporting how each request came out as `state` and `task` events.

    `Assistant.handle_text` and `handle_audio` are the two doors a request comes through, so these are the only
    methods overridden, and each calls the original unchanged. A request is "in flight" from the door to the end of
    its readout; the pet is idle again when none is in flight and no task runs.
    """

    def __init__(self, config, *, report: Callable[[PetEvent], None] = _ignore, **options) -> None:
        super().__init__(config, **options)
        self._report = report
        self._inflight = 0
        self._count_lock = threading.Lock()

    @property
    def working(self) -> bool:
        """A request is being handled or read out, or a task runs."""
        return self._inflight > 0 or self.busy

    def handle_text(self, text: str, **options) -> Reply:
        self._report(PetEvent("state", {"assistant": "thinking"}))
        return self._through(super().handle_text, text, **options)

    def handle_audio(self, chunks, **options) -> Reply:
        return self._through(super().handle_audio, chunks, **options)

    def _through(self, handle, source, **options) -> Reply:
        with self._count_lock:
            self._inflight += 1
        try:
            reply = handle(source, **options)
            if reply.route == "computer" and reply.task is not None:
                self._report(PetEvent("task", {"phase": "started", "act": reply.task.act}))
                self._report(PetEvent("state", {"assistant": "acting"}))
                threading.Thread(target=self._await_task, args=(reply.task,), name="glide-pet-task", daemon=True).start()
            elif reply.route == "answer":
                self.wait_idle()  # the readout, if any, is part of the request
            return reply
        finally:
            with self._count_lock:
                self._inflight -= 1
            self._settle()

    def _await_task(self, task) -> None:
        task.wait()
        result = task.result
        if result is not None:
            self._report(PetEvent("task", {"phase": task_phase(result), "outcome": result.outcome, "act": result.act}))
        self._settle()

    def _settle(self) -> None:
        with self._count_lock:
            quiet = self._inflight == 0
        if quiet and not self.busy:
            self._report(PetEvent("state", {"assistant": "idle"}))


def _open_voice(config, settings, *, io, act, assistant_factory):
    """The one call that builds real audio hardware: tests replace it (tests/guards_pet-point.py refuses it)."""
    from ..speech.session import build_voice

    return build_voice(config, settings, io=io, act=act, assistant_factory=assistant_factory)


class PetCore:
    """Everything the window asks of the core. Methods are safe from any thread and return at once.

    `voice_factory(config, settings, io=, act=, assistant_factory=)` returns a `VoiceLoop`-shaped object; it is
    `speech.session.build_voice` unless a test passes a fake. `record_content` is opt-in and starts False.
    """

    def __init__(
        self,
        config,
        *,
        runs_dir: Path = DEFAULT_RUNS_DIR,
        record_content: bool = False,
        voice_factory=None,
        capture=None,
        point_delay_s: float = POINT_DELAY_S,
    ) -> None:
        self._config = config
        self._runs_dir = Path(runs_dir)
        self._voice_factory = voice_factory or _open_voice
        self._events: queue.SimpleQueue[PetEvent] = queue.SimpleQueue()
        self._lock = threading.RLock()
        self._text: PetAssistant | None = None
        self._voice = None
        self._opening = False
        self._voice_epoch = 0  # bumped by every stop: a session still being built under an older epoch is abandoned
        self._closed = False
        self.act = False  # computer tasks are dry runs until the person turns this on
        self.headset = False
        self.silence_ms = config.speech.silence_ms
        config.record_content = bool(record_content)  # D3: off unless asked for, whatever the config object held
        self._unsubscribe = config.on_switch(self._on_switch)
        self.point = PointMode(
            config,
            lambda kind, **data: self._emit("point", kind=kind, **data),
            start_voice=self._start_point_voice,
            end_voice=self._end_voice_now,
            capture=capture or capture_point,
            delay_s=point_delay_s,
        )
        if record_content:
            self._emit("recording", on=True)

    # -- settings --------------------------------------------------------------------------------

    @property
    def record_content(self) -> bool:
        return bool(getattr(self._config, "record_content", False))

    @record_content.setter
    def record_content(self, on: bool) -> None:
        """The core reads this when a task starts (assistant/tasks.py). Content is kept only while it is True."""
        self._config.record_content = bool(on)
        self._emit("recording", on=bool(on))

    @property
    def busy(self) -> bool:
        return any(getattr(a, "working", a.busy) for a in self._assistants())

    @property
    def voice_active(self) -> bool:
        return self._voice is not None or self._opening

    def drain(self) -> list[PetEvent]:
        """Every event since the last call, oldest first. For the window's timer."""
        out = []
        while True:
            try:
                out.append(self._events.get_nowait())
            except queue.Empty:
                return out

    # -- requests --------------------------------------------------------------------------------

    def send_text(self, text: str) -> bool:
        """Handle a typed request on a worker thread. False when there is nothing to send or it is too long."""
        text = text.strip()
        if not text or len(text) > MAX_TEXT or self._closed:
            return False
        assistant = self._assistant()
        self._emit("transcript", role="user", text=text, partial=False)
        threading.Thread(target=self._handle_text, args=(assistant, text, self.act), name="glide-pet-turn", daemon=True).start()
        return True

    def _handle_text(self, assistant: PetAssistant, text: str, act: bool) -> None:
        try:
            reply = assistant.handle_text(text, act=act, wait=False)
        except Exception as exc:  # a bug must not leave the pet thinking forever
            self._emit("notice", message=f"{type(exc).__name__}: {self._scrub(str(exc))}")
            self._emit("state", assistant="error")
            return
        if reply.error:
            self._emit("state", assistant="error", detail="provider")

    def start_voice(self) -> None:
        """Open the microphone: build the voice stack the first time, or resume it after `pause_voice`."""
        with self._lock:
            if self._voice is not None and not self._closed:
                self._voice.resume()
                self._emit("mic", open=True)
                return
        epoch = self._reserve_voice()
        if epoch is not None:
            threading.Thread(
                target=self._run_voice, args=(self._make_assistant, None, self.act, epoch), name="glide-pet-voice", daemon=True
            ).start()

    def _reserve_voice(self) -> int | None:
        """Claim the single voice slot: the epoch to build under, or None when a session is open or opening, or closed."""
        with self._lock:
            if self._closed or self._opening or self._voice is not None:
                return None
            self._opening = True
            return self._voice_epoch

    def _abandoned(self, epoch: int) -> bool:
        with self._lock:
            return self._closed or epoch != self._voice_epoch

    def _run_voice(self, assistant_factory, bind, act: bool, epoch: int) -> bool:
        """Build and start a voice session in the slot `_reserve_voice` gave. `bind(assistant)` runs before listening.

        A stop that arrives while this is still building wins: nothing is started, a loop already built is ended, and
        no `mic` event says the microphone is open. (Opening a stream is what asks macOS for the microphone.)"""
        loop = None
        try:
            if self._abandoned(epoch):
                return False
            settings = dataclasses.replace(self._config.speech, headset=self.headset, silence_ms=self.silence_ms)
            loop = self._voice_factory(self._config, settings, io=self._new_io(), act=act, assistant_factory=assistant_factory)
            if bind is not None:
                bind(loop.assistant)
            with self._lock:
                registered = not self._abandoned(epoch)
                if registered:
                    loop.start()  # under the lock: a stop cannot fall between the check and the first frame
                    self._voice = loop
        except Exception as exc:  # AudioUnavailable, ConfigError, VadError: each says what is missing and never a key
            if loop is not None:
                self._end_quietly(loop)
            if not self._abandoned(epoch):
                self._emit("mic", open=False, detail="unavailable")
                self._emit("notice", message=f"Voice input could not start ({type(exc).__name__}): {self._scrub(str(exc))}")
            return False
        finally:
            with self._lock:
                self._opening = False
        if not registered:
            self._end_quietly(loop)
            return False
        self._emit("mic", open=True)
        return True

    def _end_quietly(self, loop) -> None:
        with contextlib.suppress(Exception):
            self._end_voice(loop)

    def _start_point_voice(self, assistant_factory, bind) -> bool:
        """For `PointMode`, on its worker thread: a voice session whose assistant answers about the pin."""
        epoch = self._reserve_voice()
        if epoch is None:
            self._emit("notice", message="Stop the current voice session before asking about a point by voice.")
            return False
        return self._run_voice(assistant_factory, bind, False, epoch)  # reading a point never acts

    def pause_voice(self) -> None:
        """Finish what is being said, then turn the microphone off. Answers, speech and tasks go on."""
        with self._lock:
            voice, opening = self._voice, self._opening
            if voice is None and opening:
                self._voice_epoch += 1  # nothing was being said yet: the session still being built is abandoned
        if voice is not None:
            voice.pause()
        if voice is not None or opening:
            self._emit("mic", open=False, detail="paused")

    def stop(self) -> None:
        """Stop the answer, the speech and the task, end the voice session and any pin. Returns at once."""
        self.point.close()
        for assistant in self._assistants(self._pop_voice()):
            assistant.stop()
        self._emit("state", assistant="idle")

    def close(self) -> None:
        """The window is closing: stop everything and release the microphone, speaker and provider clients."""
        with self._lock:
            if self._closed:
                return  # closing twice (the window's closeEvent, then a caller's cleanup) is harmless
            self._closed = True
            self._voice_epoch += 1
            voice, self._voice = self._voice, None
            text, self._text = self._text, None
        self._unsubscribe()
        self.point.close()
        if voice is not None:
            self._end_voice(voice)
        if text is not None:
            text.close()

    def _pop_voice(self):
        """Take the open voice session, if any, and end it on a thread of its own so the caller does not wait."""
        with self._lock:
            self._voice_epoch += 1  # also abandons a session still being built (see `_run_voice`)
            voice, self._voice = self._voice, None
        if voice is not None:
            threading.Thread(target=self._end_voice, args=(voice,), name="glide-pet-stop", daemon=True).start()
            self._emit("mic", open=False, detail="stopped")
        return voice

    def _end_voice_now(self) -> None:
        self._pop_voice()

    def _end_voice(self, voice) -> None:
        try:
            voice.stop()
        finally:
            voice.assistant.close()

    # -- plumbing --------------------------------------------------------------------------------

    def _assistant(self) -> PetAssistant:
        """Where a typed request goes: the voice session's assistant while one runs, else the text one."""
        with self._lock:
            if self._voice is not None:
                return self._voice.assistant
            if self._text is None:
                self._text = self._make_assistant(self._config, io=self._new_io())
            return self._text

    def _assistants(self, voice=None) -> list[PetAssistant]:
        with self._lock:
            voice = voice or self._voice
            return [a for a in (self._text, voice.assistant if voice is not None else None) if a is not None]

    def _make_assistant(self, config, io=None, **options) -> PetAssistant:
        """What `build_voice` calls to make the assistant: `options` are whatever it passes on (settings of the stack)."""
        return PetAssistant(config, io=io or self._new_io(), runs_dir=self._runs_dir, report=self._report, **options)

    def _new_io(self) -> IO:
        return IO(
            show=self._on_show,
            partial=self._on_partial,
            heard=self._on_heard,
            warn=lambda message: self._emit("notice", message=self._scrub(message)),
        )

    def _on_show(self, text: str) -> None:
        self._emit("transcript", role="assistant", text=text, partial=False)
        if self._voice is not None:
            self._emit("state", assistant="speaking")

    def _on_partial(self, text: str) -> None:
        self._emit("transcript", role="user", text=text, partial=True)
        self._emit("state", assistant="listening")

    def _on_heard(self, text: str) -> None:
        self._emit("transcript", role="user", text=text, partial=False)
        self._emit("state", assistant="thinking")

    def _on_switch(self, event: SwitchEvent) -> None:
        reason = " ".join(self._scrub(str(event.reason)).split())
        if len(reason) > REASON_CHARS:
            reason = reason[: REASON_CHARS - 3] + "..."
        self._emit("switch", role=event.role, from_slot=event.from_slot, to_slot=event.to_slot, kind=event.kind, reason=reason)

    def _report(self, event: PetEvent) -> None:
        self._events.put(event)

    def _emit(self, type_: str, **data) -> None:
        self._events.put(PetEvent(type_, data))

    def _scrub(self, text: str) -> str:
        scrub = getattr(self._config, "scrub", None)
        return scrub(text) if callable(scrub) else text


def provider_lines(config) -> list[str]:
    """One line per role naming its slots (as glide.toml names them) and any that are left out, for the settings view.

    Slot names only: no key, no URL, no model id typed here. A role that cannot be set up says so instead of raising.
    """
    lines = []
    for role in PROVIDER_ROLES:
        try:
            infos = config.slots(role)
        except ConfigError as exc:
            lines.append(f"{role}: {' '.join(str(exc).split())[:REASON_CHARS]}")
            continue
        lines.append(f"{role}: " + ", ".join(f"{i.name}" + ("" if i.state == "ready" else f" ({i.state})") for i in infos))
    return lines


class PetView:
    """What the window shows, as a function of the events so far. Pure: the widgets only copy its fields."""

    def __init__(self) -> None:
        self.status = "Idle · microphone off"
        self.mood = "idle"
        self.mic = False
        self.recording = False
        self.working = False
        self.lines: deque[str] = deque(maxlen=LINES)
        self._result: tuple[str, str] | None = None  # how the last task ended, kept until something else happens
        # Point and ask: the pinned item, the answer card and where the pin is drawn. All plain text, in memory only.
        self.pin: tuple[float, float] | None = None
        self.target = ""
        self.answer = POINT_IDLE
        self.answer_note = ""
        self.provider = ""

    def apply(self, event: PetEvent) -> None:
        handler = getattr(self, "_" + event.type, None)
        if handler is not None:
            handler(**event.data)

    def _state(self, assistant: str, detail: str = "") -> None:
        self.working = assistant in ("thinking", "acting", "speaking")
        if assistant == "idle":
            if self._result is not None:
                self.mood, self.status = self._result
            else:
                self.mood = "listening" if self.mic else "idle"
                self.status = "Listening · say your command" if self.mic else "Idle · microphone off"
            return
        self._result = None
        if assistant == "listening":
            self.mood, self.status = "listening", "Hearing you"
        elif assistant == "thinking":
            self.mood, self.status = "thinking", "Working on it"
        elif assistant == "acting":
            self.mood, self.status = "thinking", "Working · computer task running"
        elif assistant == "speaking":
            self.mood, self.status = "talking", "Speaking" if not self.mic else "Speaking · microphone paused"
        else:
            self.mood, self.status = "sad", "Something went wrong" + (f" ({detail})" if detail else "")

    def _transcript(self, role: str, text: str, partial: bool) -> None:
        if partial:
            return
        self.lines.append(("you: " if role == "user" else "glide: ") + text[:500])

    def _task(self, phase: str, outcome: str = "", act: bool = False) -> None:
        if phase == "started":
            self.lines.append("task started" + ("" if act else " (dry run: nothing will be clicked)"))
            return
        self.lines.append(f"task {phase}" + (f": {outcome}" if outcome else ""))
        mood = {"completed": "happy", "failed": "sad", "uncertain": "question"}.get(phase, "idle")
        status = {"stopped": "Stopped", "uncertain": "Check the screen · nothing will be repeated"}.get(
            phase, f"Result · {outcome or phase}"
        )
        self._result = (mood, status)
        self.mood, self.status = self._result

    def _switch(self, role: str, from_slot: str, to_slot: str | None, kind: str, reason: str) -> None:
        target = to_slot or "nothing left"
        self.lines.append(f"fallback: {role} {from_slot} -> {target} ({kind}: {reason})")

    def _mic(self, open: bool, detail: str = "") -> None:
        self.mic = open
        self._result = None
        if open:
            self.mood, self.status = "listening", "Listening · say your command"
        else:
            self.mood = "idle"
            self.status = "Microphone off · task and readout remain active" if detail == "paused" else "Idle · microphone off"

    def _point(self, kind: str, **data) -> None:
        text = data.get("text", "")
        if not isinstance(text, str):
            return
        if kind == "selecting":
            self.pin, self.target, self.answer_note = None, "", ""
            self.answer, self.status, self.mood = "Move the pointer to your item during the countdown.", text[:500], "thinking"
        elif kind == "selected":
            point = data.get("point")
            if isinstance(point, list) and len(point) == 2:
                self.pin = (point[0], point[1])
            self.target = "Pinned: " + text[:LABEL_CHARS]
            self.status = "Point pinned · read only"
        elif kind in ("answer", "preview"):
            uncertain = bool(data.get("uncertain"))
            self.answer = text[:ANSWER_CHARS]
            age = data.get("age_s")
            self.answer_note = (
                f"From a snapshot taken {age:.0f} s ago; the screen may have changed. Point again to refresh."
                if isinstance(age, (int, float))
                else ""
            )
            self.status = "More context needed · ask a follow-up" if uncertain else "Answer ready"
            self.mood = "question" if uncertain else "happy"
            if kind == "answer":
                self.lines.append(f"point answer: {str(data.get('model', ''))[:200]} | {float(data.get('seconds', 0)):.3f}s")
        elif kind in ("thinking", "status", "stopped"):
            if kind == "status" and text.startswith("Answer provider:"):
                self.provider = text[:1000]
            else:
                self.status = text[:500]
            self.mood = "thinking" if kind == "thinking" else self.mood
        elif kind == "error":
            self.status, self.answer, self.mood = text[:500], text[:ANSWER_CHARS], "sad"
            self.lines.append(text[:500])
            if data.get("closed", True):
                self.pin, self.target = None, ""
        elif kind == "closed":
            self.pin, self.target = None, ""

    def _recording(self, on: bool) -> None:
        self.recording = on
        self.lines.append("recording content: ON" if on else "recording content: off")

    def _notice(self, message: str) -> None:
        self.lines.append(message[:500])
        self.status = message[:200]

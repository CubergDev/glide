"""The Chat/Ask tab's engine: one request at a time through the real `Assistant`, the same object `glide chat` builds.

A computer task needs the person's yes before it starts (the assistant's own `IO.approve` hook, so even a dry run, which
looks at the screen, waits for a click). A task is a dry run unless the request was sent with `act` and confirmed, which
is `--act`. Stop reaches the assistant's own stop, which reaches the task's `RunControl`. While a real run is in flight a
marker file is on disk; it is removed when the run ends with an observed result, and left in place when the process died
or the run ended with a write whose effect was never seen, so the Status tab can say so.
"""

from __future__ import annotations

import contextlib
import json
import threading
import time
from collections.abc import Callable
from pathlib import Path

from glide.setup import model as setup_model

from .settings import PanelSettings

PanelError = setup_model.SetupError
MAX_TEXT = 2000
APPROVAL_WAIT_S = 120.0
MARKER = "active-run.json"
LOG = "chat-log.jsonl"
UNCERTAIN = "completion unknown; nothing was retried"


def real_assistant(config, io, runs_dir):
    """The assistant `glide chat` uses. Tests replace this seam; it is the one that can reach the machine."""
    from glide.assistant.core import Assistant

    return Assistant(config, io=io, runs_dir=runs_dir)


class ChatSession:
    def __init__(
        self,
        *,
        load_config: Callable[[], object],
        settings: Callable[[], PanelSettings],
        runs_dir: Path,
        state_dir: Path,
        assistant_factory: Callable | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.load_config = load_config
        self.settings = settings
        self.runs_dir = runs_dir
        self.state_dir = state_dir
        self.factory = assistant_factory or real_assistant
        self.clock = clock
        self.lock = threading.RLock()
        self.config = None
        self.assistant = None
        self.turn: dict | None = None
        self.seq = 0

    # -- requests ----------------------------------------------------------------------------------------------

    def start(self, text: object, *, act: bool, engine: object, confirm_act: bool) -> dict:
        if not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT:
            raise PanelError(f"write a request of up to {MAX_TEXT} characters")
        if engine not in (None, "", "legacy", "structured"):
            raise PanelError("engine must be legacy or structured")
        if act and not self.settings().computer:
            raise PanelError("computer control is off: switch it on under Features and safety first")
        if act and confirm_act is not True:
            raise PanelError("a real run needs the confirm: Glide will click and type on this Mac")
        with self.lock:
            if self.turn is not None and self.turn["status"] == "running":
                raise PanelError("a request is already running: wait for it or press Stop")
            self.seq += 1
            turn = {
                "id": self.seq, "status": "running", "events": [], "approval": None, "result": None,
                "act": act, "gate": threading.Event(), "decision": False,
            }  # fmt: skip
            self.turn = turn
        threading.Thread(target=self._run, args=(turn, text, act, engine or None), daemon=True).start()
        return {"id": turn["id"]}

    def _event(self, turn: dict, kind: str, text: str) -> None:
        with self.lock:
            turn["events"].append({"kind": kind, "text": self._clean(text)})

    def _clean(self, text: str) -> str:
        from glide.cli import clean

        return clean(str(text), self.config)

    def _ensure(self, turn: dict, engine: str | None):
        from glide import features
        from glide.assistant.core import IO

        with self.lock:
            if self.assistant is None:
                config = self.load_config()
                self.config = config
                if hasattr(config, "on_switch"):
                    config.on_switch(lambda event: self._switch(turn, event))
                io = IO(
                    show=lambda t: self._event(self.turn, "say", t),
                    warn=lambda t: self._event(self.turn, "notice", t),
                    approve=self._approve,
                )
                self.assistant = self.factory(config, io, self.runs_dir)
            config = self.config
        try:
            features.engine_for(config, engine)
        except ValueError as error:
            raise PanelError(str(error)) from None
        config.engine_choice = engine

    def _switch(self, turn: dict, event) -> None:
        from glide.cli import format_switch

        self._event(self.turn or turn, "switch", format_switch(event, self.config))

    def _approve(self, goal: str, act: bool) -> bool:
        turn = self.turn
        if turn is None:
            return False
        if not self.settings().computer:
            self._event(turn, "notice", "computer control is off (Features and safety): nothing was started")
            return False
        with self.lock:
            turn["approval"] = {"goal": self._clean(goal), "act": bool(act)}
            turn["gate"].clear()
        ok = turn["gate"].wait(APPROVAL_WAIT_S) and turn["decision"] is True
        with self.lock:
            turn["approval"] = None
        if ok and act:
            self._marker_write()
        return ok

    def decide(self, approve: object, confirm_act: object) -> dict:
        with self.lock:
            turn = self.turn
            pending = turn["approval"] if turn else None
            if pending is None:
                raise PanelError("nothing is waiting for a decision")
            if approve is True and pending["act"] and confirm_act is not True:
                raise PanelError("a real run needs the confirm: Glide will click and type on this Mac")
            turn["decision"] = approve is True
            turn["gate"].set()
        return {"decided": approve is True}

    def _run(self, turn: dict, text: str, act: bool, engine: str | None) -> None:
        crashed, result = False, None
        try:
            self._ensure(turn, engine)
            reply = self.assistant.handle_text(text, act=act, wait=True)
            result = self._summary(reply)
        except PanelError as error:
            result = {"route": "none", "text": "", "error": str(error), "task": None, "chains": []}
        except Exception as error:  # the type only: a message could carry request text
            crashed = True
            result = {"route": "none", "text": "", "error": type(error).__name__, "task": None, "chains": []}
        task = (result or {}).get("task") or {}
        with self.lock:
            turn["result"] = result
            turn["status"] = "done"
        if turn["act"] and not crashed and not task.get("uncertain"):
            self._marker_clear()
        self._log(text, result)

    def _summary(self, reply) -> dict:
        task = getattr(reply, "task", None)
        done = getattr(task, "result", None) if task is not None else None
        info = None
        if done is not None:
            info = {
                "outcome": str(done.outcome), "act": bool(done.act), "uncertain": bool(done.uncertain),
                "readback": self._clean(done.readback), "steps": done.steps, "seconds": round(done.seconds, 1),
                "would_do": self._clean(done.would_do or ""), "failure": self._clean(done.failure or ""),
                "answer": self._clean(done.answer or ""), "stopped": bool(done.stopped),
                "can_run_for_real": done.outcome == "dry run" and not done.act,
                "uncertain_note": UNCERTAIN if done.uncertain else "",
            }  # fmt: skip
        return {
            "route": str(reply.route),
            "text": self._clean(reply.text or ""),
            "error": self._clean(reply.error or ""),
            "task": info,
            "chains": self._chains(),
        }

    def _chains(self) -> list[dict]:
        rows = []
        for role in ("llm.fast", "llm.smart"):
            try:
                for r in self.config.chain(role).status():
                    rows.append(
                        {"role": role, "slot": r["name"], "calls": r["calls"], "failures": r["failures"],
                         "resting_s": r["resting_s"], "last_error": self._clean(r["last_error"] or "")}
                    )  # fmt: skip
            except Exception:
                continue
        return rows

    def poll(self, since: int) -> dict:
        with self.lock:
            turn = self.turn
            if turn is None:
                return {"id": 0, "status": "idle", "events": [], "next": 0, "approval": None, "result": None}
            return {
                "id": turn["id"], "status": turn["status"], "events": turn["events"][since:],
                "next": len(turn["events"]), "approval": turn["approval"], "result": turn["result"],
            }  # fmt: skip

    def stop(self) -> dict:
        with self.lock:
            turn = self.turn
            if turn is not None and turn["approval"] is not None:
                turn["decision"] = False
                turn["gate"].set()
            assistant = self.assistant
        stopped = bool(assistant.stop()) if assistant is not None else False
        return {"stopped": stopped}

    def reset(self) -> None:
        """The config file changed: the next request builds the assistant again from it."""
        with self.lock:
            assistant, self.assistant, self.config = self.assistant, None, None
        if assistant is not None and not (self.turn and self.turn["status"] == "running"):
            assistant.close()

    # -- the crash-recovery marker -----------------------------------------------------------------------------

    def _marker_write(self) -> None:
        try:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            note = {"started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.clock())), "act": True}
            (self.state_dir / MARKER).write_text(json.dumps(note), encoding="utf-8")
        except OSError:
            pass

    def _marker_clear(self) -> None:
        with contextlib.suppress(OSError):
            (self.state_dir / MARKER).unlink(missing_ok=True)

    def marker(self) -> dict | None:
        path = self.state_dir / MARKER
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        running = bool(self.turn and self.turn["status"] == "running" and self.turn["act"])
        return {"path": str(path), "started": str(data.get("started", "unknown")), "running_now": running}

    def clear_marker(self, confirm: object) -> dict:
        if confirm is not True:
            raise PanelError("clearing needs an explicit confirm, after you have looked at the screen and the run folder")
        if (self.marker() or {}).get("running_now"):
            raise PanelError("a real run is in progress: stop it first")
        self._marker_clear()
        return {"cleared": True}

    # -- the optional local chat log (detailed recording) --------------------------------------------------------------

    def _log(self, text: str, result: dict | None) -> None:
        settings = self.settings()
        if not settings.record_content:
            return
        path = self.state_dir / LOG
        now = self.clock()
        keep = []
        try:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            if path.is_file():
                for line in path.read_text(encoding="utf-8").splitlines():
                    try:
                        if now - json.loads(line)["t"] <= settings.retention_days * 86400:
                            keep.append(line)
                    except (ValueError, KeyError, TypeError):
                        continue
            keep.append(json.dumps({"t": now, "request": self._clean(text), "reply": (result or {}).get("text", "")}))
            path.write_text("\n".join(keep) + "\n", encoding="utf-8")
        except OSError:
            pass

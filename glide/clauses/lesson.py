"""A short lesson, one step at a time. It holds the steps it was given and moves a pointer; it never calls a model and
never touches the screen. The caller presents whatever step comes back.

Control words are tolerant ("okay next step please") but the WHOLE utterance must be a control: "next to the printer"
is not one. Back stops at the first step, past the last ends the lesson, and with no lesson a control word gets a plain
"no_lesson" result while any other speech gets None (not ours, route it as usual).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Literal

from glide.routing.stop import normalize

Control = Literal["next", "back", "repeat", "dismiss"]
Kind = Literal["step", "done", "dismissed", "no_lesson"]

MIN_STEPS, MAX_STEPS = 2, 6

# Words that may surround a control without changing it.
_FILLER = frozenset(
    {"ok", "okay", "please", "glide", "now", "yes", "yeah", "thanks", "thank", "you", "and", "then", "the", "step", "one"}
)
_PHRASES: dict[Control, frozenset[str]] = {
    "next": frozenset({"next", "nex", "necks", "continue", "go on", "go next", "go", "onward", "keep going", "next up"}),
    "back": frozenset({"back", "go back", "previous", "last", "before", "back up", "go previous"}),
    "repeat": frozenset({"repeat", "again", "say again", "say that again", "repeat that", "repeat it", "once more", "more time"}),
    "dismiss": frozenset(
        {"dismiss", "done", "finished", "exit", "quit", "end", "close", "got it", "i got it", "that s all", "end lesson"}
        | {"exit lesson", "dismiss lesson", "i m done", "i am done"}
    ),
}
_TABLE = {normalize(p): control for control, phrases in _PHRASES.items() for p in phrases}


def parse_control(text: str) -> Control | None:
    """The control a whole utterance says, or None when it is anything else."""
    words = [w for w in normalize(text).split() if w not in _FILLER] or normalize(text).split()
    return _TABLE.get(" ".join(words))


@dataclass(frozen=True)
class LessonResult:
    kind: Kind
    index: int | None = None  # 0-based position of the step to present; None unless kind is "step"
    text: str = ""
    total: int = 0


NO_LESSON = LessonResult("no_lesson")


class LessonStepper:
    def __init__(self) -> None:
        self._steps: tuple[str, ...] = ()
        self._at = 0
        self._lock = threading.Lock()

    @property
    def active(self) -> bool:
        return bool(self._steps)

    def start(self, steps: list[str] | tuple[str, ...]) -> LessonResult:
        """Begin a lesson (replacing any running one) and return its first step. ValueError leaves the old lesson alone."""
        clean = tuple(s.strip() for s in steps if isinstance(s, str))
        if len(clean) != len(steps) or not MIN_STEPS <= len(clean) <= MAX_STEPS or not all(clean):
            raise ValueError(f"a lesson has {MIN_STEPS} to {MAX_STEPS} non-empty text steps")
        with self._lock:
            self._steps, self._at = clean, 0
            return self._present()

    def current(self) -> LessonResult:
        with self._lock:
            return self._present()

    def handle(self, text: str) -> LessonResult | None:
        """Apply a spoken or typed control word. None when `text` is not a control."""
        control = parse_control(text)
        if control is None:
            return None
        with self._lock:
            if not self._steps:
                return NO_LESSON
            if control == "dismiss":
                self._steps = ()
                return LessonResult("dismissed")
            if control == "next":
                if self._at + 1 >= len(self._steps):
                    self._steps = ()
                    return LessonResult("done")
                self._at += 1
            elif control == "back":
                self._at = max(0, self._at - 1)
            return self._present()

    def _present(self) -> LessonResult:
        if not self._steps:
            return NO_LESSON
        return LessonResult("step", self._at, self._steps[self._at], len(self._steps))

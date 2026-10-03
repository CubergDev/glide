"""Fakes for the point-to-ask tests: a synthetic desktop that only reads, a writer with a fixed reply, a hand-wound clock."""

from __future__ import annotations

from types import SimpleNamespace

from PIL import Image

from glide.computer.generation import GenerationResult
from glide.computer.point_types import PointTarget, point_box

DEFAULT_TARGET = PointTarget("AXStaticText", "錯誤 0007: connection failed")
GOOD = '{"answer":"This is error 0007. Check the connection.","uncertain":false}'


class SyntheticDesktop:
    """The three read-only calls point-to-ask makes. Any other call is a failure: it must never act."""

    def __init__(self, target=DEFAULT_TARGET, scale=1):
        self.target = target
        self.point = (200.0, 150.0)
        self.calls = []
        self.scale = scale
        self.after = target

    def mouse_location(self):
        self.calls.append(("pointer",))
        return self.point

    def point_target(self, point):
        self.calls.append(("target", point))
        return self.after if any(c[0] == "region" for c in self.calls) else self.target

    def point_region(self, point, radius):
        self.calls.append(("region", point, radius))
        box = point_box(point, radius, (0, 0, 1200, 800))
        self.raw = Image.new("RGB", (int((box[2] - box[0]) * self.scale), int((box[3] - box[1]) * self.scale)), "white")
        return self.raw, box

    def __getattr__(self, name):
        raise AssertionError(f"point-to-ask must not call {name}")


def reply_writer(text=GOOD, stop="end_turn", model="fixture"):
    return SimpleNamespace(
        generate=lambda request, cancel=None: GenerationResult(
            text, model, completed=stop in {"end_turn", "stop"}, stop_reason=stop
        )
    )


class Clock:
    """A clock that moves only when told to."""

    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds

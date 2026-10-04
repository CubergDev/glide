"""Read-only point observations and pure geometry shared by the desktop adapters and point-to-ask."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .models import Box


@dataclass(frozen=True)
class PointTarget:
    role: str
    label: str = ""
    value: str = ""
    help: str = ""
    protected: bool = False

    def packet(self) -> dict:
        if self.protected:
            raise ValueError("Protected fields cannot be read or described.")
        return {"role": self.role[:80], "label": self.label[:1024], "value": self.value[:2048], "help": self.help[:1024]}


def point_box(point: tuple[float, float], radius: float, display: Box) -> Box:
    """Clamp a small square to the main display; never substitute another point."""
    x, y = point
    left, top, right, bottom = display
    if not all(math.isfinite(v) for v in (*point, radius, *display)):
        raise ValueError("Point and display coordinates must be finite.")
    if not 24 <= radius <= 240 or not left <= x < right or not top <= y < bottom:
        raise ValueError("Point must be on the primary display; radius must be 24-240.")
    return (
        math.floor(max(left, x - radius)),
        math.floor(max(top, y - radius)),
        math.ceil(min(right, x + radius)),
        math.ceil(min(bottom, y + radius)),
    )

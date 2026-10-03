"""Pin one point of the screen, then answer about that observation. This module never acts.

`capture_point` reads one exact hit-tested item, and optionally a small crop around it, from the desktop adapter
(`point_target`, `point_region`): no clicking, moving, focusing, OCR, browser access or output file. What was read is
frozen in a `PointSelection`. Moving the pointer afterwards changes nothing, and the selection is thrown away when it
is cancelled, closed, or two minutes old (`HARD_TTL_S`, the longest a snapshot is trusted to describe the screen).
Everything captured is untrusted data: it is handed to the writer as data and nothing here reads an instruction out of
it. Nothing is written to disk, whatever the recording setting: a selection lives in memory until it is closed.

Time comes from an injected `clock`, so expiry and the age sent to the model are tested without sleeping.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from PIL import Image, ImageDraw

from ..computer.config import writer_vision
from ..computer.models import Box
from ..computer.platform_adapter import desktop
from ..computer.point_types import PointTarget
from ..computer.writer import Writer
from .point_answer import PointAnswer, compose_point_answer

HARD_TTL_S = 120.0  # a pinned item is never answered about after this long, however it is used
MAX_QUESTION = 2048
IMAGE_EDGE = 768  # the longest side of the crop that leaves this machine
RING = 9  # pixels of the red ring that marks the point in the crop


class PointUnavailable(ValueError):
    """No usable observation or permission; do not guess another target. The message is ours and safe to show."""


class PointStopped(RuntimeError):
    """A cancelled or expired selection, or a late model answer."""


@dataclass
class PointSelection:
    point: tuple[float, float]
    target: PointTarget | None = field(repr=False)
    captured_at: float  # on `clock`
    ttl: float
    image: Image.Image | None = field(default=None, repr=False)
    region: Box | None = None
    clock: Callable[[], float] = field(default=time.monotonic, repr=False)
    _cancelled: threading.Event = field(default_factory=threading.Event, repr=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    @property
    def expires_at(self) -> float:
        return self.captured_at + self.ttl

    def age_s(self) -> float:
        """Seconds since the item was read: how stale a description of it may be."""
        return max(0.0, self.clock() - self.captured_at)

    def check(self) -> None:
        if self._cancelled.is_set() or self.clock() >= self.expires_at:
            raise PointStopped("Selection expired or was cancelled; point again.")

    def cancel(self) -> None:
        self._cancelled.set()

    def close(self) -> None:
        self.cancel()
        with self._lock:
            if self.image is not None:
                self.image.close()
                self.image = None
            self.target = None

    def __enter__(self) -> PointSelection:
        self.check()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def observation(self) -> dict:
        with self._lock:
            self.check()
            return {
                "age_s": round(self.age_s(), 1),
                "point": self.point,
                "target": self.target.packet() if self.target else None,
                "image_region": self.region,
                "image_marker": "red ring at the selected point" if self.image is not None else None,
            }

    def ask(
        self,
        writer: Writer,
        question: str,
        *,
        allow_model: bool = False,
        history: list[dict] | None = None,
        vision: Callable[[], bool] | None = None,
    ) -> PointAnswer:
        """One explicit disclosure decision for the selected text and optional crop.

        Capture on the desktop thread, then call this on a worker with the typed or transcribed question. `cancel`
        and expiry drop a late reply; they cannot recall a request already sent.
        """
        self.check()
        if not allow_model:
            raise PointUnavailable("Allow disclosure to the selected answer provider before asking.")
        if not question.strip() or len(question) > MAX_QUESTION:
            raise PointUnavailable(f"Question must contain 1-{MAX_QUESTION} characters.")
        if self.image is not None and not (vision or writer_vision)():  # looked up now, so a setting change is seen
            raise PointUnavailable("The configured answer model is text-only; capture without an image.")
        with self._lock:
            observation = self.observation()
            image = self.image.copy() if self.image is not None else None
        try:
            self.check()
            answer = compose_point_answer(writer, question, observation, image, history)
            self.check()
            return answer
        finally:
            if image is not None:
                image.close()


def capture_point(
    point: tuple[float, float] | None = None,
    *,
    with_image: bool = False,
    radius: float = 160,
    ttl: float = HARD_TTL_S,
    clock: Callable[[], float] = time.monotonic,
) -> PointSelection:
    """Read one exact hit-tested item, optionally a bounded primary-display crop around it.

    Region capture is implemented by each adapter, never by capturing a full screenshot and cropping it afterwards.
    """
    if not math.isfinite(ttl) or not 1 <= ttl <= HARD_TTL_S or not math.isfinite(radius) or not 24 <= radius <= 240:
        raise PointUnavailable(f"Selection lifetime must be 1-{HARD_TTL_S:g} seconds and the radius 24-240.")
    point = tuple(point) if point is not None else desktop.mouse_location()
    if len(point) != 2 or not all(math.isfinite(v) for v in point):
        raise PointUnavailable("A finite screen point is required.")
    target = desktop.point_target(point)
    if target is not None and target.protected:
        raise PointUnavailable("Protected fields cannot be read or captured.")
    image = region = None
    if with_image:
        image, region = desktop.point_region(point, radius)
        # Accessibility state may change during capture: reject a protected target or a changed hit test rather
        # than pair two different observations.
        try:
            if desktop.point_target(point) != target:
                raise PointUnavailable("The pointed item changed during capture; point again.")
            left, top, right, bottom = region
            if (
                not all(math.isfinite(v) for v in region)
                or not left <= point[0] < right
                or not top <= point[1] < bottom
                or right - left > 2 * radius + 2
                or bottom - top > 2 * radius + 2
                or image.width <= 0
                or image.height <= 0
            ):
                raise PointUnavailable("The selected region is outside its bounded range.")
            prepared = image.convert("RGB")
            image.close()
            image = prepared
            image.thumbnail((IMAGE_EDGE, IMAGE_EDGE))
            x = (point[0] - left) * image.width / (right - left)
            y = (point[1] - top) * image.height / (bottom - top)
            ImageDraw.Draw(image).ellipse((x - RING, y - RING, x + RING, y + RING), outline="red", width=2)
        except BaseException:
            image.close()
            raise
    if target is None and image is None:
        raise PointUnavailable("No accessible item at this point; enable a small image crop or point again.")
    return PointSelection(point, target, clock(), ttl, image, region, clock)

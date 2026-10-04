"""Fakes for the pet window tests: a core that records what the window asks of it, and the window factory."""

from __future__ import annotations

from test_assistant_fakes import FakeLLM
from test_pet_core import PetConfig

from glide.ui.core import PetEvent


class FakePoint:
    """What the window asks of `PointMode`, recorded."""

    def __init__(self, calls) -> None:
        self.calls, self.active, self.holding = calls, False, False

    def start(self, question="", **options):
        self.calls.append(("point_start", question, options))

    def ask(self, text):
        self.calls.append(("point_ask", text))
        return True


class FakeCore:
    """What the window asks of `PetCore`, recorded."""

    def __init__(self) -> None:
        self.act = self.headset = self.record_content = False
        self.silence_ms = 600
        self.engine = "legacy"
        self.voice_active = self.busy = False
        self.calls: list = []
        self.queue: list[PetEvent] = []
        self.point = FakePoint(self.calls)

    def drain(self):
        out, self.queue = self.queue, []
        return out

    def send_text(self, text):
        self.calls.append(("text", text))
        return True

    def start_voice(self):
        self.calls.append("start_voice")

    def pause_voice(self):
        self.calls.append("pause_voice")

    def stop(self):
        self.calls.append("stop")

    def close(self):
        self.calls.append("close")


def qt_application():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def window_factory(qt_app):
    """Yields `make(core=None, config=None)`; every window made is torn down afterwards."""
    from glide.ui.pet_ui import PetWindow

    made = []

    def make(core=None, config=None):
        window = PetWindow(core or FakeCore(), config or PetConfig(llm=FakeLLM()), poll=False)
        made.append(window)
        return window

    yield make
    for window in made:
        window.timer.stop()
        window.deleteLater()
    qt_app.processEvents()

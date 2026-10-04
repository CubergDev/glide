"""A fixed visual pin. It never follows, moves or clicks the real pointer, and accepts no input or focus."""

from __future__ import annotations

import math
import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QApplication, QWidget

SIZE = 32


class PointMarker(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.WindowTransparentForInput
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAccessibleName("Pinned screen point")
        self.setFixedSize(SIZE, SIZE)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor("#ff687d"), 3))
        painter.drawEllipse(5, 5, 22, 22)
        painter.end()


def show_marker(point):
    """Draw the pin at a screen point of the primary display, or None when the point is not on it."""
    if len(point) != 2 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in point):
        return None
    screen = QApplication.primaryScreen()
    if screen is None:
        return None
    rect = screen.geometry()
    # Points arrive as the desktop adapter reads them: Windows physical pixels, macOS logical points. The pin is
    # primary-display only; a secondary display's scale is not guessed.
    scale = screen.devicePixelRatio() if sys.platform == "win32" else 1.0
    x, y = point[0] / scale, point[1] / scale
    if not 0 <= x < rect.width() or not 0 <= y < rect.height():
        return None
    marker = PointMarker()
    marker.move(rect.x() + round(x) - SIZE // 2, rect.y() + round(y) - SIZE // 2)
    marker.show()
    return marker

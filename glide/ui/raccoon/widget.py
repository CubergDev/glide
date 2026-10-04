"""Floating raccoon with a pill bar: compact (type / voice), chat input, and voice recording.

UI only. Nothing here talks to a model or opens the microphone; hook up the signals instead.
"""

from __future__ import annotations

import math
import random
import time
from collections import deque

from PySide6.QtCore import QEasingCurve, QPointF, QRect, QRectF, QSize, Qt, QTimer, QVariantAnimation, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QAbstractButton, QHBoxLayout, QLabel, QLineEdit, QStackedLayout, QWidget

from . import sprites

BAR_BG = QColor(30, 34, 42, 238)
BAR_EDGE = QColor(255, 255, 255, 26)
ICON = QColor("#c9d2dc")
ICON_DIM = QColor("#7d8896")
MINT = QColor("#67e8ba")
INK = QColor("#0e1620")
PIXEL = 3  # screen pixels per art pixel

COMPACT, CHAT, VOICE = 0, 1, 2
BAR_SIZE = {COMPACT: QSize(100, 42), CHAT: QSize(380, 48), VOICE: QSize(300, 48)}
OVERLAP = {COMPACT: 12, CHAT: 28, VOICE: 28}  # how far the raccoon sits into the bar


# --- icons --------------------------------------------------------------------


def _pen(p, color, width=1.8):
    p.setPen(QPen(color, width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    p.setBrush(Qt.BrushStyle.NoBrush)


def icon_edit(p, r, c):
    _pen(p, c)
    s = r.width()
    x, y = r.x(), r.y()
    path = QPainterPath(QPointF(x + s * 0.55, y + s * 0.18))
    path.lineTo(x + s * 0.25, y + s * 0.18)
    path.quadTo(x + s * 0.15, y + s * 0.18, x + s * 0.15, y + s * 0.28)
    path.lineTo(x + s * 0.15, y + s * 0.75)
    path.quadTo(x + s * 0.15, y + s * 0.85, x + s * 0.25, y + s * 0.85)
    path.lineTo(x + s * 0.72, y + s * 0.85)
    path.quadTo(x + s * 0.82, y + s * 0.85, x + s * 0.82, y + s * 0.75)
    path.lineTo(x + s * 0.82, y + s * 0.48)
    p.drawPath(path)
    p.drawLine(QPointF(x + s * 0.42, y + s * 0.6), QPointF(x + s * 0.82, y + s * 0.2))


def icon_wave(p, r, c):
    _pen(p, c, 2.0)
    s, cx, cy = r.width(), r.center().x(), r.center().y()
    for i, h in enumerate((0.22, 0.45, 0.7, 0.45, 0.22)):
        x = cx + (i - 2) * s * 0.17
        p.drawLine(QPointF(x, cy - s * h / 2), QPointF(x, cy + s * h / 2))


def icon_close(p, r, c):
    _pen(p, c, 2.0)
    s, cx, cy = r.width() * 0.2, r.center().x(), r.center().y()
    p.drawLine(QPointF(cx - s, cy - s), QPointF(cx + s, cy + s))
    p.drawLine(QPointF(cx - s, cy + s), QPointF(cx + s, cy - s))


def icon_send(p, r, c):
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(c)
    p.drawEllipse(QRectF(r).adjusted(2, 2, -2, -2))
    _pen(p, INK, 2.2)
    s, cx, cy = r.width() * 0.2, r.center().x(), r.center().y()
    p.drawLine(QPointF(cx, cy + s), QPointF(cx, cy - s))
    p.drawLine(QPointF(cx - s * 0.8, cy - s * 0.1), QPointF(cx, cy - s))
    p.drawLine(QPointF(cx + s * 0.8, cy - s * 0.1), QPointF(cx, cy - s))


def icon_stop(p, r, c):
    p.setPen(Qt.PenStyle.NoPen)
    p.setBrush(c)
    p.drawEllipse(QRectF(r).adjusted(2, 2, -2, -2))
    s = r.width() * 0.16
    p.setBrush(INK)
    p.drawRoundedRect(QRectF(r.center().x() - s, r.center().y() - s, 2 * s, 2 * s), 2, 2)


class IconButton(QAbstractButton):
    def __init__(self, icon, tip, size=32, filled=False, parent=None):
        super().__init__(parent)
        self.icon_fn, self.filled = icon, filled
        self.setFixedSize(size, size)
        self.setToolTip(tip)
        self.setAccessibleName(tip)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.TabFocus)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self.rect()
        if self.filled:
            color = MINT if self.isEnabled() else QColor("#4a5462")
        else:
            if self.underMouse() or self.hasFocus():
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(255, 255, 255, 22))
                p.drawEllipse(QRectF(r).adjusted(1, 1, -1, -1))
            color = ICON if self.isEnabled() else ICON_DIM
        side = min(r.width(), r.height())
        box = QRect(r.center().x() - side // 2 + 1, r.center().y() - side // 2 + 1, side - 2, side - 2)
        if not self.filled:
            box = box.adjusted(side // 6, side // 6, -side // 6, -side // 6)
        self.icon_fn(p, box, color)

    def enterEvent(self, e):
        self.update()
        super().enterEvent(e)

    def leaveEvent(self, e):
        self.update()
        super().leaveEvent(e)


# --- pieces -------------------------------------------------------------------


class RaccoonSprite(QWidget):
    """Animated pixel raccoon. Frames are drawn once per mood at the screen's pixel ratio."""

    clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(sprites.GRID_W * PIXEL, sprites.GRID_H * PIXEL)
        self.poses = sprites.moods()
        self.cache: dict[str, list[QPixmap]] = {}
        self.mood, self.frame = "", 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.advance)
        self.set_mood("idle")

    def frames(self, mood):
        if mood not in self.cache:
            ratio = max(1, round(self.devicePixelRatioF()))
            out = []
            for pose in self.poses[mood]:
                im = sprites.render(pose, PIXEL * ratio)
                qimg = QImage(im.tobytes("raw", "RGBA"), im.width, im.height, QImage.Format.Format_RGBA8888).copy()
                pix = QPixmap.fromImage(qimg)
                pix.setDevicePixelRatio(ratio)
                out.append(pix)
            self.cache[mood] = out
        return self.cache[mood]

    def set_mood(self, mood):
        if mood not in self.poses:
            raise ValueError(f"unknown mood {mood!r}; choose from {', '.join(self.poses)}")
        if mood == self.mood:
            return
        self.mood, self.frame = mood, 0
        self.setAccessibleName(f"Raccoon, {mood}")
        self.timer.start(sprites.FRAME_MS.get(mood, 180))
        self.update()

    def advance(self):
        self.frame = (self.frame + 1) % len(self.frames(self.mood))
        self.update()

    def paintEvent(self, _):
        QPainter(self).drawPixmap(0, 0, self.frames(self.mood)[self.frame])


class Waveform(QWidget):
    """Scrolling level bars. Feed real levels with push(); otherwise it idles with a gentle fake signal."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.levels = deque([0.05] * 40, maxlen=40)
        self.external = False
        self.setMinimumHeight(28)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)

    def start(self):
        self.levels.extend([0.05] * self.levels.maxlen)
        self.external = False
        self.timer.start(60)

    def stop(self):
        self.timer.stop()

    def push(self, level: float):
        self.external = True
        self.levels.append(max(0.0, min(1.0, level)))
        self.update()

    def _tick(self):
        if not self.external:
            t = time.monotonic()
            speech = max(0.0, math.sin(t * 2.3)) * (0.35 + 0.5 * random.random())
            self.levels.append(0.06 + speech)
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        n, w, h = len(self.levels), self.width(), self.height()
        step = w / n
        _pen(p, MINT, max(1.5, step * 0.5))
        for i, level in enumerate(self.levels):
            x = step * (i + 0.5)
            half = max(1.0, level * (h / 2 - 2))
            p.drawLine(QPointF(x, h / 2 - half), QPointF(x, h / 2 + half))


class Bar(QWidget):
    """Rounded glass pill holding one page per mode."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pages = QStackedLayout(self)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        p.setPen(QPen(BAR_EDGE, 1))
        p.setBrush(BAR_BG)
        p.drawRoundedRect(r, r.height() / 2, r.height() / 2)


def _row(margins, spacing=4):
    page = QWidget()
    row = QHBoxLayout(page)
    row.setContentsMargins(*margins)
    row.setSpacing(spacing)
    return page, row


# --- window -------------------------------------------------------------------


class PetWindow(QWidget):
    text_submitted = Signal(str)
    voice_started = Signal()
    voice_finished = Signal(float)  # seconds recorded
    voice_cancelled = Signal()

    def __init__(self):
        super().__init__(None, Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_MacAlwaysShowToolWindow)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowTitle("Raccoon")
        self.mode = COMPACT
        self.rest_mood = "idle"
        self.drag_from = None
        self.recording_since = 0.0

        self.pet = RaccoonSprite(self)
        self.bar = Bar(self)
        self.pet.raise_()

        # compact: two icons, like the reference pill
        page, row = _row((10, 4, 10, 4), 6)
        self.type_button = IconButton(icon_edit, "Type to the raccoon")
        self.voice_button = IconButton(icon_wave, "Talk to the raccoon")
        row.addWidget(self.type_button)
        row.addWidget(self.voice_button)
        self.bar.pages.addWidget(page)

        # chat: input | send
        page, row = _row((16, 6, 6, 6))
        self.input = QLineEdit()
        self.input.setPlaceholderText("Ask the raccoon…")
        self.input.setAccessibleName("Message")
        self.input.setFrame(False)
        self.input.setStyleSheet(
            "QLineEdit { background: transparent; color: #e8eef5; font-size: 14px; selection-background-color: #2fae86; }"
        )
        self.send_button = IconButton(icon_send, "Send", 34, filled=True)
        self.send_button.setEnabled(False)
        row.addWidget(self.input, 1)
        row.addWidget(self.send_button)
        self.bar.pages.addWidget(page)

        # voice: x | waveform | 0:03 | stop
        page, row = _row((8, 6, 6, 6), 8)
        self.cancel_button = IconButton(icon_close, "Cancel recording")
        self.wave = Waveform()
        self.clock = QLabel("0:00")
        self.clock.setStyleSheet("color: #aab6c3; font-size: 13px; font-family: Menlo, monospace;")
        self.done_button = IconButton(icon_stop, "Stop and send", 34, filled=True)
        row.addWidget(self.cancel_button)
        row.addWidget(self.wave, 1)
        row.addWidget(self.clock)
        row.addWidget(self.done_button)
        self.bar.pages.addWidget(page)

        self.clock_timer = QTimer(self)
        self.clock_timer.timeout.connect(self._tick_clock)
        self.morph = QVariantAnimation(self)
        self.morph.setDuration(170)
        self.morph.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.morph.valueChanged.connect(self._layout_bar)

        self.type_button.clicked.connect(self.open_chat)
        self.voice_button.clicked.connect(self.start_voice)
        self.input.textChanged.connect(lambda t: self.send_button.setEnabled(bool(t.strip())))
        self.input.returnPressed.connect(self.submit)
        self.send_button.clicked.connect(self.submit)
        self.cancel_button.clicked.connect(self.cancel_voice)
        self.done_button.clicked.connect(self.finish_voice)

        self.morph.finished.connect(lambda: self._fit(BAR_SIZE[self.mode]))
        self._fit(BAR_SIZE[COMPACT])
        self._layout_bar(1.0)

    # public API -----------------------------------------------------------

    def set_mood(self, mood: str, rest: bool = False):
        """Show a mood now. With rest=True it also becomes the mood the raccoon returns to."""
        if rest:
            self.rest_mood = mood
        self.pet.set_mood(mood)

    def push_level(self, level: float):
        """Feed a live microphone level (0..1) while recording."""
        self.wave.push(level)

    # modes ----------------------------------------------------------------

    def _fit(self, bar):
        """Size the window around the bar, keeping its bottom-centre where the user left it."""
        w = max(bar.width(), self.pet.width()) + 8
        h = self.pet.height() + bar.height() - OVERLAP[self.mode] + 6
        g = self.geometry()
        anchor = QPointF(g.center().x(), g.bottom()) if self.isVisible() else None
        self.resize(w, h)
        if anchor is not None:
            self.move(round(anchor.x() - w / 2), round(anchor.y() - h + 1))
        self._layout_bar(self.morph.currentValue() if self.morph.state() == QVariantAnimation.State.Running else 1.0)

    def _switch(self, mode):
        start = self.bar.size()
        self.mode = mode
        # grow now, shrink once the morph lands
        self._fit(QSize(max(start.width(), BAR_SIZE[mode].width()), max(start.height(), BAR_SIZE[mode].height())))
        self.bar.pages.setCurrentIndex(mode)
        self.morph.stop()
        self.morph.setStartValue(0.0)
        self.morph.setEndValue(1.0)
        self._from = start
        self.morph.start()

    def _layout_bar(self, t):
        t = float(t)
        target = BAR_SIZE[self.mode]
        start = getattr(self, "_from", target)
        w = round(start.width() + (target.width() - start.width()) * t)
        h = round(start.height() + (target.height() - start.height()) * t)
        bottom = self.height() - 2
        self.bar.setGeometry((self.width() - w) // 2, bottom - h, w, h)
        overlap = OVERLAP[self.mode]
        self.pet.move((self.width() - self.pet.width()) // 2, bottom - h - self.pet.height() + overlap)

    def open_chat(self):
        self._switch(CHAT)
        self.set_mood("typing")
        self.input.setFocus()

    def close_bar(self):
        if self.mode == VOICE:
            self.cancel_voice()
            return
        self.input.clear()
        self._switch(COMPACT)
        self.set_mood(self.rest_mood)

    def submit(self):
        text = self.input.text().strip()
        if not text:
            return
        self.input.clear()
        self._switch(COMPACT)
        self.set_mood(self.rest_mood)
        self.text_submitted.emit(text)

    def start_voice(self):
        self._switch(VOICE)
        self.set_mood("listening")
        self.recording_since = time.monotonic()
        self.clock.setText("0:00")
        self.wave.start()
        self.clock_timer.start(250)
        self.voice_started.emit()

    def _end_voice(self):
        self.wave.stop()
        self.clock_timer.stop()
        self._switch(COMPACT)
        self.set_mood(self.rest_mood)
        return time.monotonic() - self.recording_since

    def finish_voice(self):
        self.voice_finished.emit(self._end_voice())

    def cancel_voice(self):
        self._end_voice()
        self.voice_cancelled.emit()

    def _tick_clock(self):
        s = int(time.monotonic() - self.recording_since)
        self.clock.setText(f"{s // 60}:{s % 60:02d}")

    # window behaviour -----------------------------------------------------

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Escape and self.mode != COMPACT:
            self.close_bar()
        else:
            super().keyPressEvent(e)

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton and self.pet.geometry().contains(e.position().toPoint()):
            self.drag_from = e.globalPosition().toPoint() - self.frameGeometry().topLeft()
            self.drag_moved = False
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self.drag_from is not None:
            self.move(e.globalPosition().toPoint() - self.drag_from)
            self.drag_moved = True

    def mouseReleaseEvent(self, e):
        if self.drag_from is not None and not self.drag_moved:
            self.pet.clicked.emit()
        self.drag_from = None
        super().mouseReleaseEvent(e)

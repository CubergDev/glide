"""The raccoon window. A thin view: it forwards what the person does to `PetCore` and shows what `PetView` says.

Nothing here talks to a model, opens the microphone, writes a file or starts a process. Text from the core, the
provider chains, the screen or the person is data and is drawn as plain text, never interpreted.
"""

from __future__ import annotations

import sys

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QScrollArea,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..assistant.point_ask import MAX_QUESTION
from ..computer.config import writer_vision
from . import pet, pet_overlay
from .core import MAX_TEXT, PetCore, PetView, provider_lines
from .raccoon.widget import VOICE, icon_send
from .raccoon.widget import PetWindow as RaccoonWindow

POLL_MS = 50  # how often the window drains the core's events
DEFAULT_QUESTION = "What is this, and what should I do next?"
CONTROL, POINT = 0, 1  # the activity selector

BADGE_STYLE = "QLabel { background: %s; color: white; font-size: 11px; font-weight: bold; padding: 2px 6px; border-radius: 6px; }"
REC_COLOR, ACT_COLOR = "#c62839", "#b36b00"


def _image_allowed() -> bool:
    try:
        return writer_vision()
    except ValueError:  # a malformed setting is not a yes
        return False


def _plain(label: QLabel, name: str) -> QLabel:
    """A label that shows text as text. Anything a provider, a page or a person wrote may land in one."""
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setWordWrap(True)
    label.setAccessibleName(name)
    return label


class PetWindow(RaccoonWindow):
    def __init__(self, core: PetCore, config, *, poll: bool = True):
        super().__init__()
        self.core, self.config = core, config
        self.view = PetView()
        self.view.recording = core.record_content
        self.setWindowTitle("Glide · Raccoon")
        self.input.setMaxLength(MAX_TEXT)
        self.input.setPlaceholderText("Ask Glide…")
        self.done_button.icon_fn = icon_send
        self.done_button.setToolTip("Stop listening and run")
        self.done_button.setAccessibleName("Stop listening and run")
        self.cancel_button.setToolTip("Stop the voice session and cancel the task")
        self.cancel_button.setAccessibleName("Stop the voice session and cancel the task")

        # Always visible while they apply: content is being recorded, Glide may click and type on this Mac, or point
        # questions send the pinned item to the answer provider.
        self.badges = QLabel(self)
        self.badges.setTextFormat(Qt.TextFormat.PlainText)
        self.badges.setAccessibleName("Recording and computer-action indicators")
        self.badges.move(0, 0)

        self.settings = QWidget(self, Qt.WindowType.Popup)
        self.settings.setWindowTitle("Glide settings and activity")
        self.settings.setStyleSheet("""
            QWidget { background: #142233; color: #eff6ff; font-size: 13px; }
            QTextEdit { background: #0e1927; border: 1px solid #33465b; border-radius: 8px; padding: 8px; }
        """)
        self.settings.resize(440, 640)
        layout = QVBoxLayout(self.settings)
        self.status = _plain(QLabel(self.view.status), "Session status")
        layout.addWidget(self.status)
        # RaccoonWindow.mode is its COMPACT/CHAT/VOICE layout state: the selector must never replace it.
        self.activity = QComboBox()
        self.activity.addItems(["Control computer", "Point & ask (read only)"])
        self.activity.setAccessibleName("Interaction activity")
        layout.addWidget(self.activity)
        self.act = QCheckBox("Allow computer actions (Glide clicks and types on this Mac)")
        self.act.setChecked(False)
        self.act_banner = _plain(
            QLabel("ACT MODE: Glide will click and type on this Mac. Say stop, or move the mouse to the top-left corner."),
            "Computer actions are on",
        )
        self.act_banner.setStyleSheet(f"color: white; background: {ACT_COLOR}; padding: 4px; border-radius: 6px;")
        self.headset = QCheckBox("Headphones · allow spoken interruptions")
        self.record = QCheckBox("Record task and page content for debugging (off by default)")
        self.record.setChecked(core.record_content)
        self.record_banner = _plain(
            QLabel("RECORDING CONTENT: what you say, type and see on screen is saved by Glide's run folders."),
            "Content recording is on",
        )
        self.record_banner.setStyleSheet(f"color: white; background: {REC_COLOR}; padding: 4px; border-radius: 6px;")
        for widget in (self.act, self.act_banner, self.headset, self.record, self.record_banner):
            layout.addWidget(widget)
        self.providers = _plain(QLabel(""), "Provider chains")
        layout.addWidget(self.providers)
        self.share = QCheckBox("Share selected context for an answer")
        self.share.setToolTip("Sends your question and the selected item's text to the answer provider shown below.")
        self.include_image = QCheckBox("Include a small image crop")
        self.include_image.setToolTip("Also shares nearby visible content. Needs an image-capable answer model.")
        self.question = QLineEdit()
        self.question.setMaxLength(MAX_QUESTION)
        self.question.setPlaceholderText(DEFAULT_QUESTION)
        self.question.setAccessibleName("Question about the selected point")
        self.ask_button = QPushButton("Read this point")
        self.target = _plain(QLabel(""), "Pinned item")
        self.answer = _plain(QLabel(self.view.answer), "Point answer")
        self.answer.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.answer.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.answer_card = QScrollArea()
        self.answer_card.setWidgetResizable(True)
        self.answer_card.setMinimumHeight(110)
        self.answer_card.setMaximumHeight(200)
        self.answer_card.setWidget(self.answer)
        self.answer_note = _plain(QLabel(""), "How old the pinned snapshot is")
        self.point_provider = _plain(QLabel(""), "Answer provider")
        self.point_widgets = (
            self.point_provider,
            self.share,
            self.include_image,
            self.question,
            self.ask_button,
            self.target,
            self.answer_card,
            self.answer_note,
        )
        for widget in self.point_widgets:
            layout.addWidget(widget)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setAcceptRichText(False)
        self.log.setAccessibleName("Activity, kept in memory only")
        self.log.setPlaceholderText("Activity appears here. It is kept in memory and never written to disk by the pet.")
        layout.addWidget(self.log)

        self.menu = QMenu(self)
        self.menu.addAction("Settings and activity", self.show_settings)
        self.stop_action = self.menu.addAction("Stop current session", self.stop)
        self.menu.addSeparator()
        self.menu.addAction("Quit", self.close)

        self.activity.currentIndexChanged.connect(self.refresh)
        self.share.toggled.connect(self.refresh)
        self.ask_button.clicked.connect(self.ask_typed)
        self.act.toggled.connect(self._set_act)
        self.headset.toggled.connect(self._set_headset)
        self.record.toggled.connect(self._set_record)
        self.text_submitted.connect(self.submit_text)
        self.voice_started.connect(self.begin_voice)
        self.voice_finished.connect(lambda _seconds: self.core.pause_voice())
        self.voice_cancelled.connect(self.stop)

        self.marker = None  # the pin drawn on screen, and the point it was drawn for
        self.marker_point = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll)
        if poll:
            self.timer.start(POLL_MS)
        self.refresh()

    # -- point and ask ---------------------------------------------------------------------------

    @property
    def point_mode(self) -> bool:
        return self.activity.currentIndex() == POINT

    def _image(self) -> bool:
        return self.share.isChecked() and self.include_image.isChecked() and _image_allowed()

    def submit_text(self, text: str) -> None:
        """What the bar's text box sends: a request to Glide, or in point mode a question about the pin."""
        if self.point_mode:
            self.point_question(text)
        else:
            self.core.send_text(text)

    def point_question(self, text: str) -> None:
        if self.core.point.holding:
            self.core.point.ask(text)
        else:
            self.core.point.start(text, share=self.share.isChecked(), with_image=self._image())

    def ask_typed(self) -> None:
        if self.point_mode:
            self.point_question(self.question.text().strip() or DEFAULT_QUESTION)

    def start_voice(self) -> None:
        if self.point_mode and not self.share.isChecked():
            self.view.status = "Tick “Share selected context for an answer” to ask about a point by voice."
            self.refresh()
            return
        super().start_voice()

    def begin_voice(self) -> None:
        if self.point_mode and not self.core.voice_active:
            self.core.point.start("", share=True, with_image=self._image(), voice=True)
        else:
            self.core.start_voice()

    def _sync_marker(self) -> None:
        pin = self.view.pin
        if pin == self.marker_point:
            return
        if self.marker is not None:
            pet.dismiss_point_marker(self.marker)
            self.marker = None
        self.marker_point = pin
        if pin is not None:
            self.marker = pet.show_point_marker(list(pin))

    # -- settings --------------------------------------------------------------------------------

    def _set_act(self, on: bool) -> None:
        self.core.act = on
        self.refresh()

    def _set_headset(self, on: bool) -> None:
        self.core.headset = on

    def _set_record(self, on: bool) -> None:
        self.core.record_content = on
        self.view.recording = on  # the core also emits `recording`; this keeps a refresh in between from undoing the click
        self.refresh()

    def show_settings(self) -> None:
        self.providers.setText("\n".join(provider_lines(self.config)))
        screen = self.screen().availableGeometry()
        self.settings.move(
            max(screen.left(), min(self.x(), screen.right() - self.settings.width() + 1)),
            max(screen.top(), min(self.y() - self.settings.height(), screen.bottom() - self.settings.height() + 1)),
        )
        self.settings.show()

    def contextMenuEvent(self, event):
        self.menu.popup(event.globalPos())
        event.accept()

    # -- events from the core --------------------------------------------------------------------

    def poll(self) -> None:
        events = self.core.drain()
        for event in events:
            self.view.apply(event)
            if event.type == "mic" and not event.data["open"] and self.mode == VOICE:
                self._end_voice()  # the core said the microphone is off: the voice bar must not claim otherwise
        if events:
            self.refresh()

    def refresh(self, *_) -> None:
        self.set_mood(self.view.mood, rest=True)
        self.status.setText(self.view.status)
        self.log.setPlainText("\n".join(self.view.lines))
        bar = self.log.verticalScrollBar()
        bar.setValue(bar.maximum())
        locked = self.core.voice_active or self.core.busy
        for box in (self.act, self.headset, self.record):
            box.setEnabled(not locked)
        self.record.blockSignals(True)  # showing the state must never change it
        self.record.setChecked(self.view.recording)
        self.record.blockSignals(False)
        self.record_banner.setVisible(self.view.recording)
        sharing = self.point_mode and self.share.isChecked()
        badges = [
            text for text, show in (("● REC", self.view.recording), ("ACT", self.act.isChecked()), ("SHARE", sharing)) if show
        ]
        self.badges.setText("  ".join(badges))
        self.badges.setStyleSheet(BADGE_STYLE % (REC_COLOR if self.view.recording else ACT_COLOR))
        self.badges.setVisible(bool(badges))
        self.badges.adjustSize()
        self._refresh_point(locked)
        self._sync_marker()

    def _refresh_point(self, locked: bool) -> None:
        point, active, holding = self.point_mode, self.core.point.active, self.core.point.holding
        self.activity.setEnabled(not locked and not active)
        self.act.setVisible(not point)
        self.act_banner.setVisible(self.act.isChecked() and not point)
        for widget in self.point_widgets:
            widget.setVisible(point)
        self.share.setEnabled(not active)
        self.include_image.setEnabled(not active and self.share.isChecked() and _image_allowed())
        self.ask_button.setText("Ask this point" if self.share.isChecked() else "Read this point")
        self.ask_button.setEnabled(not active or holding)
        self.question.setEnabled(not active or holding)
        self.voice_button.setEnabled(not point or self.share.isChecked())
        self.voice_button.setToolTip("Ask about the point by voice" if point else "Talk to the raccoon")
        self.input.setMaxLength(MAX_QUESTION if point else MAX_TEXT)
        self.input.setPlaceholderText("Ask about the pinned point…" if point else "Ask Glide…")
        self.target.setText(self.view.target)
        self.answer.setText(self.view.answer)
        self.answer_note.setText(self.view.answer_note)
        self.point_provider.setText(self.view.provider)
        self.stop_action.setEnabled(locked or active)

    def stop(self) -> None:
        """Stop the answer, the speech and the task, and end the voice session."""
        self.core.stop()
        if self.mode == VOICE:
            self._end_voice()

    def closeEvent(self, event):
        self.timer.stop()
        self.settings.close()
        if self.marker is not None:
            pet.dismiss_point_marker(self.marker)
            self.marker = None
        self.core.close()
        event.accept()


def show_pet(core: PetCore, config) -> int:
    """Open the window and run Qt's event loop until it closes. The one function that makes a real window."""
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName("Glide Pet")
    window = PetWindow(core, config)
    area = app.primaryScreen().availableGeometry()
    window.move(area.right() - window.width() - 24, area.bottom() - window.height() - 24)
    window.show()
    pet_overlay.configure_overlay(window)
    return app.exec()  # passive until the person asks: no microphone, request or task starts by itself

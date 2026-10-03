"""The raccoon window. A thin view: it forwards what the person does to `PetCore` and shows what `PetView` says.

Nothing here talks to a model, opens the microphone, writes a file or starts a process. Text from the core, the
provider chains, the screen or the person is data and is drawn as plain text, never interpreted.
"""

from __future__ import annotations

import sys

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication, QCheckBox, QLabel, QMenu, QTextEdit, QVBoxLayout, QWidget

from . import pet_overlay
from .core import MAX_TEXT, PetCore, PetView, provider_lines
from .raccoon.widget import VOICE, icon_send
from .raccoon.widget import PetWindow as RaccoonWindow

POLL_MS = 50  # how often the window drains the core's events

BADGE_STYLE = "QLabel { background: %s; color: white; font-size: 11px; font-weight: bold; padding: 2px 6px; border-radius: 6px; }"
REC_COLOR, ACT_COLOR = "#c62839", "#b36b00"


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

        # Always visible while they apply: content is being recorded, or Glide may click and type on this Mac.
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
        self.settings.resize(440, 560)
        layout = QVBoxLayout(self.settings)
        self.status = _plain(QLabel(self.view.status), "Session status")
        layout.addWidget(self.status)
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

        self.act.toggled.connect(self._set_act)
        self.headset.toggled.connect(self._set_headset)
        self.record.toggled.connect(self._set_record)
        self.text_submitted.connect(self.core.send_text)
        self.voice_started.connect(self.core.start_voice)
        self.voice_finished.connect(lambda _seconds: self.core.pause_voice())
        self.voice_cancelled.connect(self.stop)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll)
        if poll:
            self.timer.start(POLL_MS)
        self.refresh()

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

    def refresh(self) -> None:
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
        self.act_banner.setVisible(self.act.isChecked())
        badges = [text for text, show in (("● REC", self.view.recording), ("ACT", self.act.isChecked())) if show]
        self.badges.setText("  ".join(badges))
        self.badges.setStyleSheet(BADGE_STYLE % (REC_COLOR if self.view.recording else ACT_COLOR))
        self.badges.setVisible(bool(badges))
        self.badges.adjustSize()
        self.stop_action.setEnabled(locked)

    def stop(self) -> None:
        """Stop the answer, the speech and the task, and end the voice session."""
        self.core.stop()
        if self.mode == VOICE:
            self._end_voice()

    def closeEvent(self, event):
        self.timer.stop()
        self.settings.close()
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

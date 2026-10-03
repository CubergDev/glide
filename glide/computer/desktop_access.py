"""Permission setup before a voice session opens its microphone or accepts tasks."""

from __future__ import annotations

import os
import time

from .models import DesktopPermissionError
from .platform_adapter import desktop


def prepare_desktop(act: bool, *, output=print, wait_seconds=60, cancelled=lambda: False):
    if cancelled():
        raise KeyboardInterrupt

    def missing():
        return {
            "accessibility": act and not desktop.accessibility_trusted(),
            "screen_capture": not desktop.screen_capture_trusted(),
        }

    required = missing()
    if not any(required.values()):
        return
    launcher = {"Apple_Terminal": "Terminal", "iTerm.app": "iTerm", "vscode": "Visual Studio Code"}.get(
        os.environ.get("TERM_PROGRAM"), "the application launching this session"
    )
    output(
        f"Desktop access is required for {launcher}. Approve the native permission requests in "
        "Privacy & Security: Screen Recording for all voice sessions, Accessibility for computer actions. "
        "Permission granted to another launcher, such as Codex, does not authorize this session. "
        "Waiting for approval before opening the microphone. Ctrl-C exits."
    )
    desktop.request_permissions(**required)
    deadline = time.monotonic() + wait_seconds
    while True:
        if cancelled():
            raise KeyboardInterrupt
        required = missing()
        if not any(required.values()):
            return
        if time.monotonic() >= deadline:
            labels = [
                name
                for key, name in (("accessibility", "Accessibility"), ("screen_capture", "Screen Recording"))
                if required[key]
            ]
            raise DesktopPermissionError(
                f"{', '.join(labels)} access is still missing for {launcher}. No microphone or task was started. "
                "If already enabled, macOS may require quitting and reopening the launching application before it applies."
            )
        time.sleep(0.5)

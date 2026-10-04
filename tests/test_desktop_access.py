"""Launch-context permission setup and capture failures, without touching macOS."""

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from glide.computer import desktop_access, macos
from glide.computer.models import DesktopError, DesktopPermissionError
from glide.computer.platform_adapter import using

SCREENSHOT = macos.screenshot
REQUEST_PERMISSIONS = macos.request_permissions


@pytest.mark.parametrize("act", [False, True])
def test_permissions_are_requested_for_the_actual_launch_context(monkeypatch, act):
    calls, output = [], []
    monkeypatch.setenv("TERM_PROGRAM", "Apple_Terminal")
    adapter = SimpleNamespace(
        accessibility_trusted=lambda: False,
        screen_capture_trusted=lambda: False,
        request_permissions=lambda **kw: calls.append(kw),
    )
    with using(adapter), pytest.raises(DesktopPermissionError, match="Terminal") as error:
        desktop_access.prepare_desktop(act, output=output.append, wait_seconds=0)
    assert calls == [{"accessibility": act, "screen_capture": True}]
    assert "No microphone or task was started" in str(error.value)
    assert ("Accessibility access" in str(error.value) or "Accessibility, Screen Recording" in str(error.value)) == act
    assert "another launcher" in output[0]


def test_setup_resumes_when_native_permission_is_granted(monkeypatch):
    state = {"allowed": False}
    monkeypatch.setattr(desktop_access.time, "sleep", lambda _: state.update(allowed=True))
    requested = []
    adapter = SimpleNamespace(
        accessibility_trusted=lambda: state["allowed"],
        screen_capture_trusted=lambda: state["allowed"],
        request_permissions=lambda **kw: requested.append(kw),
    )
    with using(adapter):
        desktop_access.prepare_desktop(True, output=lambda _: None)
    assert requested == [{"accessibility": True, "screen_capture": True}]


def test_granted_capture_does_not_request_accessibility_for_preview():
    adapter = SimpleNamespace(
        accessibility_trusted=lambda: pytest.fail("Preview must not need Accessibility"),
        screen_capture_trusted=lambda: True,
        request_permissions=lambda **kw: pytest.fail("Prompted despite having permission"),
    )
    with using(adapter):
        desktop_access.prepare_desktop(False)


def test_native_setup_uses_os_requests_not_privacy_database_mutation(monkeypatch):
    calls = []
    monkeypatch.setattr(
        macos,
        "AS",
        SimpleNamespace(kAXTrustedCheckOptionPrompt="prompt", AXIsProcessTrustedWithOptions=lambda kw: calls.append(kw)),
    )
    monkeypatch.setattr(macos, "Quartz", SimpleNamespace(CGRequestScreenCaptureAccess=lambda: calls.append("screen")))
    REQUEST_PERMISSIONS(accessibility=True, screen_capture=True)
    assert calls == [{"prompt": True}, "screen"]


def test_denied_capture_never_launches_screencapture(monkeypatch):
    monkeypatch.setattr(macos, "screen_capture_trusted", lambda: False)
    monkeypatch.setattr(macos.subprocess, "run", lambda *a, **kw: pytest.fail("Started capture without permission"))
    with pytest.raises(DesktopPermissionError, match="Screen Recording"):
        SCREENSHOT()


@pytest.mark.parametrize("success", [False, True])
def test_capture_cleanup_and_failure_are_typed(monkeypatch, success):
    paths = []
    monkeypatch.setattr(macos, "screen_capture_trusted", lambda: True)

    def capture(args, **kwargs):
        paths.append(Path(args[-1]))
        if not success:
            raise subprocess.CalledProcessError(1, args, stderr=b"could not create image from display")
        Image.new("RGB", (64, 64), "red").save(paths[-1])

    monkeypatch.setattr(macos.subprocess, "run", capture)
    if success:
        result = SCREENSHOT()
        assert result.getpixel((0, 0)) == (255, 0, 0)
    else:
        with pytest.raises(DesktopError, match="Screen capture failed"):
            SCREENSHOT()
    assert not paths[0].parent.exists()

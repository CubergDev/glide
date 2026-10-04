"""Only fake panels and headless Qt; never operate a real macOS window."""

from types import SimpleNamespace

import pytest

from glide.ui import pet_overlay

CONFIGURE_PANEL = pet_overlay.configure_panel
CONFIGURE_OVERLAY = pet_overlay.configure_overlay


@pytest.mark.parametrize("modern", [False, True])
def test_overlay_retains_other_flags_and_replaces_conflicting_space_behaviors(modern):
    kit = SimpleNamespace(
        NSWindowCollectionBehaviorMoveToActiveSpace=1,
        NSWindowCollectionBehaviorFullScreenPrimary=2,
        NSWindowCollectionBehaviorFullScreenNone=4,
        NSWindowCollectionBehaviorCanJoinAllSpaces=8,
        NSWindowCollectionBehaviorFullScreenAuxiliary=16,
        NSFloatingWindowLevel=3,
    )
    current, expected = 1024 | 1 | 2 | 4, 1024 | 8 | 16
    if modern:
        kit.NSWindowCollectionBehaviorPrimary = 32
        kit.NSWindowCollectionBehaviorAuxiliary = 64
        kit.NSWindowCollectionBehaviorCanJoinAllApplications = 128
        current |= 32 | 64
        expected |= 128
    calls = {}
    panel = SimpleNamespace(
        collectionBehavior=lambda: current,
        setCollectionBehavior_=lambda value: calls.update(behavior=value),
        setHidesOnDeactivate_=lambda value: calls.update(hides=value),
        setLevel_=lambda value: calls.update(level=value),
    )
    CONFIGURE_PANEL(panel, kit)
    assert calls == {"behavior": expected, "hides": False, "level": 3}


def test_offscreen_qt_never_requests_a_native_window_handle(monkeypatch):
    pytest.importorskip("PySide6.QtGui")
    from PySide6.QtGui import QGuiApplication

    monkeypatch.setattr(QGuiApplication, "platformName", lambda: "offscreen")
    widget = SimpleNamespace(winId=lambda: pytest.fail("Offscreen tests must not reach AppKit"))
    CONFIGURE_OVERLAY(widget)


@pytest.mark.expect_refusals
def test_new_native_panel_entrypoints_are_guarded():
    with pytest.raises(RuntimeError, match="real machine"):
        pet_overlay.configure_overlay(object())
    with pytest.raises(RuntimeError, match="real machine"):
        pet_overlay.configure_panel(object(), object())

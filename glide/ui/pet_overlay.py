"""Configure only this app's macOS panel; never activate another app or capture/input."""

from __future__ import annotations

import sys


def configure_overlay(widget):
    from PySide6.QtGui import QGuiApplication

    if sys.platform != "darwin" or QGuiApplication.platformName() != "cocoa":
        return
    import AppKit
    import objc

    view = objc.objc_object(c_void_p=int(widget.winId()))
    configure_panel(view.window(), AppKit)


def configure_panel(panel, appkit):
    """Respect other flags, excluding mutually exclusive Space/fullscreen policies."""
    mask = appkit.NSWindowCollectionBehaviorMoveToActiveSpace | appkit.NSWindowCollectionBehaviorFullScreenPrimary
    mask |= getattr(appkit, "NSWindowCollectionBehaviorFullScreenNone", 0)
    flags = appkit.NSWindowCollectionBehaviorCanJoinAllSpaces | appkit.NSWindowCollectionBehaviorFullScreenAuxiliary
    # New macOS versions expose an explicit utility-window policy for Stage Manager.
    if hasattr(appkit, "NSWindowCollectionBehaviorCanJoinAllApplications"):
        mask |= appkit.NSWindowCollectionBehaviorPrimary | appkit.NSWindowCollectionBehaviorAuxiliary
        flags |= appkit.NSWindowCollectionBehaviorCanJoinAllApplications
    panel.setCollectionBehavior_((panel.collectionBehavior() & ~mask) | flags)
    if hasattr(panel, "setHidesOnDeactivate_"):
        panel.setHidesOnDeactivate_(False)
    panel.setLevel_(appkit.NSFloatingWindowLevel)

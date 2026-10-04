"""Guards for the machine-reaching calls added with the computer core (registered by conftest, by path).

An autouse fixture, like `no_real_machine` in conftest. It refuses what the permission setup, the capture
preflight, the foreground-app query and the execution-phase adapter methods can do on this computer: show
the permission prompts, ask the OS about the screen-recording grant, read the frontmost app, drive a
browser's tabs through a scripted process, send shortcuts, and walk or scroll another app's elements.

A test that needs one of these patches it itself, after this fixture has run. Tests that exercise the real
body keep a reference to it at import (`SCREENSHOT = macos.screenshot`) and patch what it calls.

Not covered here, because the module is not in this tree yet: `glide.computer.execution.dom` binds
`_get_json` at import (`from ..browser.cdp import _get_json`), so refusing `cdp._get_json` does not reach its
copy. When that module is ported it is refused below as soon as it can be imported.
"""

from __future__ import annotations

import importlib

import pytest

from glide.computer import macos, windows

MACOS_CALLS = (
    "request_permissions",
    "screen_capture_trusted",
    "frontmost_app",
    "frontmost_app_and_pid",
    "frontmost_pid",
    "_browser_jxa",
    "execution_tabs",
    "execution_tab",
    "execution_shortcut",
    "execution_scrolls",
    "execution_labels",
    "execution_scroll",
)
WINDOWS_CALLS = (
    "request_permissions",
    "screen_capture_trusted",
    "execution_tabs",
    "execution_tab",
    "execution_shortcut",
    "execution_scrolls",
    "execution_labels",
    "execution_scroll",
)
# The native entry points under them, refused where the real framework is loaded.
NATIVE_CALLS = (
    ("ApplicationServices", ("AXIsProcessTrustedWithOptions",)),
    ("Quartz", ("CGRequestScreenCaptureAccess", "CGPreflightScreenCaptureAccess")),
)
# Modules of a later phase that bind a machine-reaching name at import. Refused once they exist.
LATER_MODULES = (("glide.computer.execution.dom", ("_get_json",)),)


def _refuse(what: str):
    def call(*args, **kwargs):
        from conftest import refusal  # the shared ledger lives in conftest: this file is loaded under two names

        raise refusal(what)

    return call


def _loaded(module, name: str) -> bool:
    """Whether `module` really has `name`: the import-only stand-ins raise on any unset attribute."""
    try:
        return hasattr(module, name)
    except RuntimeError:
        return False


@pytest.fixture(autouse=True)
def no_real_machine_core_run(monkeypatch):
    for name in MACOS_CALLS:
        monkeypatch.setattr(macos, name, _refuse(f"macos.{name}"))
    for name in WINDOWS_CALLS:
        monkeypatch.setattr(windows, name, _refuse(f"windows.{name}"))
    for module_name, names in NATIVE_CALLS:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        for name in names:
            if _loaded(module, name):
                monkeypatch.setattr(module, name, _refuse(f"{module_name}.{name}"))
    try:
        import AppKit
    except ImportError:
        AppKit = None
    if AppKit is not None:
        # `macos.frontmost_app_and_pid` imports the name at call time, so replacing the module attribute is
        # enough; the Objective-C class itself cannot be patched and restored.
        refused = type("NSWorkspace", (), {"sharedWorkspace": staticmethod(_refuse("AppKit.NSWorkspace"))})
        monkeypatch.setattr(AppKit, "NSWorkspace", refused)
    for module_name, names in LATER_MODULES:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        for name in names:
            if _loaded(module, name):
                monkeypatch.setattr(module, name, _refuse(f"{module_name}.{name}"))

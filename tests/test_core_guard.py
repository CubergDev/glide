"""The refusals in tests/guards_core-run.py hold: each call below fails instead of reaching this machine."""

import pytest

from glide.computer import macos, windows

REFUSED = "reached the real machine"
MACOS = {
    "request_permissions": {"accessibility": True, "screen_capture": True},
    "screen_capture_trusted": {},
    "frontmost_app": {},
    "frontmost_app_and_pid": {},
    "frontmost_pid": {},
}
EXECUTION = {
    "execution_tabs": ("Google Chrome",),
    "execution_tab": ("Google Chrome", "navigate", "1", "https://example.test/"),
    "execution_shortcut": ("l", ("command",)),
    "execution_scrolls": (1,),
    "execution_labels": (1,),
    "execution_scroll": (object(), "down"),
}


@pytest.mark.parametrize("name", MACOS)
def test_macos_permission_and_foreground_queries_are_refused(name):
    with pytest.raises(RuntimeError, match=REFUSED):
        getattr(macos, name)(**MACOS[name])


@pytest.mark.parametrize("adapter", [macos, windows], ids=["macos", "windows"])
@pytest.mark.parametrize("name", EXECUTION)
def test_execution_adapter_methods_are_refused(adapter, name):
    with pytest.raises(RuntimeError, match=REFUSED):
        getattr(adapter, name)(*EXECUTION[name])


def test_windows_permission_calls_are_refused():
    for name in ("request_permissions", "screen_capture_trusted"):
        with pytest.raises(RuntimeError, match=REFUSED):
            getattr(windows, name)()


def test_the_scripted_browser_process_cannot_start():
    with pytest.raises(RuntimeError, match=REFUSED):
        macos._browser_jxa("Google Chrome", "JSON.stringify({})")


def test_the_native_permission_prompts_and_workspace_query_are_refused():
    ApplicationServices = pytest.importorskip("ApplicationServices")
    Quartz = pytest.importorskip("Quartz")
    AppKit = pytest.importorskip("AppKit")
    for call in (
        lambda: ApplicationServices.AXIsProcessTrustedWithOptions({}),
        Quartz.CGRequestScreenCaptureAccess,
        Quartz.CGPreflightScreenCaptureAccess,
        AppKit.NSWorkspace.sharedWorkspace,
    ):
        with pytest.raises(RuntimeError, match=REFUSED):
            call()

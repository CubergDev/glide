"""The native-API refusals: every Quartz, ApplicationServices, AppKit, Foundation and Vision call production code uses is
refused unless a test patches it, and a new call cannot slip past.

Three layers of proof. Each refusal the table applied is called once with a synthetic call (it must raise and record). The
source of the adapters is read, so a framework call added to `macos.py` or `windows.py` that no row refuses fails here by
name. And the session layer, which exists where no per-test patch does, is checked with the per-test layer removed.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import conftest
import pytest
from conftest import REFUSALS, SESSION_ROWS, _Refused

pytestmark = pytest.mark.expect_refusals

COMPUTER = Path(__file__).resolve().parents[1] / "glide" / "computer"


def applied_refusals() -> list[tuple[object, str, str]]:
    """Every refusal the table puts on this machine's modules, found the way the fixture finds them."""
    guard = conftest.Guard(pytest.MonkeyPatch())
    try:
        guard.table()
        return list(guard.applied)
    finally:
        guard.monkeypatch.undo()


def test_every_refusal_the_table_applied_refuses_a_synthetic_call():
    applied = applied_refusals()
    assert applied, "no refusal applied"
    ledger = conftest._LEDGERS[-1]
    for owner, name, what in applied:
        target = getattr(owner, name)
        assert isinstance(target, _Refused), f"{what} is live in a test"
        before = len(ledger.fired)
        with pytest.raises(RuntimeError, match="real machine"):
            target()
        with pytest.raises(RuntimeError, match="real machine"):
            target(object(), "synthetic", 1)
        assert ledger.fired[before:] == [what, what], f"{what} did not record itself"


def test_the_calls_the_macos_adapter_makes_are_each_among_the_refusals():
    refused = {what for _, _, what in applied_refusals()}
    for what in (
        "Quartz.CGEventCreate",
        "Quartz.CGEventPost",
        "Quartz.CGMainDisplayID",
        "Quartz.CGDisplayBounds",
        "Quartz.CGWindowListCopyWindowInfo",
        "Quartz.CGPreflightScreenCaptureAccess",
        "Quartz.CGRequestScreenCaptureAccess",
        "ApplicationServices.AXIsProcessTrusted",
        "ApplicationServices.AXIsProcessTrustedWithOptions",
        "ApplicationServices.AXUIElementCreateSystemWide",
        "ApplicationServices.AXUIElementCreateApplication",
        "ApplicationServices.AXUIElementCopyAttributeValue",
        "ApplicationServices.AXUIElementCopyActionNames",
        "ApplicationServices.AXUIElementSetMessagingTimeout",
        "ApplicationServices.AXUIElementPerformAction",
        "ApplicationServices.AXUIElementSetAttributeValue",
        "AppKit.NSWorkspace",
        "AppKit.NSScreen",
        "Vision.VNImageRequestHandler",
        "ocrmac.ocrmac.OCR",
    ):
        if conftest._import_quiet(what.split(".")[0]) is None:
            continue  # the framework is not installed on this machine
        assert what in refused, f"{what} is used by the macOS adapter and not refused"


# What `macos.py` may call that reaches nothing: it builds or unpacks values, and posting is refused separately.
MACOS_PURE = {
    ("Quartz", "CGEventCreateMouseEvent"),
    ("Quartz", "CGEventCreateKeyboardEvent"),
    ("Quartz", "CGEventCreateScrollWheelEvent"),
    ("Quartz", "CGEventKeyboardSetUnicodeString"),
    ("Quartz", "CGEventSetFlags"),
    ("Quartz", "CGEventGetLocation"),
    ("AS", "AXValueGetValue"),
}
MACOS_ALIASES = {"AS": "ApplicationServices", "Quartz": "Quartz", "ocrmac": "ocrmac.ocrmac"}


def framework_calls(source: Path, aliases: dict[str, str]) -> set[tuple[str, str]]:
    """`(alias, name)` for every `alias.name` the file reads where the name is not a constant (`k...`), plus every
    `from AppKit import X` and its relatives."""
    tree = ast.parse(source.read_text(encoding="utf-8"))
    found: set[tuple[str, str]] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in aliases:
            if not (node.attr.startswith("k") and node.attr[1:2].isupper()):
                found.add((node.value.id, node.attr))
        elif isinstance(node, ast.ImportFrom) and node.module in aliases.values():
            found.update((node.module, alias.name) for alias in node.names)
    return found


def covered_names(module_name: str) -> set[str]:
    names: set[str] = set()
    for spec, paths in REFUSALS:
        if spec == module_name:
            for module in conftest._modules(spec):
                for path in paths:
                    names.update(name for _, name, _ in conftest.resolve_row(spec, module, path))
    return names


def test_a_framework_call_added_to_macos_py_must_be_refused_or_listed_pure():
    if sys.platform != "darwin":
        pytest.skip("the refusals cover the macOS frameworks only where they are the real ones")
    pytest.importorskip("Quartz")
    pytest.importorskip("ApplicationServices")
    aliases = {**MACOS_ALIASES, "AppKit": "AppKit"}
    calls = framework_calls(COMPUTER / "macos.py", aliases)
    assert ("AS", "AXUIElementCopyAttributeValue") in calls, "the scan reads nothing: macos.py changed shape"
    assert ("AppKit", "NSWorkspace") in calls
    missing = []
    for alias, name in sorted(calls):
        if (alias, name) in MACOS_PURE:
            continue
        module_name = aliases.get(alias, alias)
        if name not in covered_names(module_name):
            missing.append(f"{module_name}.{name}")
    assert not missing, (
        f"macos.py calls {missing}: add each to REFUSALS in tests/conftest.py, or to MACOS_PURE if it reaches nothing"
    )


def test_the_pure_list_names_calls_macos_py_still_makes():
    calls = framework_calls(COMPUTER / "macos.py", MACOS_ALIASES)
    stale = MACOS_PURE - calls
    assert not stale, f"{sorted(stale)} are no longer called: drop them from MACOS_PURE"
    refused = {(alias, name) for alias, name in MACOS_PURE if name in covered_names(MACOS_ALIASES[alias])}
    assert not refused, f"{sorted(refused)} are refused after all: they are not pure"


WINDOWS_ALIASES = {
    "win32api": "win32api",
    "win32gui": "win32gui",
    "auto": "uiautomation",
    "winocr": "winocr",
    "ImageGrab": "PIL.ImageGrab",
    "webbrowser": "webbrowser",
}
WINDOWS_PURE = {("auto", "PatternId")}  # UI Automation's pattern ids are constants


def test_a_framework_call_added_to_windows_py_must_be_refused_or_listed_pure():
    """Read from the source, so it runs on any OS: the Windows modules themselves are import-only stand-ins here."""
    calls = framework_calls(COMPUTER / "windows.py", WINDOWS_ALIASES)
    assert ("win32gui", "GetForegroundWindow") in calls, "the scan reads nothing: windows.py changed shape"
    named: dict[str, set[str]] = {}
    for spec, paths in REFUSALS:
        for path in paths:
            named.setdefault(spec, set()).update(path.lstrip("?").split("|"))
    missing = []
    for alias, name in sorted(calls):
        if (alias, name) in WINDOWS_PURE:
            continue
        module_name = WINDOWS_ALIASES.get(alias, alias)
        if name not in named.get(module_name, set()):
            missing.append(f"{module_name}.{name}")
    assert not missing, f"windows.py calls {missing}: add each to REFUSALS in tests/conftest.py, or to WINDOWS_PURE"


# ------------------------------------------------------------------ the session layer


def test_a_process_start_or_an_input_event_is_refused_with_no_per_test_layer(monkeypatch):
    """Collection, session fixtures and interpreter exit have only the session layer. Undo the test's own, then look."""
    monkeypatch.undo()
    # Assert each is the refusal first: if the layer were missing, calling the real thing would start a process.
    import os

    if "_posixsubprocess" in sys.modules:
        assert isinstance(sys.modules["_posixsubprocess"].fork_exec, _Refused)
    assert subprocess.Popen.__init__ is conftest._refusing_popen_init

    assert isinstance(os.fork, _Refused)
    assert isinstance(os.system, _Refused)
    with pytest.raises(RuntimeError, match="real machine"):
        subprocess.Popen(["true"])
    with pytest.raises(RuntimeError, match="real machine"):
        os.system("true")
    import multiprocessing.util

    with pytest.raises(RuntimeError, match="real machine"):
        multiprocessing.util.spawnv_passfds("/usr/bin/true", ["true"], ())
    from glide.computer import macos

    assert isinstance(macos._post, _Refused)
    with pytest.raises(RuntimeError, match="real machine"):
        macos._post(object())
    try:
        import Quartz
    except ImportError:
        return
    if "Quartz" not in conftest._STUBBED:
        assert isinstance(Quartz.CGEventPost, _Refused)
        with pytest.raises(RuntimeError, match="real machine"):
            Quartz.CGEventPost(0, object())


def test_the_session_rows_are_part_of_the_table_and_resolve():
    assert all(row in REFUSALS for row in SESSION_ROWS)
    assert conftest._SESSION_LAYER, "the session layer was never installed"
    applied = {what for _, _, what in conftest._SESSION_LAYER[0].applied}
    assert "multiprocessing.util.spawnv_passfds" in applied
    assert "os.fork" in applied


def test_spawnv_passfds_reaches_fork_exec_and_not_popen():
    """The finding: multiprocessing's spawn path forks through `_posixsubprocess` without `Popen`."""
    pytest.importorskip("_posixsubprocess")
    import _posixsubprocess

    assert isinstance(_posixsubprocess.fork_exec, _Refused)
    assert isinstance(subprocess._fork_exec, _Refused)  # `subprocess` holds its own reference to it

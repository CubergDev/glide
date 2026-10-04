"""Shared fixtures, a guard that keeps every test off the real machine, and import-only
stand-ins for the platform-only modules.

The suite is pure logic and should run on any OS. The platform adapters
(`glide.computer.macos`, `glide.computer.windows`) import their
platform's packages at module scope, but the tests only ever import them -- they
call nothing but the pure rules -- so a stand-in that exists and raises on any
real use is enough to run the whole suite anywhere. Where a real module is
installed, nothing is registered for it.

The guard has these layers (see `no_real_machine`):

1. Refusals written out by name for code that exists in the tree today (the macOS and Windows adapters, the
   browser backend, processes, sockets). These fail loudly if the target is renamed away.
2. `REFUSALS`, a table of targets that exist once a package is installed or a module is ported (`frontmost_app`,
   `sounddevice`, `websockets`, `uvicorn`, `mcp` stdio, `keyring`, the pet, webhooks, the Quartz and ApplicationServices
   calls ...). A row whose module is absent (not installed, an import-only stand-in, a module not ported yet) is skipped. A
   row whose module IS importable must resolve every name it lists, or every test fails with the missing name: a renamed
   attribute can no longer leave a dead row. A change that adds a call reaching the machine adds its row in the same commit.
3. Alias rebinding: `from x import y` copies a reference at import time, so refusing `x.y` alone leaves the copy live
   (that is how `dom._get_json` escaped the `cdp._get_json` guard). Every refusal of a function or class also replaces the
   same object in every module already imported, except the test modules themselves (they keep real references on purpose,
   `SCREENSHOT = macos.screenshot`, and patch what the body calls).
4. The refusal ledger. Production code often catches `Exception` (`perception.ax_nodes`, `macos._ax_attr`), so a refusal
   that raises can be swallowed and the test still passes on whatever the fallback returned. Every refusal therefore also
   records itself (`refusal`), and a test that fired one fails at teardown unless it said so with
   `@pytest.mark.expect_refusals` (optionally naming substrings of the refusals it expects).
5. A session layer, installed at `pytest_configure` and never undone, so collection, session fixtures and interpreter exit
   cannot start a process or post an input event either (the per-test layer lives in a monkeypatch and does not exist there).
6. Sockets: a test may connect only to a listener it bound itself (see `_loopback_only`), and write files only under the
   temp directory (see `_no_writes_outside_temp`).

`tests/test_guard_*.py` prove each refusal with a synthetic call.
"""

from __future__ import annotations

import builtins
import importlib.util
import ipaddress
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import types
import urllib.request
import warnings
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

TESTS_DIR = Path(__file__).resolve().parent
_STUBBED: dict[str, types.ModuleType] = {}  # the import-only stand-ins `_stub` registered: never "installed", never refused


def _absent(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is None
    except (ImportError, ValueError):
        return True


# Load only Qt's library before the process/keyring guards and platform stubs. PySide's import hook inspects those
# modules, and loading it under patched native dependencies can crash its Windows extension. No QApplication, window
# or audio is created here; widget tests explicitly use the offscreen platform.
if not _absent("PySide6"):
    importlib.import_module("PySide6.QtWidgets")


def _stub(name: str) -> types.ModuleType:
    """A module that can be imported and nothing else: every unset attribute raises."""
    module = types.ModuleType(name)

    def _getattr(attr: str) -> object:
        # Import machinery and PySide inspect module metadata; missing dunders follow normal lookup rules. Every
        # native API still raises.
        if attr.startswith("__") and attr.endswith("__"):
            raise AttributeError(attr)
        raise RuntimeError(f"{name}.{attr} is unavailable on this OS; tests must not call the platform adapter")

    module.__getattr__ = _getattr
    sys.modules[name] = module
    _STUBBED[name] = module
    return module


REAL_ACCESSIBILITY = not _absent("ApplicationServices")

if _absent("Quartz"):
    _stub("Quartz").kCGHIDEventTap = 0
if not REAL_ACCESSIBILITY:
    _stub("ApplicationServices")
if _absent("ocrmac"):

    class _OCR:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError("ocrmac is unavailable off macOS")

    _package = _stub("ocrmac")
    _package.__path__ = []
    _package.ocrmac = _stub("ocrmac.ocrmac")
    _package.ocrmac.OCR = _OCR
for _windows_module in ("psutil", "uiautomation", "win32api", "win32con", "win32gui", "win32process", "win32ui", "winocr"):
    if _absent(_windows_module):
        _stub(_windows_module)

import pytest  # noqa: E402
from PIL import Image  # noqa: E402

import glide  # noqa: E402
from glide.computer import macos, windows  # noqa: E402
from glide.computer.browser import cdp  # noqa: E402
from glide.computer.models import Item, Screen  # noqa: E402

# ------------------------------------------------------------------ the refusal ledger


def refusal(
    what: str, hint: str = "patch it in the test", kind: type[Exception] = RuntimeError, message: str | None = None
) -> Exception:
    """Record that something reached for the machine, and return the error to raise: `raise refusal("os.fork")`.

    Every refusal in the suite goes through here (conftest, every `tests/guards_*.py`), so the ledger below sees the ones
    production code swallows. `hint`, `kind` and `message` keep each guard's own wording and exception type."""
    (_LEDGERS[-1].fired if _LEDGERS else _OUTSIDE_A_TEST).append(what)
    return kind(message or f"a test reached the real machine through {what}; {hint}")


_LEDGERS: list[RefusalLedger] = []  # the ledgers of the running tests, innermost last (a nested pytest run adds one)
_OUTSIDE_A_TEST: list[str] = []  # refusals during collection, session fixtures or interpreter exit

# Tests whose whole job is to call a refused thing, in files this one does not own: each is named (a file, or a node id
# prefix that covers its parametrizations) so nothing else in those files is excused. Their owners should put
# `@pytest.mark.expect_refusals` on them; an entry is then dead weight and can go.
EXPECTS_REFUSALS: tuple[str, ...] = (
    "tests/test_core_guard.py",
    "tests/test_exec_backends_guard.py",
    "tests/test_execution_guard.py",
    "tests/test_no_real_machine.py",
    "tests/test_writer_boundary_guard.py",
    "tests/speech/test_voice_guards.py",
    "tests/test_native_observation.py::test_offline_guard_covers_native_foreground_entrypoints",
    "tests/test_browser_providers.py::test_cli_process_remains_guarded_offline",
    "tests/test_browser_providers.py::test_cdp_providers_remain_guarded_offline",
    "tests/test_tts.py::test_an_unpatched_say_is_stopped_by_the_machine_guard_not_turned_into_a_provider_error",
    "tests/mcp/test_mcp_transport.py::test_the_guard_refuses_asyncio_subprocesses",
    "tests/webhooks/test_webhooks_worker.py::test_the_real_executor_is_refused_by_the_guard",
)

# Tests that pass today only because a refusal was swallowed (a mock is missing and production code hid the error). One
# entry per node id, with the reason and the owner's fix. Nothing may be added without a reason; remove the entry when the
# test gets its mock. This is a debt ledger, not an allowlist to grow.
KNOWN_UNMOCKED: dict[str, str] = {}


class RefusalLedger:
    """A pytest plugin: fail, at teardown, any test during which a refusal fired that the test did not expect.

    `@pytest.mark.expect_refusals` expects any refusal; `expect_refusals("os.fork", "Popen")` expects only refusals whose
    description contains one of the fragments (a different one still fails the test)."""

    def __init__(self) -> None:
        self.fired: list[str] = []

    @pytest.hookimpl(tryfirst=True)
    def pytest_runtest_setup(self, item) -> None:
        self.fired = []
        _LEDGERS.append(self)

    @pytest.hookimpl(wrapper=True)
    def pytest_runtest_teardown(self, item, nextitem):
        try:
            result = yield
        finally:
            if self in _LEDGERS:
                _LEDGERS.remove(self)
            problem = self.unexpected(item)
        if problem:
            pytest.fail(problem, pytrace=False)
        return result

    def unexpected(self, item) -> str | None:
        if not self.fired:
            return None
        marker = item.get_closest_marker("expect_refusals")
        if marker is not None:
            fragments = marker.args
            stray = [what for what in self.fired if fragments and not any(fragment in what for fragment in fragments)]
        elif item.nodeid in KNOWN_UNMOCKED:
            stray = []
        else:
            stray = list(self.fired)
        if not stray:
            return None
        listed = "; ".join(dict.fromkeys(stray))
        return (
            f"a refusal fired during this test that it did not expect ({listed}). Production code may have swallowed the "
            "error and let the test pass on a fallback: patch the call in the test, or mark a test of the guard itself "
            "with @pytest.mark.expect_refusals."
        )

    def pytest_sessionfinish(self, session, exitstatus) -> None:
        """Say so when a refusal fired outside any test (collection, session fixtures), and fail a run that was green."""
        if not _OUTSIDE_A_TEST:
            return
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        line = f"refusals fired outside any test (collection or session fixtures): {'; '.join(_OUTSIDE_A_TEST)}"
        if reporter is not None:
            reporter.write_line(line, red=True)
        if not exitstatus:
            session.exitstatus = pytest.ExitCode.TESTS_FAILED


def pytest_collection_modifyitems(items):
    for item in items:
        if item.nodeid.startswith(EXPECTS_REFUSALS):
            item.add_marker(pytest.mark.expect_refusals)


def pytest_configure(config):
    """Register the ledger, every per-owner guard module `tests/guards_*.py` (an autouse fixture module each), and the
    session layer.

    The guard modules are loaded by path, since a name like `guards_core-run` is not an importable identifier. A new call
    that reaches the machine gets its refusal in its owner's guards file, in the same change that adds the call."""
    config.addinivalue_line(
        "markers",
        "expect_refusals(*fragments): the test fires the guard's refusals on purpose (every one, or only those whose "
        "description contains a fragment); any other refusal fails the test at teardown",
    )
    config.pluginmanager.register(RefusalLedger(), "refusal-ledger")
    basetemp = config.option.basetemp
    if basetemp:
        _EXTRA_TEMP_ROOTS.append(os.path.realpath(basetemp))
    for path in sorted(Path(__file__).parent.glob("guards_*.py")):
        spec = importlib.util.spec_from_file_location("tests_" + path.stem.replace("-", "_"), path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        config.pluginmanager.register(module, name=path.stem)
    _install_session_layer()


# ------------------------------------------------------------------ finding what is installed


_MISSING = object()


def _glide_module_names() -> list[str]:
    root = Path(glide.__file__).parent
    names = []
    for path in sorted(root.rglob("*.py")):
        parts = list(path.relative_to(root.parent).with_suffix("").parts)
        if parts[-1] == "__main__":
            continue
        names.append(".".join(parts[:-1] if parts[-1] == "__init__" else parts))
    return names


def _import_quiet(name: str) -> types.ModuleType | None:
    """The module, or None when it is not installed, cannot load here (a missing native library), or is one of this
    file's import-only stand-ins. Modules a test has put in `sys.modules` count as installed."""
    module = sys.modules.get(name)
    if module is not None:
        return None if _STUBBED.get(name) is module else module
    if name in _NOT_IMPORTABLE:
        return None
    try:
        if importlib.util.find_spec(name) is None:
            _NOT_IMPORTABLE.add(name)
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # deprecated aliases (websockets.legacy) warn on import
            return importlib.import_module(name)
    except Exception:
        _NOT_IMPORTABLE.add(name)
        return None


_NOT_IMPORTABLE: set[str] = set()

# Every module of the package is imported now, before any guard is installed and before the first test. A module first
# imported inside a test would otherwise copy the refusals, not the real functions, into its own namespace for the
# rest of the session. A module that cannot import is left for the test that needs it to report.
for _name in _glide_module_names():
    _import_quiet(_name)


def _is_real(module: types.ModuleType) -> bool:
    """Whether `module` is the installed thing, as opposed to a hand-made stand-in a test put in `sys.modules` or a
    namespace directory that happens to share the name (`tests/mcp` when the MCP SDK is absent)."""
    spec = vars(module).get("__spec__")
    if spec is None or spec.origin in (None, "namespace"):
        return False
    places = [spec.origin if spec.origin not in ("built-in", "frozen") else None, *(spec.submodule_search_locations or ())]
    return not any(place and Path(place).resolve().is_relative_to(TESTS_DIR) for place in places)


class _Refused:
    """Stands in for something that reaches the machine. Calling it, or using any attribute of it, refuses, so it is
    equally good for a function, a class, or a whole namespace (`AppKit.NSWorkspace`)."""

    def __init__(self, what: str) -> None:
        self._what = what

    def _refuse(self):
        raise refusal(self._what)

    def __call__(self, *args, **kwargs):
        self._refuse()

    def __getattr__(self, attr: str):
        if attr.startswith("__") and attr.endswith("__"):
            raise AttributeError(attr)
        self._refuse()

    def __repr__(self) -> str:
        return f"<refused: {self._what}>"


def _probe(owner: object, name: str) -> object:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # a lazy alias that imports a deprecated module warns on first access
            return getattr(owner, name)
    except Exception:  # absent, or a stand-in module that refuses every unknown attribute
        return _MISSING


# ------------------------------------------------------------------ refusing a name and every copy of it

_ALIASES: dict[int, list[tuple[object, types.ModuleType, str]]] = {}
_SCANNED: dict[str, types.ModuleType] = {}


def _owned_by_tests(module: types.ModuleType) -> bool:
    file = vars(module).get("__file__")  # not getattr: a stand-in module raises on unknown attributes
    try:
        return isinstance(file, str) and Path(file).resolve().is_relative_to(TESTS_DIR)
    except (OSError, ValueError):
        return False


def _scan_modules() -> None:
    """Index, by object id, which module attributes hold which callable: the copies `from x import y` made at import time.

    Incremental: a module is read once (again if its name is bound to a different module), so the first call after
    collection reads everything already imported and each later call only what is new. Test modules are left out on purpose:
    they keep real references (`SCREENSHOT = macos.screenshot`) and patch what the body calls."""
    for name, module in list(sys.modules.items()):
        if not isinstance(module, types.ModuleType) or _STUBBED.get(name) is module or _SCANNED.get(name) is module:
            continue
        _SCANNED[name] = module
        namespace = getattr(module, "__dict__", None)
        if not isinstance(namespace, dict) or _owned_by_tests(module):
            continue
        for attr, value in list(namespace.items()):
            if callable(value) and not (attr.startswith("__") and attr.endswith("__")):
                _ALIASES.setdefault(id(value), []).append((value, module, attr))


def _is_home(module: types.ModuleType, thing: object) -> bool:
    """Whether `module` is where `thing` is defined (and not a built-in one that merely re-exports it, like `posix`).

    A home module keeps its own name for the real thing: `websocket._core.create_connection` is where a test that needs the
    real function goes to fetch it (`test_stt` patches it back from there), while every module that imported it is refused."""
    spec = vars(module).get("__spec__")
    return (
        getattr(thing, "__module__", None) == module.__name__
        and spec is not None
        and spec.origin not in (None, "built-in", "frozen")
    )


class Guard:
    """One application of every refusal to one test, undone with its `monkeypatch`."""

    def __init__(self, monkeypatch) -> None:
        self.monkeypatch = monkeypatch
        self.applied: list[tuple[object, str, str]] = []  # (owner, name, what) of every refusal, for the tests of the guard
        self._scanned = (
            False  # the module index is brought up to date once per guard: a pass over sys.modules costs a millisecond
        )

    def replace(self, owner: object, name: str, new: object) -> bool:
        """Put `new` at `owner.name` and in every already-imported module that holds the same object. False when absent."""
        original = _probe(owner, name)
        if original is _MISSING:
            return False
        self.monkeypatch.setattr(owner, name, new)
        if isinstance(owner, types.ModuleType) and callable(original):
            if not self._scanned:
                _scan_modules()
                self._scanned = True
            for held, module, attr in _ALIASES.get(id(original), ()):
                if (
                    held is original
                    and (module is not owner or attr != name)
                    and vars(module).get(attr) is original
                    and not _is_home(module, original)
                ):
                    self.monkeypatch.setattr(module, attr, new)
        return True

    def ban(self, owner: object, name: str, what: str) -> bool:
        """Refuse `owner.name`, and every module's copy of it. False when it is not there."""
        if not self.replace(owner, name, _Refused(what)):
            return False
        self.applied.append((owner, name, what))
        return True

    def strict(self, owner: object, name: str, what: str) -> None:
        """A refusal for code that exists now: a rename must not silently drop it."""
        if not self.ban(owner, name, what):
            raise AttributeError(f"the guard in conftest.py refuses {what}, which no longer exists; update the guard")

    def table(self, rows: tuple[tuple[str, tuple[str, ...]], ...] | None = None) -> None:
        """Every row of REFUSALS (or `rows`) whose module is installed. A name an installed module lacks is an error."""
        done: set[tuple[int, str]] = set()
        for spec, paths in REFUSALS if rows is None else rows:
            for module in _modules(spec):
                for path in paths:
                    for owner, name, dotted in resolve_row(spec, module, path):
                        if (id(owner), name) not in done:
                            done.add((id(owner), name))
                            self.ban(owner, name, f"{module.__name__}.{dotted}")


def _modules(spec: str) -> list[types.ModuleType]:
    """The modules a REFUSALS row names. `glide:tail` matches every `glide` module whose dotted name ends in `tail`, so
    a module may move between packages without losing its guard; anything else is an exact import name."""
    if not spec.startswith("glide:"):
        module = _import_quiet(spec)
        return [] if module is None else [module]
    tail = spec.removeprefix("glide:")
    found = [
        module
        for name, module in list(sys.modules.items())
        if module is not None and name.startswith("glide.") and (name == f"glide.{tail}" or name.endswith(f".{tail}"))
    ]
    return sorted(found, key=lambda module: module.__name__)


_FAMILIES: dict[tuple[int, str], tuple[object, list[str]]] = {}


def _family(owner: object, prefix: str) -> list[str]:
    """The callables of `owner` whose names start with `prefix`. A framework has tens of thousands of names, so each
    family is read once per session."""
    cached = _FAMILIES.get((id(owner), prefix))
    if cached is None or cached[0] is not owner:
        cached = (owner, [n for n in dir(owner) if n.startswith(prefix) and callable(_probe(owner, n))])
        _FAMILIES[(id(owner), prefix)] = cached
    return cached[1]


def resolve_row(spec: str, module: types.ModuleType, path: str) -> list[tuple[object, str, str]]:
    """The (owner, attribute, dotted name) triples one path of a REFUSALS row stands for in `module`.

    A path is a dotted name that may reach into a class (`Server.run`). Its last part may be `prefix*` (every callable
    whose name starts with it: a framework's whole family of reads, so a new call in it is refused without a new row) or
    `a|b` (alternatives that differ between versions: those present are refused, and at least one must be). A leading `?`
    marks a name that may legitimately be absent. In an installed module (`_is_real`) anything else absent is an
    AttributeError, so a row can never go dead after a rename; a stand-in module's missing names are skipped."""
    optional = path.startswith("?")
    *chain, last = path.lstrip("?").split(".")
    owner: object = module
    for part in chain:
        owner = _probe(owner, part)
        if owner is _MISSING:
            break
    names: list[str] = []
    if owner is not _MISSING:
        if last.endswith("*"):
            names = _family(owner, last[:-1])
        else:
            names = [n for n in last.split("|") if _probe(owner, n) is not _MISSING]
    if not names and not optional and _is_real(module):
        raise AttributeError(
            f"the REFUSALS row ({spec!r}, {path!r}) names something {module.__name__} does not have; it would refuse "
            "nothing. Fix the row in tests/conftest.py to the real name (or mark it optional with a leading '?')."
        )
    return [(owner, name, ".".join([*chain, name])) for name in names]


# ------------------------------------------------------------------ the table


# Refused for the whole session as well as in each test (`_install_session_layer`): starting a process and posting an input
# event are the two things that must never happen, whatever fixture or import is running.
SESSION_ROWS: tuple[tuple[str, tuple[str, ...]], ...] = (
    # `subprocess` ends in `_posixsubprocess.fork_exec`, and so does `multiprocessing.util.spawnv_passfds` (the resource
    # tracker, the spawn and forkserver starts) without going through `Popen` at all.
    ("_posixsubprocess", ("fork_exec",)),
    ("multiprocessing.util", ("spawnv_passfds",)),
    ("multiprocessing.process", ("BaseProcess.start",)),
    ("Quartz", ("CGEventPost", "CGEventPostToPid", "CGEventTapCreate", "CGWarpMouseCursorPosition")),
    ("ApplicationServices", ("AXUIElementPerformAction", "AXUIElementSetAttributeValue", "AXUIElementPostKeyboardEvent")),
    ("glide:macos", ("_post",)),
    ("glide:windows", ("_send", "_move")),
    ("win32api", ("mouse_event", "keybd_event", "SetCursorPos")),
)

# Targets that exist once a package is installed or a module is ported. (module, attribute paths); see `resolve_row` for
# the path syntax. A row is skipped when its module is not installed or is a stand-in, and fails every test when its
# module is installed but lacks a name it lists. Add a row in the commit that adds the call.
REFUSALS: tuple[tuple[str, tuple[str, ...]], ...] = (
    *SESSION_ROWS,
    # --- the platform adapters: what the macOS and Windows variants added beyond the strict list in `no_real_machine`
    (
        "glide:macos",
        (
            "frontmost_app",
            "frontmost_app_and_pid",
            "frontmost_pid",
            "screen_capture_trusted",
            "request_permissions",
            "execution_tabs",
            "execution_tab",
            "execution_shortcut",
            "execution_scrolls",
            "execution_scroll",
            "execution_labels",
        ),
    ),
    (
        "glide:windows",
        (
            "frontmost_app",
            "frontmost_app_and_pid",
            "frontmost_pid",
            "request_permissions",
            "execution_tabs",
            "execution_tab",
            "execution_shortcut",
            "execution_scrolls",
            "execution_scroll",
            "execution_labels",
        ),
    ),
    # --- macOS frameworks called directly instead of through the adapter. Every call `glide/computer/macos.py` makes that
    # reads or acts on this machine is named; the families after them refuse the neighbours a port would reach for. What
    # stays live is pure: building an event (`CGEventCreateMouseEvent` ...) posts nothing, and `AXValueGetValue` only
    # unpacks a value. `test_guard_native` fails if macos.py grows a call that is neither refused nor on its allowlist.
    (
        "Quartz",
        (
            "CGEventCreate",  # reads the pointer: `mouse_location`
            "CGMainDisplayID",
            "CGDisplayBounds",
            "CGWindowListCopyWindowInfo",
            "CGPreflightScreenCaptureAccess",
            "CGRequestScreenCaptureAccess",
            "CGWindowListCreateImage",
            "CGDisplayCreateImage",
            "CGDisplayCreateImageForRect",
            "CGDisplayMoveCursorToPoint",
            "CGRequestPostEventAccess",
            "CGRequestListenEventAccess",
            "CGDisplay*",
            "CGWindowList*",
            "CGGet*",
            "CGEventSource*",
            "CGEventTap*",
            "CGEventPost*",
            "CGPreflight*",
            "CGRequest*",
            "CGAssociateMouseAndMouseCursorPosition",
            "CGSessionCopyCurrentDictionary",
            "CGSetLocalEventsSuppressionInterval",
            "CGCapture*",
        ),
    ),
    (
        "ApplicationServices",
        (
            "AXIsProcessTrusted",
            "AXIsProcessTrustedWithOptions",
            "AXUIElementCreateSystemWide",
            "AXUIElementCreateApplication",
            "AXUIElementCopyAttributeValue",
            "AXUIElementCopyActionNames",
            "AXUIElementSetMessagingTimeout",
            "AXUIElement*",
            "AXObserver*",
        ),
    ),
    (
        "AppKit",
        (
            "NSWorkspace",
            "NSRunningApplication",
            "NSPasteboard",
            "NSAppleScript",
            "NSScreen",
            "NSEvent",
            "NSCursor",
            "NSSound",
            "NSSpeechSynthesizer",
            "NSTask",
            "NSURLSession",
            "NSURLConnection",
        ),
    ),
    (
        "Cocoa",
        ("NSWorkspace", "NSRunningApplication", "NSPasteboard", "NSAppleScript", "NSScreen", "NSEvent", "NSCursor", "NSTask"),
    ),
    ("Foundation", ("NSAppleScript", "NSTask", "NSURLSession", "NSURLConnection")),
    # --- Vision OCR (macos.recognize_text goes through ocrmac)
    ("Vision", ("VNImageRequestHandler", "VNRecognizeTextRequest", "VNSequenceRequestHandler")),
    ("ocrmac.ocrmac", ("OCR", "text_from_image", "livetext_from_image")),
    # --- Windows APIs called directly (a stand-in module refuses everything already and is skipped)
    ("win32api", ("GetCursorPos", "GetSystemMetrics")),
    (
        "win32gui",
        (
            "SetForegroundWindow",
            "ShowWindow",
            "BringWindowToTop",
            "PostMessage",
            "SendMessage",
            "EnumWindows",
            "GetForegroundWindow",
            "GetWindowRect",
            "GetWindowText",
            "IsWindowVisible",
            "CreateDC",
            "DeleteDC",
            "DeleteObject",
        ),
    ),
    ("uiautomation", ("GetFocusedControl", "GetRootControl", "ControlFromHandle", "ControlFromPoint")),
    ("winocr", ("recognize_pil_sync",)),
    # --- screen capture and input libraries a port might reach for
    ("PIL.ImageGrab", ("grab", "grabclipboard")),
    (
        "pyautogui",
        (
            "click",
            "doubleClick",
            "rightClick",
            "moveTo",
            "moveRel",
            "dragTo",
            "dragRel",
            "press",
            "hotkey",
            "keyDown",
            "keyUp",
            "typewrite",
            "write",
            "scroll",
            "mouseDown",
            "mouseUp",
            "screenshot",
        ),
    ),
    ("pynput.mouse", ("Controller", "Listener")),
    ("pynput.keyboard", ("Controller", "Listener")),
    ("mss", ("mss", "MSS")),
    ("pyperclip", ("copy", "paste")),
    # --- the browser: opening a URL in the person's browser
    ("webbrowser", ("open", "open_new", "open_new_tab", "get")),
    # --- the execution engine's approved-browser HTTP endpoint (bound at import time, past the cdp guard)
    ("glide:execution.dom", ("_get_json",)),
    # --- microphone and speaker. The streams are where a sound device is reached, so they are what is refused. Nothing in
    # `glide` is refused above them: `FullDuplexDevice.start` runs with injected stream factories all over the speech tests,
    # and `sounddevice_factories` only returns functions (refusing it would turn "the extra is not installed" into a refusal).
    (
        "sounddevice",
        (
            "InputStream",
            "OutputStream",
            "Stream",
            "RawInputStream",
            "RawOutputStream",
            "RawStream",
            "play",
            "rec",
            "playrec",
            "query_devices",
            "query_hostapis",
        ),
    ),
    # --- realtime websockets (the loopback rule would let a test reach a server on this machine)
    ("websockets", ("connect", "unix_connect")),
    ("websockets.sync.client", ("connect", "unix_connect")),
    ("websockets.asyncio.client", ("connect", "unix_connect")),
    ("websockets.legacy.client", ("connect", "unix_connect")),
    # --- listeners: a test never binds a service (a listener on a non-loopback address is refused at the socket)
    ("uvicorn", ("run",)),
    ("uvicorn.main", ("run",)),
    ("uvicorn.server", ("Server.run", "Server.serve", "Server.startup")),
    ("glide:webhooks.cli", ("serve",)),
    ("glide:webhooks.worker", ("ComputerExecutor.execute",)),
    # --- MCP: stdio spawns a server process or takes over this process's stdin and stdout, the rest open connections
    ("mcp.client.stdio", ("stdio_client",)),
    ("mcp.server.stdio", ("stdio_server",)),
    ("mcp.client.sse", ("sse_client",)),
    ("mcp.client.streamable_http", ("streamablehttp_client|streamable_http_client",)),
    ("mcp.client.websocket", ("websocket_client",)),
    (
        "mcp.server.fastmcp",
        ("FastMCP.run", "FastMCP.run_stdio_async", "FastMCP.run_sse_async", "FastMCP.run_streamable_http_async"),
    ),
    # --- the OS keyring and the code that wraps it
    ("keyring", ("get_password", "set_password", "delete_password", "get_credential")),
    ("keyring.core", ("get_password", "set_password", "delete_password", "get_credential")),
    ("keyring.backends.macOS", ("Keyring.get_password", "Keyring.set_password", "Keyring.delete_password")),
    ("glide:credentials", ("_native_key",)),
    # --- the pet: processes and native windows
    ("glide:pet", ("open_window", "show_point_marker", "dismiss_point_marker")),
    ("glide:pet_overlay", ("configure_overlay", "configure_panel")),
)

# `glide:` rows for modules that are not in this tree yet. A row for any other `glide` module that matches nothing is a
# dead row (`test_guard_table` fails it); an entry here that starts matching must be removed, so the row is enforced.
UNPORTED = ("credentials",)

# os functions that start, replace or signal a process. `subprocess.Popen` is refused by name in the fixture, which also
# covers subprocess.run, asyncio's subprocess functions and os.popen, because they all end in it.
_PROCESS_CALLS = (
    "fork",
    "forkpty",
    "execl",
    "execle",
    "execlp",
    "execlpe",
    "execv",
    "execve",
    "execvp",
    "execvpe",
    "spawnl",
    "spawnle",
    "spawnlp",
    "spawnlpe",
    "spawnv",
    "spawnve",
    "spawnvp",
    "spawnvpe",
    "popen",
    "startfile",
    "killpg",
    "system",
    "posix_spawn",
    "posix_spawnp",
)


# ------------------------------------------------------------------ the session layer


def _refusing_popen_init(self, *args, **kwargs):
    raise refusal("subprocess.Popen")


_SESSION_LAYER: list[Guard] = []


def _install_session_layer() -> None:
    """Refuse process starts and input events for the whole session: collection, session and module fixtures, and
    interpreter exit, where no per-test monkeypatch exists. Installed once, by `pytest_configure`, and never undone (an
    atexit handler must not be able to start a process either). `Popen` keeps its class and loses its `__init__`, so
    imports that subclass or annotate with it still work. The per-test layer adds the rest and the by-name messages."""
    if _SESSION_LAYER:
        return
    patch = pytest.MonkeyPatch()  # deliberately never undone
    guard = Guard(patch)
    patch.setattr(subprocess.Popen, "__init__", _refusing_popen_init)
    for name in _PROCESS_CALLS:
        guard.ban(os, name, f"os.{name}")
    guard.table(SESSION_ROWS)
    _SESSION_LAYER.append(guard)


# ------------------------------------------------------------------ the per-test layer

_EXTRA_TEMP_ROOTS: list[str] = []


def _temp_roots() -> list[str]:
    return [os.path.realpath(tempfile.gettempdir()), *_EXTRA_TEMP_ROOTS]


def local_socket_path(path: object) -> bool:
    """A unix socket path inside the temp directory: one a test created itself, never a service of the person's
    (the Glide app, an agent, the container runtime)."""
    if isinstance(path, (bytes, bytearray, os.PathLike)):
        path = os.fsdecode(path)
    if not isinstance(path, str) or not path or path.startswith("\0"):
        return False
    real = os.path.realpath(path)
    return any(real == root or real.startswith(root + os.sep) for root in _temp_roots())


@pytest.fixture(autouse=True)
def no_real_machine(monkeypatch):
    """No test reaches the computer it runs on, whoever wrote it.

    The suite runs on the developer's own Mac, often while they use it. Every call that would
    move the pointer, press a key, run AppleScript (which opens apps and URLs), capture the
    screen, open a file, or act on another app's accessibility element refuses here, so a test
    that forgot to patch one fails instead of taking over the machine. A test that needs one
    patches it itself, after this. The pointer reads as mid-screen, never the abort corner.
    """
    guard = Guard(monkeypatch)

    for name in ("_post", "osascript", "screenshot", "open_path"):
        guard.strict(macos, name, f"macos.{name}")
    monkeypatch.setattr(macos, "mouse_location", lambda: (500.0, 500.0))
    # The Windows adapter: SendInput and the cursor carry all input; the rest launch, activate,
    # open, capture, or act on another app's element.
    for name in ("_send", "_move", "screenshot", "activate", "open_url", "open_path", "ax_press", "ax_focus", "ax_set_value"):
        guard.strict(windows, name, f"windows.{name}")
    monkeypatch.setattr(windows, "mouse_location", lambda: (500.0, 500.0))

    # The browser backend: no Chrome and no process of any kind, nothing over CDP, and no
    # connection except to a server on this machine that the test started itself.
    guard.strict(subprocess, "Popen", "subprocess.Popen")
    guard.strict(cdp, "find_chrome", "cdp.find_chrome")
    guard.strict(cdp, "_get_json", "the CDP HTTP endpoint")
    guard.strict(cdp.websocket, "create_connection", "a CDP websocket")
    for name in _PROCESS_CALLS:
        guard.ban(os, name, f"os.{name}")
    _only_own_process(guard)
    _loopback_only(guard)
    _no_writes_outside_temp(guard)
    guard.table()


def apply_refusals(monkeypatch) -> Guard:
    """The table again, for a test that has just put a module in `sys.modules`."""
    guard = Guard(monkeypatch)
    guard.table()
    return guard


def _only_own_process(guard: Guard) -> None:
    """A test may signal itself, and probe whether a pid exists (signal 0), and nothing else."""
    real_kill = os.kill

    def kill(pid, sig):
        if pid != os.getpid() and sig != 0:
            raise refusal(f"os.kill({pid!r}, {sig!r}) of another process")
        return real_kill(pid, sig)

    guard.replace(os, "kill", kill)


# ------------------------------------------------------------------ sockets


def _is_loopback(host: object) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(str(host).split("%")[0]).is_loopback
    except ValueError:
        return False


class _Listeners:
    """What this test bound itself: the loopback ports and the unix socket paths it may connect (or send) to."""

    def __init__(self) -> None:
        self.ports: set[int] = set()
        self.paths: set[str] = set()

    def add(self, address: object) -> None:
        if isinstance(address, (str, bytes, os.PathLike)):
            text = os.fsdecode(address)
            if text and not text.startswith("\0"):
                self.paths.add(os.path.realpath(text))
        elif isinstance(address, tuple) and len(address) >= 2:
            self.ports.add(int(address[1]))

    def reaches(self, family: int, address: object) -> bool:
        if family == getattr(socket, "AF_UNIX", None):
            return local_socket_path(address) and os.path.realpath(os.fsdecode(address)) in self.paths
        return _is_loopback(address[0]) and address[1] in self.ports


_LISTENERS: list[_Listeners] = []


def allow_connect(address: object) -> None:
    """Let this test connect to `address`: a loopback `(host, port)` or a unix socket path in the temp directory that it
    did not bind itself in this process (a server in another thread's helper that bound before the test began, a port a
    fixture hands over). Everything the test binds is allowed already; this is the explicit door for fakes the guard cannot
    see. A non-loopback address is still refused at the socket."""
    if not _LISTENERS:
        raise RuntimeError("allow_connect needs the guard's fixture: call it from a test")
    _LISTENERS[-1].add(address)


def _loopback_only(guard: Guard) -> None:
    """A test reaches only sockets it made itself: a loopback port or a temp-directory unix socket that this test bound
    (the `endpoint` fixture, a fake it starts), whatever the address spelling (`localhost` may try `::1` first, so the
    host is any loopback host and the port is what is matched). Another service of the person's on the same loopback
    interface, or under the temp directory, is refused like the network is. A name other than localhost is not even looked
    up. Binding is held to the same line, so a test never exposes a listener to the network. `socketpair` makes connected
    sockets without a connect call (POSIX), and its fallback (Windows) binds a loopback listener, so both just work."""
    own = _Listeners()
    guard.monkeypatch.setitem(globals(), "_LISTENERS", [own])  # this test's, undone with it
    real_connect, real_connect_ex, real_getaddrinfo = socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo
    real_bind, real_sendto, real_sendmsg = socket.socket.bind, socket.socket.sendto, socket.socket.sendmsg
    inet = (socket.AF_INET, socket.AF_INET6)
    unix = getattr(socket, "AF_UNIX", None)

    def remote(sock: socket.socket, address: object) -> bool:
        if sock.family == unix or sock.family in inet:
            return not own.reaches(sock.family, address)
        return False

    def connect(sock, address):
        if remote(sock, address):
            raise refusal(f"a connection to {address!r}", "the test did not bind it")
        return real_connect(sock, address)

    def connect_ex(sock, address):
        if remote(sock, address):
            raise refusal(f"a connection to {address!r}", "the test did not bind it")
        return real_connect_ex(sock, address)

    def getaddrinfo(host, *args, **kwargs):
        if host is not None and not _is_loopback(host.decode() if isinstance(host, bytes) else host):
            raise refusal(f"a lookup of {host!r}")
        return real_getaddrinfo(host, *args, **kwargs)

    def bind(sock, address):
        if sock.family == unix:
            if address and not local_socket_path(address):
                raise refusal(f"binding the unix socket {address!r}")
        elif sock.family in inet and not _is_loopback(address[0]):
            raise refusal(f"binding a listener to {address!r}")
        result = real_bind(sock, address)
        if sock.family == unix or sock.family in inet:
            own.add(sock.getsockname())  # the real port: a bind to port 0 is only known now
        return result

    def sendto(sock, data, *rest):
        if rest and remote(sock, rest[-1]):
            raise refusal(f"a datagram to {rest[-1]!r}", "the test did not bind it")
        return real_sendto(sock, data, *rest)

    def sendmsg(sock, buffers, *rest, **kwargs):
        address = kwargs.get("address", rest[2] if len(rest) > 2 else None)
        if address and remote(sock, address):
            raise refusal(f"a datagram to {address!r}", "the test did not bind it")
        return real_sendmsg(sock, buffers, *rest, **kwargs)

    patch = guard.monkeypatch
    patch.setattr(socket.socket, "connect", connect)
    patch.setattr(socket.socket, "connect_ex", connect_ex)
    patch.setattr(socket.socket, "bind", bind)
    patch.setattr(socket.socket, "sendto", sendto)
    patch.setattr(socket.socket, "sendmsg", sendmsg)
    guard.replace(socket, "getaddrinfo", getaddrinfo)
    # The older resolver calls do not go through getaddrinfo. This machine's own name stays resolvable
    # (`socket.getfqdn()` asks for it, as does `email.utils.make_msgid`).
    own_names = {"", socket.gethostname()}

    def lookup(real):
        def resolve(host, *args, **kwargs):
            if host not in own_names and not _is_loopback(host):
                raise refusal(f"a lookup of {host!r}")
            return real(host, *args, **kwargs)

        return resolve

    for name in ("gethostbyname", "gethostbyname_ex", "gethostbyaddr"):
        guard.replace(socket, name, lookup(getattr(socket, name)))
    real_getnameinfo = socket.getnameinfo

    def getnameinfo(sockaddr, flags):
        if not _is_loopback(sockaddr[0]):
            raise refusal(f"a reverse lookup of {sockaddr[0]!r}")
        return real_getnameinfo(sockaddr, flags)

    guard.replace(socket, "getnameinfo", getnameinfo)
    _no_proxy(patch)


def _no_proxy(monkeypatch) -> None:
    """A proxy named in the environment is a loopback hop that carries the request anywhere, past the
    loopback rule above. No test goes through one: the settings are dropped and `urllib` is told there are none."""
    for name in ("http_proxy", "https_proxy", "all_proxy", "ftp_proxy"):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.upper(), raising=False)
    monkeypatch.setenv("NO_PROXY", "*")
    monkeypatch.setenv("no_proxy", "*")
    monkeypatch.setattr(urllib.request, "getproxies", dict)


# ------------------------------------------------------------------ files

# Where a test may open a file for writing: the temp directory (and pytest's own base temp), /dev/null, and the tools'
# cache directories inside the repository. Anything else (the repository itself, the person's home, a relative path under the
# working directory) is refused: a test writes under `tmp_path`.
WRITE_CACHE_DIRS = (".pytest_cache", "__pycache__", ".ruff_cache", ".mypy_cache", ".hypothesis")


def writable_place(path: object) -> bool:
    """Whether a test may open `path` for writing."""
    if isinstance(path, int):  # a file descriptor the caller already holds
        return True
    try:
        real = os.path.realpath(os.fsdecode(path))
    except (TypeError, ValueError):
        return False
    if real == os.devnull or any(real == root or real.startswith(root + os.sep) for root in _temp_roots()):
        return True
    return real.startswith(str(TESTS_DIR.parent) + os.sep) and any(part in WRITE_CACHE_DIRS for part in Path(real).parts)


REAL_OPEN = builtins.open


def _no_writes_outside_temp(guard: Guard) -> None:
    """`open(..., 'w'/'a'/'x'/'+')` and `Path.write_text`/`write_bytes`/`Path.open` (they all end in `io.open`) refuse
    outside `writable_place`. `builtins.open` and `io.open` are one object, so the replacement reaches both and every
    module that imported it. Not covered: `os.open`, `sqlite3`, `shutil` copies of directories, `os.rename` and friends."""
    real_open = REAL_OPEN

    def guarded_open(file, mode="r", *args, **kwargs):
        if isinstance(mode, str) and any(flag in mode for flag in "wax+") and not writable_place(file):
            raise refusal(f"opening {os.fsdecode(file)!r} for writing", "a test writes under tmp_path")
        return real_open(file, mode, *args, **kwargs)

    guard.replace(builtins, "open", guarded_open)


@pytest.fixture
def screen() -> Screen:
    return Screen(image=Image.new("RGB", (2000, 1200)), scale=2.0, app="Google Chrome", field=None, url=None)


def busy_page(size: tuple[int, int]) -> Image.Image:
    """A white page ruled with a gray line every third row: something is drawn under every control on
    it, so none reads as hidden (see `perception.drawn`)."""
    image = Image.new("RGB", size, "white")
    for y in range(0, size[1], 3):
        image.paste((180, 180, 180), (0, y, size[0], y + 1))
    return image


def item(index: int, text: str, x1=100, y1=100, x2=400, y2=130, conf=1.0) -> Item:
    return Item(index, text, conf, x1, y1, x2, y2)


@pytest.fixture
def make_item():
    return item


@pytest.fixture
def tmp_env(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("GLIDE_TEST_KEY", raising=False)
    return tmp_path


WRITER_ENV = (
    "GLIDE_WRITER_VISION",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_PROFILE",
)


@pytest.fixture
def clean_env(monkeypatch):
    """No writer configuration from the shell running the tests."""
    for name in WRITER_ENV:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


@pytest.fixture
def endpoint():
    """A writer endpoint on localhost, as a proxy or a local model would serve one.

    It answers the Anthropic Messages API or OpenAI's Chat Completions API by path, records each
    request with its headers, and replies with `state["reply"]`. `state["reject"]` may return an
    error message for a request body, which then gets a 400, the way an endpoint refuses a parameter.
    `state["usage"]`, when set, is the Chat Completions reply's `usage`; without it there is none.
    """
    seen: list[dict] = []
    state: dict = {"reply": '{"ok": true, "url": "https://example.com", "reason": ""}', "reject": lambda body: None}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append({"path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()}, "body": body})
            refusal = state["reject"](body)
            if refusal:
                status, out = 400, {"type": "error", "error": {"type": "invalid_request_error", "message": refusal}}
            elif self.path.endswith("/chat/completions"):
                status, out = (
                    200,
                    {
                        "id": "chatcmpl-1",
                        "object": "chat.completion",
                        "created": 0,
                        "model": body["model"],
                        "choices": [
                            {"index": 0, "message": {"role": "assistant", "content": state["reply"]}, "finish_reason": "stop"}
                        ],
                        **({"usage": state["usage"]} if "usage" in state else {}),
                    },
                )
            else:
                status, out = (
                    200,
                    {
                        "id": "msg_1",
                        "type": "message",
                        "role": "assistant",
                        "model": body["model"],
                        "content": [{"type": "text", "text": state["reply"]}],
                        "stop_reason": "end_turn",
                        "stop_sequence": None,
                        "usage": {"input_tokens": 1, "output_tokens": 1},
                    },
                )
            data = json.dumps(out).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
    yield SimpleNamespace(url=f"http://127.0.0.1:{server.server_port}", seen=seen, state=state)
    server.shutdown()
    server.server_close()

"""Shared fixtures, a guard that keeps every test off the real machine, and import-only
stand-ins for the platform-only modules.

The suite is pure logic and should run on any OS. The platform adapters
(`glide.computer.macos`, `glide.computer.windows`) import their
platform's packages at module scope, but the tests only ever import them -- they
call nothing but the pure rules -- so a stand-in that exists and raises on any
real use is enough to run the whole suite anywhere. Where a real module is
installed, nothing is registered for it.

The guard has three layers (see `no_real_machine`):

1. Refusals written out by name for code that exists in the tree today (the macOS and Windows adapters, the
   browser backend, processes, sockets). These fail loudly if the target is renamed away.
2. `REFUSALS`, a table of targets that only exist once a module is ported or a package is installed
   (`frontmost_app`, `sounddevice`, `websockets`, `uvicorn`, `mcp` stdio, `keyring`, the pet, webhooks, ...). A row
   applies when its module imports and has the attribute, and is skipped without failure otherwise, so a port cannot
   forget it. A change that adds a call reaching the machine adds its row here in the same commit.
3. Alias rebinding: `from glide.x import y` copies a reference at import time, so refusing `glide.x.y` alone leaves
   the copy live (that is how `dom._get_json` escaped the `cdp._get_json` guard). Every refusal of a function or class
   also replaces the same object wherever a `glide` module holds it.

`tests/test_guard_*.py` prove each refusal with a synthetic call.
"""

from __future__ import annotations

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


def pytest_configure(config):
    """Register every per-owner guard module `tests/guards_*.py` (an autouse fixture module each).

    Loaded by path, since a name like `guards_core-run` is not an importable identifier. A new call that
    reaches the machine gets its refusal in its owner's guards file, in the same change that adds the call.
    """
    for path in sorted(Path(__file__).parent.glob("guards_*.py")):
        spec = importlib.util.spec_from_file_location("tests_" + path.stem.replace("-", "_"), path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        config.pluginmanager.register(module, name=path.stem)


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
    """The module, or None when it is not installed, cannot load here (a missing native library), or is a
    stand-in that refuses. Modules a test has put in `sys.modules` count as installed."""
    module = sys.modules.get(name)
    if module is not None:
        return module
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


class _Refused:
    """Stands in for something that reaches the machine. Calling it, or using any attribute of it, refuses, so it is
    equally good for a function, a class, or a whole namespace (`AppKit.NSWorkspace`)."""

    def __init__(self, what: str) -> None:
        self._what = what

    def _refuse(self):
        raise RuntimeError(f"a test reached the real machine through {self._what}; patch it in the test")

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


def _glide_aliases() -> dict[int, list[tuple[types.ModuleType, str]]]:
    """Where each callable is held by a `glide` module, by id: the copies `from x import y` makes at import time."""
    index: dict[int, list[tuple[types.ModuleType, str]]] = {}
    for name, module in list(sys.modules.items()):
        if module is None or not (name == "glide" or name.startswith("glide.")):
            continue
        for attr, value in list(vars(module).items()):
            if callable(value) and not (attr.startswith("__") and attr.endswith("__")):
                index.setdefault(id(value), []).append((module, attr))
    return index


class Guard:
    """One application of every refusal to one test, undone with its `monkeypatch`."""

    def __init__(self, monkeypatch) -> None:
        self.monkeypatch = monkeypatch
        self._aliases: dict[int, list[tuple[types.ModuleType, str]]] | None = None

    def ban(self, owner: object, name: str, what: str) -> bool:
        """Refuse `owner.name`, and every `glide` module's copy of it. False when it is not there."""
        original = _probe(owner, name)
        if original is _MISSING:
            return False
        refused = _Refused(what)
        self.monkeypatch.setattr(owner, name, refused)
        if isinstance(owner, types.ModuleType) and callable(original):
            if self._aliases is None:
                self._aliases = _glide_aliases()
            for module, attr in self._aliases.get(id(original), ()):
                if module is not owner or attr != name:
                    self.monkeypatch.setattr(module, attr, refused)
        return True

    def strict(self, owner: object, name: str, what: str) -> None:
        """A refusal for code that exists now: a rename must not silently drop it."""
        if not self.ban(owner, name, what):
            raise AttributeError(f"the guard in conftest.py refuses {what}, which no longer exists; update the guard")

    def table(self) -> None:
        """Every row of REFUSALS whose module and attribute are present."""
        for spec, paths in REFUSALS:
            for module in _modules(spec):
                for path in paths:
                    *chain, name = path.split(".")
                    owner = module
                    for part in chain:
                        owner = _probe(owner, part)
                    if owner is not _MISSING:
                        self.ban(owner, name, f"{module.__name__}.{path}")


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


# Targets that exist only after a port or an install. (module, attribute paths); a path may reach into a class
# ("Server.run"). A row is skipped when its module or attribute is absent. Add a row in the commit that adds the call.
REFUSALS: tuple[tuple[str, tuple[str, ...]], ...] = (
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
            "point_target",
            "point_region",
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
            "point_target",
            "point_region",
        ),
    ),
    # --- macOS frameworks called directly instead of through the adapter
    (
        "Quartz",
        (
            "CGEventPost",
            "CGEventPostToPid",
            "CGEventTapCreate",
            "CGWarpMouseCursorPosition",
            "CGDisplayMoveCursorToPoint",
            "CGWindowListCreateImage",
            "CGDisplayCreateImage",
            "CGDisplayCreateImageForRect",
            "CGRequestScreenCaptureAccess",
            "CGRequestPostEventAccess",
            "CGRequestListenEventAccess",
        ),
    ),
    (
        "ApplicationServices",
        (
            "AXUIElementPerformAction",
            "AXUIElementSetAttributeValue",
            "AXUIElementPostKeyboardEvent",
            "AXIsProcessTrustedWithOptions",
        ),
    ),
    ("AppKit", ("NSWorkspace", "NSRunningApplication", "NSPasteboard", "NSAppleScript")),
    ("Cocoa", ("NSWorkspace", "NSRunningApplication", "NSPasteboard", "NSAppleScript")),
    ("Foundation", ("NSAppleScript",)),
    # --- Windows APIs called directly (a stand-in module refuses everything already and is skipped)
    ("win32api", ("mouse_event", "keybd_event", "SetCursorPos", "ShellExecute")),
    ("win32gui", ("SetForegroundWindow", "ShowWindow", "BringWindowToTop", "PostMessage", "SendMessage")),
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
    # --- microphone and speaker
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
    ("glide:audio", ("AudioDevice.start", "AudioDevice.pause_input", "AudioDevice.resume_input")),
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
    ("glide:webhooks.worker", ("execute_native",)),
    # --- MCP: stdio spawns a server process or takes over this process's stdin and stdout, the rest open connections
    ("mcp", ("stdio_client",)),
    ("mcp.client.stdio", ("stdio_client",)),
    ("mcp.server.stdio", ("stdio_server",)),
    ("mcp.client.sse", ("sse_client",)),
    ("mcp.client.streamable_http", ("streamablehttp_client", "streamable_http_client")),
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
    (
        "glide:pet",
        ("launch_voice", "launch_text", "launch_point", "open_window", "show_point_marker", "dismiss_point_marker"),
    ),
    ("glide:pet_overlay", ("configure_overlay", "configure_panel")),
)

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
)


def _temp_roots() -> list[str]:
    return [os.path.realpath(tempfile.gettempdir())]


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

    def refuse(what: str):
        def call(*args, **kwargs):
            raise RuntimeError(f"a test reached the real machine through {what}; patch it in the test")

        return call

    guard = Guard(monkeypatch)

    for name in ("_post", "osascript", "screenshot", "open_path"):
        guard.strict(macos, name, f"macos.{name}")
    monkeypatch.setattr(macos, "mouse_location", lambda: (500.0, 500.0))
    if REAL_ACCESSIBILITY:
        for name in ("AXUIElementPerformAction", "AXUIElementSetAttributeValue"):
            guard.strict(macos.AS, name, f"ApplicationServices.{name}")
    # The Windows adapter: SendInput and the cursor carry all input; the rest launch, activate,
    # open, capture, or act on another app's element.
    for name in ("_send", "_move", "screenshot", "activate", "open_url", "open_path", "ax_press", "ax_focus", "ax_set_value"):
        guard.strict(windows, name, f"windows.{name}")
    monkeypatch.setattr(windows, "mouse_location", lambda: (500.0, 500.0))

    # The browser backend: no Chrome and no process of any kind, nothing over CDP, and no
    # connection except to a server on this machine that the test started itself.
    monkeypatch.setattr(subprocess, "Popen", refuse("subprocess.Popen"))
    for name in ("system", "posix_spawn", "posix_spawnp"):
        if hasattr(os, name):
            monkeypatch.setattr(os, name, refuse(f"os.{name}"))
    guard.strict(cdp, "find_chrome", "cdp.find_chrome")
    guard.strict(cdp, "_get_json", "the CDP HTTP endpoint")
    guard.strict(cdp.websocket, "create_connection", "a CDP websocket")
    for name in _PROCESS_CALLS:
        guard.ban(os, name, f"os.{name}")
    _only_own_process(monkeypatch, refuse)
    _loopback_only(monkeypatch, refuse)
    guard.table()
    _no_processes_from_multiprocessing(guard)


def apply_refusals(monkeypatch) -> Guard:
    """The table and process layers again, for a test that has just put a module in `sys.modules`."""
    guard = Guard(monkeypatch)
    guard.table()
    _no_processes_from_multiprocessing(guard)
    return guard


def _no_processes_from_multiprocessing(guard: Guard) -> None:
    process = _import_quiet("multiprocessing.process")
    if process is not None:
        guard.ban(process.BaseProcess, "start", "multiprocessing.Process.start")


def _only_own_process(monkeypatch, refuse) -> None:
    """A test may signal itself, and probe whether a pid exists (signal 0), and nothing else."""
    real_kill = os.kill

    def kill(pid, sig):
        if pid != os.getpid() and sig != 0:
            refuse(f"os.kill({pid!r}, {sig!r}) of another process")()
        return real_kill(pid, sig)

    monkeypatch.setattr(os, "kill", kill)


def _is_loopback(host: object) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(str(host).split("%")[0]).is_loopback
    except ValueError:
        return False


def _loopback_only(monkeypatch, refuse) -> None:
    """Sockets reach this machine's loopback address and nothing else, so a test can talk to a
    fake server it started (the `endpoint` fixture) but never to the network. A name other than
    localhost is not even looked up. Unix sockets are the same rule: only ones in the temp directory.
    Binding is held to the same line, so a test never exposes a listener to the network."""
    real_connect, real_connect_ex, real_getaddrinfo = socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo
    real_bind, real_sendto, real_sendmsg = socket.socket.bind, socket.socket.sendto, socket.socket.sendmsg

    def remote(sock: socket.socket, address: object) -> bool:
        if sock.family == getattr(socket, "AF_UNIX", None):
            return not local_socket_path(address)
        return sock.family in (socket.AF_INET, socket.AF_INET6) and not _is_loopback(address[0])

    def connect(sock, address):
        if remote(sock, address):
            refuse(f"a connection to {address!r}")()
        return real_connect(sock, address)

    def connect_ex(sock, address):
        if remote(sock, address):
            refuse(f"a connection to {address!r}")()
        return real_connect_ex(sock, address)

    def getaddrinfo(host, *args, **kwargs):
        if host is not None and not _is_loopback(host.decode() if isinstance(host, bytes) else host):
            refuse(f"a lookup of {host!r}")()
        return real_getaddrinfo(host, *args, **kwargs)

    def bind(sock, address):
        if sock.family == getattr(socket, "AF_UNIX", None):
            if address and not local_socket_path(address):
                refuse(f"binding the unix socket {address!r}")()
        elif sock.family in (socket.AF_INET, socket.AF_INET6) and not _is_loopback(address[0]):
            refuse(f"binding a listener to {address!r}")()
        return real_bind(sock, address)

    def sendto(sock, data, *rest):
        if sock.family in (socket.AF_INET, socket.AF_INET6) and rest and not _is_loopback(rest[-1][0]):
            refuse(f"a datagram to {rest[-1]!r}")()
        return real_sendto(sock, data, *rest)

    def sendmsg(sock, buffers, *rest, **kwargs):
        address = kwargs.get("address", rest[2] if len(rest) > 2 else None)
        if sock.family in (socket.AF_INET, socket.AF_INET6) and address and not _is_loopback(address[0]):
            refuse(f"a datagram to {address!r}")()
        return real_sendmsg(sock, buffers, *rest, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
    monkeypatch.setattr(socket.socket, "bind", bind)
    monkeypatch.setattr(socket.socket, "sendto", sendto)
    monkeypatch.setattr(socket.socket, "sendmsg", sendmsg)
    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    # The older resolver calls do not go through getaddrinfo. This machine's own name stays resolvable
    # (`socket.getfqdn()` asks for it, as does `email.utils.make_msgid`).
    own = {"", socket.gethostname()}

    def lookup(name: str, real):
        def resolve(host, *args, **kwargs):
            if host not in own and not _is_loopback(host):
                refuse(f"a lookup of {host!r}")()
            return real(host, *args, **kwargs)

        return resolve

    for name in ("gethostbyname", "gethostbyname_ex", "gethostbyaddr"):
        monkeypatch.setattr(socket, name, lookup(name, getattr(socket, name)))
    real_getnameinfo = socket.getnameinfo

    def getnameinfo(sockaddr, flags):
        if not _is_loopback(sockaddr[0]):
            refuse(f"a reverse lookup of {sockaddr[0]!r}")()
        return real_getnameinfo(sockaddr, flags)

    monkeypatch.setattr(socket, "getnameinfo", getnameinfo)
    _no_proxy(monkeypatch)


def _no_proxy(monkeypatch) -> None:
    """A proxy named in the environment is a loopback hop that carries the request anywhere, past the
    loopback rule above. No test goes through one: the settings are dropped and `urllib` is told there are none."""
    for name in ("http_proxy", "https_proxy", "all_proxy", "ftp_proxy"):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.upper(), raising=False)
    monkeypatch.setenv("NO_PROXY", "*")
    monkeypatch.setenv("no_proxy", "*")
    monkeypatch.setattr(urllib.request, "getproxies", dict)


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

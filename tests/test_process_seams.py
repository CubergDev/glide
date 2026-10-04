"""Seams: which code of `glide/` may start a process, that every input to the machine goes through `dispatch`, and where
text from the screen or a model may take an action.

Process starts. Every `.py` file under `glide/` is scanned. A file may start a process only when it is in `ALLOWED`
below, with the findings listed (a second `Popen` in an allowed file fails), a reason, and the owner who can remove it.
Only growth fails: the owner of an allowed file cannot edit this one, so when they remove a process start the suite stays
green and warns which entry to trim. `glide/computer/execution/spawn.py` is the seam the execution package uses; the other
entries are older code that predates it.

Input. Production code reaches the screen, keyboard, pointer or another app only through `control.dispatch`
(AGENTS.md: an attempted action is not a verified effect; `dispatch` is what marks a write as possibly in flight).
`INPUT` lists every `Desktop` call that does that. A static scan fails on any reference to one that is not the first
argument of `dispatch(...)`, unless it is in `INPUT_ALLOWED` with a reason; a runtime check runs the production
callers on a fake desktop that records whether `dispatch` was on the stack.

Destinations. The address `use_browser` opens and the fields it types into are checked in code (the last section).
"""

from __future__ import annotations

import ast
import sys
import warnings
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from execution_world import SCENARIOS, Computer, Jev, Reasoner, drive, response
from PIL import Image
from test_exec_backends_guard import process_starts

import glide
from glide.computer import actions, control, runner
from glide.computer.models import AxNode, Field, Item, Popup, Screen
from glide.computer.platform_adapter import Desktop, desktop
from glide.computer.writer import Answer

GLIDE = Path(glide.__file__).resolve().parent
ROOT = GLIDE.parent


def sources() -> dict[str, str]:
    return {p.relative_to(ROOT).as_posix(): p.read_text() for p in sorted(GLIDE.rglob("*.py"))}


# ------------------------------------------------------------------ process starts

SUBPROCESS_RUNNERS = {"run", "call", "check_call", "check_output", "getoutput", "getstatusoutput"}  # Popen is counted by name


def process_uses(source: str) -> Counter:
    """What `source` uses to start a process, with how many times: the names of `process_starts`, and `subprocess.run`
    and its relatives (a `subprocess` import alone does not say how many processes a file starts)."""
    found = Counter(process_starts(source))
    for node in ast.walk(ast.parse(source)):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "subprocess"
            and node.attr in SUBPROCESS_RUNNERS
        ):
            found[f"subprocess.{node.attr}"] += 1
    return found


@dataclass(frozen=True)
class Allowed:
    uses: Counter
    reason: str
    owner: str
    todo: str = ""  # a pending removal, with the file that carries it


ALLOWED: dict[str, Allowed] = {
    "glide/computer/execution/spawn.py": Allowed(
        Counter({"subprocess": 1, "Popen": 2}),
        "the seam: the one `Popen`, without a shell, that the execution backends call as `spawn.start`; tests refuse it",
        "execution",
    ),
    "glide/computer/macos.py": Allowed(
        Counter({"subprocess": 1, "Popen": 1, "subprocess.run": 3}),
        "the macOS adapter: osascript (a Popen the run control can terminate, with a deadline), `open` and `screencapture`; every call sits behind `osascript()` or a Desktop primitive",
        "platform",
    ),
    "glide/computer/windows.py": Allowed(
        Counter({"subprocess": 1, "Popen": 1, "webbrowser": 1, "startfile": 1}),
        "the Windows adapter: launching a browser with a URL, and opening a file for the user",
        "platform",
    ),
    "glide/computer/runner.py": Allowed(
        Counter({"subprocess": 1}),
        "names `subprocess.CalledProcessError` in one `except`, starts nothing. Fix: macos.osascript raises DesktopError, "
        "then runner.py need not import subprocess",
        "engine",
    ),
    "glide/providers/tts.py": Allowed(
        Counter({"subprocess": 1, "subprocess.run": 1}),
        "the macOS `say` voice: one synthesis command, never playback; `runner` is injectable, tests pass a fake",
        "voice",
    ),
    "glide/mcp/transport.py": Allowed(
        Counter({"stdio_client": 2}),
        "the MCP SDK starts the user's configured server; `open_stdio_client` returns before starting anything unless the "
        "caller's `approve` callback says yes to that exact command. Its docstring still says it is the only code that starts "
        "a process: stale, see the report",
        "mcp",
    ),
    "glide/computer/browser/cdp.py": Allowed(
        Counter({"subprocess": 1, "Popen": 2}),
        "TODO(docs/notes/cdp-launcher-removal.patch): `class Chrome` launches a Chrome of its own. Nothing in glide/ calls it "
        "(test_nothing_in_glide_reaches_the_chrome_launcher); only tests do. The patch deletes it and the guards on it",
        "platform (cdp.py)",
        todo="docs/notes/cdp-launcher-removal.patch",
    ),
}


def test_the_scan_counts_each_way_to_start_a_process():
    assert process_uses("import subprocess\nsubprocess.run(['x'])\nsubprocess.run(['y'])") == {
        "subprocess": 1,
        "subprocess.run": 2,
    }
    assert process_uses("import subprocess\nsubprocess.Popen(['x'])") == {"subprocess": 1, "Popen": 1}
    assert process_uses("from mcp.client.stdio import stdio_client\nstdio_client(p)") == {"stdio_client": 2}
    assert process_uses("import subprocess as sp") == {"subprocess": 1}
    assert not process_uses("def run(): pass\nx.run()\nrequest.call()")


def test_only_the_allowed_files_start_a_process_and_only_in_the_ways_listed():
    """Growth fails: a new file, or one more use in an allowed file. Shrinkage only warns, because the owner of that file
    cannot edit this one: when they remove a process start the suite must stay green, and the warning says which
    ALLOWED entry to trim."""
    grown, shrunk = {}, {}
    for path, source in sources().items():
        found = process_uses(source)
        entry = ALLOWED.get(path)
        expected = entry.uses if entry else Counter()
        if found - expected:
            grown[path] = {"found": dict(found), "allowed": dict(expected)}
        if expected - found:
            shrunk[path] = dict(expected - found)
    assert not grown, (
        "a file starts a process it is not allowed to. Go through glide.computer.execution.spawn, or add an ALLOWED "
        f"entry in this file with a reason and an owner: {grown}"
    )
    for path, gone in shrunk.items():
        warnings.warn(f"{path} no longer uses {gone}: trim its ALLOWED entry in tests/test_process_seams.py", stacklevel=1)


def test_every_allowed_entry_names_a_real_file_and_says_why():
    present = sources()
    for path, entry in ALLOWED.items():
        assert entry.reason and entry.owner, path
        if path not in present:
            warnings.warn(f"{path} is gone: delete its ALLOWED entry", stacklevel=1)
        elif entry.todo:
            assert (ROOT / entry.todo).is_file(), f"{path}: the TODO names {entry.todo}, which does not exist"


# ------------------------------------------------------------------ the Chrome launcher in cdp.py

LAUNCHER = {"CHROME_CANDIDATES", "DEFAULT_PROFILE", "find_chrome", "free_port", "Chrome"}


def launcher_references(source: str) -> list[str]:
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Name) and node.id in LAUNCHER:
            found.append(node.id)
        elif isinstance(node, ast.Attribute) and node.attr in LAUNCHER:
            found.append(node.attr)
        elif isinstance(node, ast.ImportFrom | ast.Import):
            found += [a.name for a in node.names if a.name.split(".")[-1] in LAUNCHER]
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in LAUNCHER:
            found.append(node.value)  # an `__all__` entry
    return found


def test_the_scan_sees_a_reference_to_the_launcher():
    for source in ("from .cdp import Chrome", "x = cdp.find_chrome()", "Chrome()", "__all__ = ['free_port']", "import a.Chrome"):
        assert launcher_references(source), source
    assert not launcher_references("x = 'Google Chrome'\nclass Session: pass\nchrome = 1")


def test_nothing_in_glide_reaches_the_chrome_launcher():
    """The one production `Popen` outside the seams is `cdp.Chrome.start`. No module of glide/, so no entry point in
    pyproject.toml, names Chrome, find_chrome, free_port or the constants beside them. Dynamic access cannot name them
    either: nothing in glide/ calls `getattr` on the cdp module or imports a module by a computed name from this package
    (platform_adapter picks macos or windows from two literals)."""
    users = {
        path: launcher_references(source)
        for path, source in sources().items()
        if path != "glide/computer/browser/cdp.py" and launcher_references(source)
    }
    assert not users, f"production code reaches the Chrome launcher in cdp.py: {users}"
    for path, source in sources().items():
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "getattr":
                target = node.args[0] if node.args else None
                assert not (isinstance(target, ast.Name) and target.id == "cdp"), f"{path}:{node.lineno} reads cdp by name"


def test_the_launcher_is_the_only_process_start_in_cdp_py():
    """So the patch, which deletes exactly `class Chrome` and what only it needs, leaves cdp.py with none."""
    tree = ast.parse(sources()["glide/computer/browser/cdp.py"])
    launcher = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Chrome"), None)
    if launcher is not None:  # None once the patch has landed
        tree.body.remove(launcher)
    rest = ast.unparse(tree)
    remaining = {k: v for k, v in process_uses(rest).items() if k not in {"subprocess"}}  # the import goes with the class
    assert remaining == {}, f"cdp.py starts a process outside class Chrome: {remaining}"


def test_the_removal_patch_deletes_the_launcher_and_the_guards_on_it():
    patch = (ROOT / "docs/notes/cdp-launcher-removal.patch").read_text()
    removed = {line[1:].strip() for line in patch.splitlines() if line.startswith("-") and not line.startswith("---")}
    for needed in (
        "class Chrome:",
        "def find_chrome() -> str:",
        "def free_port() -> int:",
        "CHROME_CANDIDATES = (",
        'DEFAULT_PROFILE = ""',
    ):
        assert needed in removed, needed
    assert 'guard.strict(cdp, "find_chrome", "cdp.find_chrome")' in removed  # tests/conftest.py
    assert "lambda tmp: cdp.find_chrome()," in removed  # tests/test_no_real_machine.py
    assert not [line for line in removed if "Popen" in line and "refuse" in line]  # the Popen refusal stays


# ------------------------------------------------------------------ input goes through dispatch

# Every `Desktop` call that moves the pointer, types, presses, activates or opens something, or acts on another app's
# element. A new Desktop method must be put in one of the two sets, or the first test below fails.
INPUT = {
    "click_at",
    "press",
    "type_text",
    "clear_field",
    "scroll",
    "activate",
    "open_url",
    "open_path",
    "ax_press",
    "ax_focus",
    "ax_set_value",
    "execution_tab",
    "execution_shortcut",
    "execution_scroll",
    "request_permissions",
}
READS = {
    "check_abort",
    "abort_hint",
    "sleep_watching",
    "accessibility_trusted",
    "screen_capture_trusted",
    "frontmost_app_and_pid",
    "browser_url",
    "frontmost_window_bounds",
    "screenshot",
    "display_scale",
    "recognize_text",
    "focused_field",
    "actionable_elements",
    "ax_value",
    "execution_tabs",
    "execution_scrolls",
    "execution_labels",
    "mouse_location",
    "point_target",
    "point_region",
}
RECEIVERS = {"desktop", "host", "adapter", "macos", "windows"}
ADAPTERS = {"glide/computer/macos.py", "glide/computer/windows.py", "glide/computer/platform_adapter.py"}  # they implement it


def input_uses(source: str) -> list[tuple[str, str]]:
    """(function, primitive) for each reference to an input primitive that is not the first argument of `dispatch(...)`.

    A reference is `desktop.click_at`, called or not (so `x = desktop.click_at` counts), on a receiver named in
    RECEIVERS, on `current()`, or on `<module>.desktop`; also `getattr(desktop, ...)`, and importing a primitive by name
    from an adapter module.
    """
    tree = ast.parse(source)
    dispatched = {
        id(call.args[0])
        for call in ast.walk(tree)
        if isinstance(call, ast.Call)
        and (getattr(call.func, "id", None) == "dispatch" or getattr(call.func, "attr", None) == "dispatch")
        and call.args
    }

    def on_desktop(value) -> bool:
        return (
            (isinstance(value, ast.Name) and value.id in RECEIVERS)
            or (isinstance(value, ast.Call) and getattr(value.func, "id", None) == "current")
            or (isinstance(value, ast.Attribute) and value.attr == "desktop")
        )

    found: list[tuple[str, str]] = []

    def visit(node, scope: tuple[str, ...]) -> None:
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            scope += (node.name,)
        where = ".".join(scope) or "<module>"
        if isinstance(node, ast.Attribute) and node.attr in INPUT and on_desktop(node.value) and id(node) not in dispatched:
            found.append((where, node.attr))
        elif (
            isinstance(node, ast.Call) and getattr(node.func, "id", None) == "getattr" and node.args and on_desktop(node.args[0])
        ):
            found.append((where, "getattr"))
        elif isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[-1] in {"macos", "windows", "platform_adapter"}:
            found.extend((where, a.name) for a in node.names if a.name in INPUT)
        for child in ast.iter_child_nodes(node):
            visit(child, scope)

    visit(tree, ())
    return found


def test_every_desktop_method_is_classified_as_input_or_a_read():
    methods = {n for n, v in vars(Desktop).items() if callable(v) and not n.startswith("_")}
    assert methods == INPUT | READS and not INPUT & READS, (
        f"classify these in INPUT or READS: {sorted(methods - INPUT - READS)}; gone from Desktop: {sorted((INPUT | READS) - methods)}"
    )


def test_the_input_scan_tells_dispatched_references_from_the_rest():
    assert input_uses("def f():\n    dispatch(desktop.click_at, (1, 2))\n    control.dispatch(desktop.press, 'a')") == []
    assert input_uses("def f():\n    desktop.click_at((1, 2))") == [("f", "click_at")]
    assert input_uses("class C:\n    def m(self):\n        run = desktop.type_text") == [("C.m", "type_text")]
    assert input_uses("def f():\n    dispatch(g, desktop.press)") == [("f", "press")]  # only the first argument is gated
    assert input_uses("def f():\n    getattr(desktop, name)(1)") == [("f", "getattr")]
    assert input_uses("def f():\n    platform_adapter.desktop.activate('x')") == [("f", "activate")]
    assert input_uses("def f():\n    current().open_url('b', 'u')") == [("f", "open_url")]
    assert input_uses("from .macos import click_at, screenshot") == [("<module>", "click_at")]
    assert input_uses("def f():\n    desktop.screenshot()\n    recorder.activate()\n    scroller.scroll(3)") == []


@dataclass(frozen=True)
class Outside:
    uses: frozenset[str]
    reason: str
    owner: str
    gap: bool = False  # a real hole, listed so it cannot grow; the entry may go once it is fixed


INPUT_ALLOWED: dict[tuple[str, str], Outside] = {
    ("glide/computer/execution/native.py", "NativeBackend.execute"): Outside(
        frozenset(
            {"open_url", "execution_tab", "execution_shortcut", "execution_scroll", "ax_press", "click_at", "ax_focus"}
            | {"ax_set_value", "clear_field", "type_text"}
        ),
        "the native backend's one write path; the engine calls it only as `dispatch(backend.execute, ...)` "
        "(execution/engine.py, `_dispatch`; test_the_engine_dispatches_every_write below)",
        "engine",
    ),
    ("glide/computer/runner.py", "hand_off"): Outside(
        frozenset({"activate"}),
        "GAP: after the user answers a question the terminal is in front, and `desktop.activate(state.view[0].app)` puts the "
        "work back with no checkpoint and no in_flight mark. Fix: `dispatch(desktop.activate, state.view[0].app)`",
        "engine (runner.py)",
        gap=True,
    ),
    ("glide/computer/cli.py", "inspect"): Outside(
        frozenset({"open_path"}),
        "`glide-inspect` shows its own output files to the user unless `--no-open`; it runs no task and has no RunControl",
        "cli",
    ),
    ("glide/computer/desktop_access.py", "prepare_desktop"): Outside(
        frozenset({"request_permissions"}),
        "asks macOS for Screen Recording and Accessibility before a voice session starts; no task, no input to another app",
        "voice",
    ),
}


def test_no_production_module_sends_input_outside_dispatch():
    found = {}
    for path, source in sources().items():
        if path in ADAPTERS:
            continue
        for where, name in input_uses(source):
            found.setdefault((path, where), set()).add(name)
    unexpected = {}
    for key, names in found.items():
        entry = INPUT_ALLOWED.get(key)
        extra = names - (entry.uses if entry else frozenset())
        if extra:
            unexpected[f"{key[0]}:{key[1]}"] = sorted(extra)
    assert not unexpected, (
        "input outside control.dispatch. Wrap it as dispatch(desktop.<call>, ...) or, when it cannot be, add an "
        f"INPUT_ALLOWED entry with the reason: {unexpected}"
    )
    for key, entry in INPUT_ALLOWED.items():
        if not entry.reason or not entry.owner:
            pytest.fail(f"{key} needs a reason and an owner")
        if key not in found or entry.uses - found[key]:
            gone = sorted(entry.uses - found.get(key, set()))
            warnings.warn(f"{key[0]}:{key[1]} no longer uses {gone}: trim its INPUT_ALLOWED entry", stacklevel=1)


CDP_WRITES = ("Input.", "Page.navigate", "Target.createTarget", "Target.activateTarget", "Target.closeTarget")
CDP_WRITERS = {
    "glide/computer/execution/dom.py": "DomBackend.execute and what it calls; the engine dispatches it like the native one",
    "glide/computer/execution/playwright_cli.py": "names the methods the bridge may relay (`WRITES`), and relays them for DomBackend",
    "glide/computer/browser/act.py": "GAP, legacy: the old DOM loop sends input with no dispatch at all. Unreachable from production "
    "(docs/notes/legacy-browser-loop.md); it goes with that loop",
}


def test_browser_writes_over_cdp_come_only_from_the_dispatched_backends():
    found = set()
    for path, source in sources().items():
        for node in ast.walk(ast.parse(source)):
            is_write = isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith(CDP_WRITES)
            if is_write and path != "glide/computer/browser/cdp.py":  # which only compares a method name to time its deadline
                found.add(path)
    assert found <= set(CDP_WRITERS), (
        f"new module sending browser input or navigation over CDP: {sorted(found - set(CDP_WRITERS))}"
    )


# -- at run time: a fake desktop that sees whether `dispatch` is on the stack


def dispatch_active() -> bool:
    frame = sys._getframe(1)
    while frame is not None:
        if frame.f_code is control.dispatch.__code__:
            return True
        frame = frame.f_back
    return False


@pytest.fixture
def machine(monkeypatch):
    """Every input primitive recorded with whether `dispatch` was running when it was called. Reads stay quiet."""
    log: list[tuple[str, bool]] = []

    def recorder(name):
        def record(*args, **kwargs):
            log.append((name, dispatch_active()))
            return name in {"ax_press", "ax_focus", "ax_set_value", "activate", "open_url"}

        return record

    for name in INPUT:
        monkeypatch.setattr(desktop, name, recorder(name))
    monkeypatch.setattr(desktop, "sleep_watching", lambda seconds: None)
    monkeypatch.setattr(desktop, "ax_value", lambda ref: "hello")
    monkeypatch.setattr(desktop, "focused_field", lambda: None)
    return log


def blank_screen(field: Field | None = None, **changes) -> Screen:
    return Screen(image=Image.new("RGB", (2000, 1200)), scale=2.0, app="Google Chrome", field=field, url=None, **changes)


def test_every_action_the_legacy_loop_performs_runs_under_dispatch(machine, monkeypatch):
    ref = object()
    text_field = Field("AXTextField", "Search", "", "", 10, 20, 200, 30, ref=ref)
    plain_field = replace(text_field, ref=None)
    link = Item(3, "Register", 1.0, 100, 100, 300, 140, role="link", source="ax")
    node = AxNode(role="AXLink", label="Hidden", x=0.0, y=-4200.0, w=120.0, h=32.0, pressable=True, ref=ref)
    close = AxNode(role="AXButton", label="Close", x=1800.0, y=100.0, w=20.0, h=20.0, pressable=True, ref=None)
    bubble = Popup("Restore pages?", 1700.0, 90.0, 300.0, 100.0, close=close)
    monkeypatch.setattr(actions, "compose_text", lambda *a: SimpleNamespace(text="hello", submit=True))
    monkeypatch.setattr(actions, "compose_url", lambda *a: "https://example.com/")
    ctx = actions.Context("goal", "Google Chrome", "me@example.org", None, object(), [])
    cases = [
        ("click_item pressed", lambda: actions.click_item(link, blank_screen(ax_refs={3: ref}))),
        ("click_item clicked", lambda: actions.click_item(link, blank_screen())),
        ("click_item under a popup", lambda: actions.click_item(link, blank_screen(covered={3: bubble}))),
        ("close_popup with Escape", lambda: actions.close_popup(replace(bubble, close=None))),
        ("press_offscreen", lambda: actions.press_offscreen("0", blank_screen(offscreen=[node]))),
        ("fill_field by value", lambda: actions.fill_field(text_field, "x")),
        ("fill_field by keystrokes", lambda: actions.fill_field(plain_field, "x")),
        ("restore_field", lambda: actions.restore_field(text_field, "hello")),
    ]
    for chosen, extra in [
        ("use_browser", {"site": SimpleNamespace(choice="none")}),
        ("use_browser", {"site": SimpleNamespace(choice="github")}),
        ("use_browser", {"site": SimpleNamespace(choice="other")}),
        ("type_email", {}),
        ("type_text", {}),
        ("press_enter", {}),
        ("press_escape", {}),
        ("go_back", {}),
        ("scroll_down", {}),
        ("scroll_up", {}),
        ("wait", {}),
    ]:
        decision = SimpleNamespace(chosen=chosen, **extra)
        cases.append((chosen, lambda decision=decision: actions.perform(decision, blank_screen(field=text_field), [], ctx)))
    for label, run in cases:
        machine.clear()
        run()
        assert all(on for _, on in machine), f"{label}: input outside dispatch: {machine}"
    # The check can fail: a call made straight on the desktop is seen.
    machine.clear()
    desktop.click_at((1, 2))
    assert machine == [("click_at", False)]


def test_the_engine_dispatches_every_write(monkeypatch, tmp_path):
    class Watched(Computer):
        def __init__(self):
            super().__init__()
            self.calls = []

        def execute(self, action, observed):
            self.calls.append((action.kind, dispatch_active()))
            return super().execute(action, observed)

    computer = Watched()
    drive(monkeypatch, tmp_path, computer, Reasoner([response(*SCENARIOS["edit and save a desktop form"])]), Jev("plan"))
    writes = [(kind, on) for kind, on in computer.calls if kind != "inspect"]
    assert writes, "the scenario made no write"
    assert all(on for _, on in writes), f"a write reached the backend outside dispatch: {computer.calls}"
    assert not any(on for kind, on in computer.calls if kind == "inspect"), "reads are not writes"


@pytest.mark.xfail(
    strict=False,
    reason="runner.py hand_off calls desktop.activate directly after the user answers a question; "
    "fix: dispatch(desktop.activate, ...) (see INPUT_ALLOWED)",
)
def test_putting_the_work_back_in_front_after_a_question_goes_through_dispatch(machine, monkeypatch, tmp_path):
    answers = iter([Answer("which one?", False, question="which one?"), Answer("done", True)])
    monkeypatch.setattr(runner, "review", lambda *a: next(answers))
    ctx = actions.Context("goal", "Google Chrome", None, None, object(), [], ask=lambda question: "the second")
    state = runner.RunState(outcome="done")
    state.view = (blank_screen(), [])
    cfg = runner.RunConfig(goal="goal", out=tmp_path, act=True)
    runner.hand_off(cfg, ctx, state, 1, lambda *a, **k: None)
    assert machine == [("activate", True)]


# ------------------------------------------------------------------ the legacy DOM loop is not production

LEGACY_LOOP = {f"glide.computer.browser.{name}" for name in ("runner", "decide", "perceive", "report", "act")}


def imported_modules(source: str, package: str) -> set[str]:
    """Every glide module `source` imports, at the top or inside a function, and each package above it (importing
    a.b.c runs a/__init__ and a/b/__init__). `package` is the package the file sits in, for relative imports."""
    found = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = package.split(".")[: len(package.split(".")) - (node.level - 1)] if node.level else []
            full = ".".join([*base, *([node.module] if node.module else [])])
            names = [full, *(f"{full}.{a.name}" for a in node.names)]
        else:
            continue
        for name in names:
            parts = name.split(".")
            found.update(".".join(parts[:i]) for i in range(1, len(parts) + 1))
    return {m for m in found if m.startswith("glide")}


def module_name(path: str) -> tuple[str, str]:
    parts = path.removesuffix(".py").split("/")
    if parts[-1] == "__init__":
        return ".".join(parts[:-1]), ".".join(parts[:-1])
    return ".".join(parts), ".".join(parts[:-1])


def test_the_legacy_browser_loop_is_imported_by_nothing_but_itself():
    """browser/{runner,decide,perceive,report,act}.py are the old DOM loop. No other module of glide/ imports them (the
    package `__init__` used to, so `from ..browser.cdp import ...` loaded all of it), and no entry point reaches them."""
    users = {}
    for path, source in sources().items():
        name, package = module_name(path)
        hit = (imported_modules(source, package) & LEGACY_LOOP) - {name}
        if hit and name not in LEGACY_LOOP:
            users[name] = sorted(hit)
    assert not users, f"production code imports the legacy browser loop: {users}"


def test_nothing_imports_by_a_computed_name_so_the_scan_above_sees_every_import():
    dynamic = {
        path
        for path, source in sources().items()
        if any(
            (isinstance(n, ast.Name) and n.id in {"__import__", "import_module"})
            or (isinstance(n, ast.Attribute) and n.attr in {"import_module", "__import__", "spec_from_file_location"})
            for n in ast.walk(ast.parse(source))
        )
    }
    assert dynamic == {"glide/computer/platform_adapter.py"}, dynamic
    adapter = sources()["glide/computer/platform_adapter.py"]
    assert adapter.count("import_module(") == 2 and 'f"{__package__}.windows"' in adapter and 'f"{__package__}.macos"' in adapter


def test_no_entry_point_group_loads_a_module_by_name():
    import tomllib

    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert not {"gui-scripts", "entry-points"} & set(project)
    targets = {target.split(":")[0] for target in project["scripts"].values()}
    assert targets and all(name.startswith("glide.") for name in targets), targets
    assert not targets & LEGACY_LOOP


# ------------------------------------------------------------------ destinations and credential fields
#
# The address use_browser opens, and the fields it types into. A destination comes from the catalog (configuration), from an
# address in the user's own goal, or from the writer's proposal. A proposal is model output, built from the goal and from a
# history that quotes screen text, so it is checked in code before the browser is told to open it (AGENTS.md: page, task and
# provider text is untrusted data). The macOS adapter puts the address inside an AppleScript string literal, so a quote or
# backslash in it would end that literal.

BLANK = Screen(image=Image.new("RGB", (2000, 1200)), scale=2.0, app="Google Chrome", field=None, url=None)


def goal_context(goal="find the next upcoming bruno mars concert") -> actions.Context:
    return actions.Context(goal=goal, browser="Google Chrome", email="me@example.org", typesafe=None, writer=object(), history=[])


@pytest.fixture
def opened(monkeypatch):
    log = []
    monkeypatch.setattr(desktop, "activate", lambda app: log.append(("activate", app)) or True)
    monkeypatch.setattr(desktop, "open_url", lambda app, url: log.append(("open", url)) or True)
    return log


def use_browser(opened, proposal, goal="find the next upcoming bruno mars concert"):
    """What use_browser does when the classifier says `other` and the writer proposes `proposal`."""
    decision = SimpleNamespace(chosen="use_browser", site=SimpleNamespace(choice="other"))
    return actions.perform(decision, BLANK, [], goal_context(goal))


@pytest.mark.parametrize(
    "proposal",
    [
        'https://a.example/"&(do shell script "id")&"',  # ends the AppleScript string literal
        "https://a.example/\\x",
        "https://a.example/\x07",
        "https://user:secret@a.example/",
        "https://a.example:8443/",
        "https://a.example/?q=what+the+screen+said",  # a query carries data out
        "https://a.example/#what-the-screen-said",
        "https://192.168.1.1/admin",
        "https://printer.local/",
        "https://localhost.localdomain/",
        "https://ex\u0430mple.com/",  # with a Cyrillic a
    ],
)
def test_a_proposed_address_that_could_carry_data_or_break_out_is_refused(opened, monkeypatch, proposal):
    monkeypatch.setattr(actions, "compose_url", lambda *a: proposal)
    refusal = use_browser(opened, proposal)
    assert refusal.startswith("use_browser refused: ") and "address" in refusal
    assert opened == []  # nothing was opened, and the browser was not even brought forward


@pytest.mark.parametrize(
    "proposal",
    ["https://www.songkick.com/", "https://en.wikipedia.org/wiki/Bruno_Mars", "https://docs.python.org/3/library/"],
)
def test_a_plain_public_page_the_writer_proposes_is_opened(opened, monkeypatch, proposal):
    monkeypatch.setattr(actions, "compose_url", lambda *a: proposal)
    assert use_browser(opened, proposal) == f"opened {proposal}"
    assert opened == [("open", proposal)]


def test_an_address_the_user_typed_into_the_goal_may_carry_a_query_and_a_port(opened):
    goal = "open example.com:8443/search?q=cats"  # the user's own words, which compose_url returns without the writer
    assert use_browser(opened, "", goal) == "opened https://example.com:8443/search?q=cats"


def test_an_address_in_the_goal_that_would_end_an_applescript_string_is_still_refused(opened):
    goal = 'open example.com/"x'
    assert use_browser(opened, "", goal).startswith("use_browser refused: ")
    assert opened == []


def test_a_catalog_site_needs_no_check(opened):
    decision = SimpleNamespace(chosen="use_browser", site=SimpleNamespace(choice="github"))
    assert actions.perform(decision, BLANK, [], goal_context()) == "opened https://github.com/"


def labelled(label="Search", placeholder="", role="AXTextField") -> Field:
    return Field(role=role, label=label, placeholder=placeholder, value="", x=10, y=20, w=200, h=30, ref=None)


@pytest.fixture
def typed(monkeypatch):
    log = []
    monkeypatch.setattr(desktop, "clear_field", lambda: log.append("clear"))
    monkeypatch.setattr(desktop, "type_text", lambda text: log.append(text))
    monkeypatch.setattr(desktop, "press", lambda key, command=False: log.append(f"press {key}"))
    monkeypatch.setattr(actions, "compose_text", lambda *a: SimpleNamespace(text="quarterly report", submit=True))
    return log


@pytest.mark.parametrize("chosen", ["type_text", "type_email"])
@pytest.mark.parametrize(
    "label, placeholder",
    [
        ("Password", ""),
        ("", "Enter your PIN"),
        ("Card number", ""),
        ("One-time code", ""),
        ("API key", ""),
        ("Recovery codes", ""),
    ],
)
def test_nothing_is_typed_into_a_field_that_asks_for_a_credential(typed, chosen, label, placeholder):
    screen = Screen(image=BLANK.image, scale=2.0, app="Google Chrome", field=labelled(label, placeholder), url=None)
    refusal = actions.perform(SimpleNamespace(chosen=chosen), screen, [], goal_context())
    assert refusal == f"{chosen} refused: the focused field asks for a credential"
    assert typed == []


@pytest.mark.parametrize("label", ["Search", "Shipping address", "Mapping notes", "Spinner count", "Title"])
def test_an_ordinary_field_whose_name_merely_contains_pin_or_key_is_typed_into(typed, label):
    screen = Screen(image=BLANK.image, scale=2.0, app="Google Chrome", field=labelled(label), url=None)
    result = actions.perform(SimpleNamespace(chosen="type_text"), screen, [], goal_context())
    assert result.startswith("typed 'quarterly report'") and "quarterly report" in typed

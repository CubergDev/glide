"""Seams: which code of `glide/` may start a process, and that every input to the machine goes through `dispatch`.

Process starts. Every `.py` file under `glide/` is scanned. A file may start a process only when it is in `ALLOWED`
below, with exactly the findings listed (a second `Popen` in an allowed file fails, and so does an entry that no longer
matches), a reason, and the owner who can remove it. `glide/computer/execution/spawn.py` is the seam the execution
package uses; the other entries are older code that predates it.

Input. Production code reaches the screen, keyboard, pointer or another app only through `control.dispatch`
(AGENTS.md: an attempted action is not a verified effect; `dispatch` is what marks a write as possibly in flight).
`INPUT` lists every `Desktop` call that does that. A static scan fails on any reference to one that is not the first
argument of `dispatch(...)`, unless it is in `INPUT_ALLOWED` with a reason; a runtime check runs the production
callers on a fake desktop that records whether `dispatch` was on the stack.
"""

from __future__ import annotations

import ast
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pytest
from test_exec_backends_guard import process_starts

import glide

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
    todo: str = ""  # a pending removal: the entry may be gone (the removal landed), but never grow


ALLOWED: dict[str, Allowed] = {
    "glide/computer/execution/spawn.py": Allowed(
        Counter({"subprocess": 1, "Popen": 2}),
        "the seam: the one `Popen`, without a shell, that the execution backends call as `spawn.start`; tests refuse it",
        "execution",
    ),
    "glide/computer/macos.py": Allowed(
        Counter({"subprocess": 1, "subprocess.run": 4}),
        "the macOS adapter: osascript, `open` and `screencapture`; every call sits behind `osascript()` or a Desktop primitive",
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
    wrong = {}
    for path, source in sources().items():
        found = process_uses(source)
        entry = ALLOWED.get(path)
        expected = entry.uses if entry else Counter()
        if found != expected and not (entry and entry.todo and not found):
            wrong[path] = {"found": dict(found), "allowed": dict(expected)}
    assert not wrong, (
        "a file starts a process it is not allowed to. Go through glide.computer.execution.spawn, or add an ALLOWED "
        f"entry in this file with a reason and an owner: {wrong}"
    )


def test_every_allowed_entry_names_a_real_file_and_says_why():
    present = sources()
    for path, entry in ALLOWED.items():
        assert path in present, f"{path} is gone: delete its ALLOWED entry"
        assert entry.reason and entry.owner
        if entry.todo:
            assert (ROOT / entry.todo).is_file(), f"{path}: the TODO names {entry.todo}, which does not exist"
        elif not process_uses(present[path]):
            pytest.fail(f"{path} no longer starts a process: delete its ALLOWED entry")
    for path, entry in ALLOWED.items():
        if entry.todo and not process_uses(present[path]):
            warnings.warn(f"{path}: the pending removal landed; delete its ALLOWED entry", stacklevel=1)


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

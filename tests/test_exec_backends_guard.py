"""The process seam is the only way the execution package starts anything, and a test cannot reach it by accident.

The scan itself (`process_starts`) lives here. Which file of `glide/` may start a process, and why, is
tests/test_process_seams.py.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest

from glide.computer.execution import spawn

REAL_START = spawn.start  # taken at import, before the autouse guard replaces it

BANNED_MODULES = {"subprocess", "multiprocessing", "pty", "pexpect", "webbrowser"}
# No other meaning. `stdio_client` and `open_process` are how the MCP SDK and anyio start a child process.
ANYWHERE = {"Popen", "create_subprocess_exec", "create_subprocess_shell", "stdio_client", "open_process"}
ON_OS = {"system", "popen", "fork", "forkpty", "posix_spawn", "posix_spawnp", "startfile"}  # only as os.<name>


def process_starts(source: str) -> list[str]:
    """What in `source` could start a process: banned imports, Popen, and os.<name> calls that run or replace a process."""

    def on_os(name):
        return name in ON_OS or name.startswith(("exec", "spawn"))

    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found += [a.name for a in node.names if a.name.split(".")[0] in BANNED_MODULES]
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] in BANNED_MODULES:
                found.append(node.module)
            found += [a.name for a in node.names if a.name in ANYWHERE or (node.module == "os" and on_os(a.name))]
        elif isinstance(node, ast.Attribute | ast.Name):
            name = node.attr if isinstance(node, ast.Attribute) else node.id
            via_os = isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "os"
            if name in ANYWHERE or (via_os and on_os(name)):
                found.append(name)
    return found


def test_the_scan_sees_each_way_of_starting_a_process():
    for source in (
        "import subprocess",
        "from subprocess import run",
        "import os\nos.system('x')",
        "import os\nos.execv('x', [])",
        "import os\nos.spawnl(0, 'x')",
        "from os import fork",
        "import asyncio\nasyncio.create_subprocess_exec('x')",
        "import webbrowser",
        "from mcp.client.stdio import stdio_client",
        "import anyio\nawait anyio.open_process(['x'])",
    ):
        assert process_starts(source), source
    assert not process_starts("import os\nos.path.join('a', 'b')\nx = {'run': 1}")
    # Ordinary names that happen to match are not process starts.
    assert not process_starts("system = 'prompt'\nrequest.system\nrequest.fork()\nself.spawn_count = 1\ndef execute(): pass")


def test_the_seam_itself_starts_the_process_through_popen_without_a_shell(monkeypatch):
    assert "Popen" in process_starts(Path(spawn.__file__).read_text())  # the seam is the one place that may
    seen = []

    class Process:
        def __init__(self, args, **options):
            seen.append((args, options))

    monkeypatch.setattr(subprocess, "Popen", Process)
    REAL_START(["tool", "--flag"], stdin=subprocess.DEVNULL)
    assert seen == [(["tool", "--flag"], {"shell": False, "stdin": subprocess.DEVNULL})]
    with pytest.raises(TypeError):
        REAL_START(["tool"], shell=True)  # a shell string is never an option


def test_the_seam_is_refused_in_every_test_until_a_test_replaces_it():
    with pytest.raises(RuntimeError, match="real machine"):
        spawn.start(["true"])


def test_popen_stays_refused_underneath_the_seam():
    with pytest.raises(RuntimeError, match="real machine"):
        REAL_START(["true"])

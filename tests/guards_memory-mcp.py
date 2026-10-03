"""Guard for glide.memory and glide.mcp tests: nothing reaches the user's real data or starts a real MCP server.

Registered by tests/conftest.py (by path: the hyphen in this file's name makes it unimportable as a module) and
imported by the test modules of tests/memory and tests/mcp through their loaders. It confines only the tests under
those two directories. It adds to tests/conftest.py's `no_real_machine`; it does not replace or weaken it.

What it refuses, so that a test which forgot to patch fails instead of acting on the real machine:
- the user's real home, data directory and glide.toml: HOME, GLIDE_DATA_DIR and the GLIDE_* switches are replaced
  by private temporary values, and a database or directory outside the system temp area is refused;
- the MCP SDK's real transports (stdio, SSE, streamable HTTP, websocket) and asyncio's subprocess helpers.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import tempfile
from pathlib import Path

import pytest

_SDK_TRANSPORTS = {
    "mcp.client.stdio": ("stdio_client",),
    "mcp.client.sse": ("sse_client",),
    "mcp.client.streamable_http": ("streamablehttp_client", "streamable_http_client"),
    "mcp.client.websocket": ("websocket_client",),
}


def _sdk_installed() -> bool:
    # tests/mcp is importable as a namespace package named "mcp" when the SDK is absent, so ask for a submodule
    try:
        return importlib.util.find_spec("mcp.client.stdio") is not None
    except ImportError:
        return False


def _inside_temp(path: object) -> bool:
    if str(path) == ":memory:":
        return True
    try:
        resolved = Path(os.fspath(path)).expanduser().resolve()
        return resolved.is_relative_to(Path(tempfile.gettempdir()).resolve())
    except (OSError, TypeError, ValueError):
        return False


@pytest.fixture(autouse=True)
def no_real_memory_or_mcp(request, monkeypatch, tmp_path):
    # conftest registers every tests/guards_*.py suite-wide, so this autouse fixture runs for every test: it
    # confines only the tests of the packages it guards and leaves the temp directory of all others untouched.
    if not {"memory", "mcp"} & set(request.node.path.relative_to(Path(__file__).parent).parts):
        return

    def refuse(what: str):
        def call(*args, **kwargs):
            raise RuntimeError(f"a test reached the real machine through {what}; patch it in the test")

        return call

    home = tmp_path / "guard-home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("GLIDE_DATA_DIR", str(tmp_path / "guard-data"))
    for name in ("GLIDE_MEMORY", "GLIDE_MEMORY_AUTO", "GLIDE_CONFIG", "XDG_DATA_HOME", "LOCALAPPDATA"):
        monkeypatch.delenv(name, raising=False)

    from glide.memory import service, store

    real_prepare = store._prepare_path
    real_private = service._private_dir

    def prepare(path):
        if not str(path).startswith("file:") and not _inside_temp(path):  # a file: URI is refused by the store itself
            refuse(f"a database at {path!s}")()
        return real_prepare(path)

    def private(path, **kwargs):
        if not _inside_temp(path):
            refuse(f"a directory at {path!s}")()
        return real_private(path, **kwargs)

    monkeypatch.setattr(store, "_prepare_path", prepare)
    monkeypatch.setattr(service, "_private_dir", private)

    if _sdk_installed():
        for module_name, names in _SDK_TRANSPORTS.items():
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                continue
            for name in names:
                if hasattr(module, name):
                    monkeypatch.setattr(module, name, refuse(f"{module_name}.{name}"))
    import asyncio

    for name in ("create_subprocess_exec", "create_subprocess_shell"):
        monkeypatch.setattr(asyncio, name, refuse(f"asyncio.{name}"))

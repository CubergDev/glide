"""Guard for glide.memory and glide.mcp tests: nothing reaches the user's real data or starts a real MCP server.

Loaded by path from tests/memory/conftest.py and tests/mcp/conftest.py (the hyphen in this file's name makes it
unimportable as a module). tests/conftest.py should load it the same way so the guard also covers any other test
that touches these packages. It adds to tests/conftest.py's `no_real_machine`; it does not replace or weaken it.

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


def _inside_temp(path: object) -> bool:
    if str(path) == ":memory:":
        return True
    try:
        resolved = Path(os.fspath(path)).expanduser().resolve()
        return resolved.is_relative_to(Path(tempfile.gettempdir()).resolve())
    except (OSError, TypeError, ValueError):
        return False


@pytest.fixture(autouse=True)
def no_real_memory_or_mcp(monkeypatch, tmp_path):
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

    if importlib.util.find_spec("mcp") is not None:
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

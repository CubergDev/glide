"""Refusals for the app server's calls that reach the machine.

An autouse fixture, registered by `tests/conftest.py` (`pytest_configure` loads every `tests/guards_*.py` by path).

What it refuses, wherever the package holds a reference to it:

- `glide.app_server.cli.serve`: binds the person's real socket path and builds the real assistant and, through it, every provider.
- `glide.app_server.runtime.real_voice`: opens the microphone and the speaker. Nothing in the server names it except `cli.serve`;
  a `Runtime` or `AppBridge` made without a `voice_factory` has no voice at all.

The socket itself needs no row here: `tests/conftest.py` already refuses a Unix socket anywhere but inside the temp directory, and
a `socketpair` (the tests' in-process connection) names nothing on the machine.
"""

from __future__ import annotations

import sys

import pytest


def _refuse(what: str):
    def call(*args, **kwargs):
        raise RuntimeError(f"a test reached the real machine through {what}; use a fake")

    return call


@pytest.fixture(autouse=True)
def no_app_server_machine_reach(monkeypatch):
    import importlib

    for name in ("glide.app_server.cli", "glide.app_server.runtime"):  # imported now, so their names exist to be replaced
        importlib.import_module(name)

    for name, module in list(sys.modules.items()):
        if module is None or not name.startswith("glide.app_server"):
            continue
        if hasattr(module, "real_voice"):
            monkeypatch.setattr(module, "real_voice", _refuse("glide.app_server.runtime.real_voice (the microphone and speaker)"))
        if name == "glide.app_server.cli":
            monkeypatch.setattr(module, "serve", _refuse("glide.app_server.cli.serve (the real socket and the real assistant)"))

"""Guards for the voice stack: no test opens a sound device or a websocket.

`no_real_audio` is an autouse fixture. Each module under `tests/speech/` imports it (a second
`conftest.py` there would shadow the root one: tests do `from conftest import busy_page`), and the root
`tests/conftest.py` must import it too (`from guards_voice import no_real_audio  # noqa: F401`) so
that every test in the suite is covered, not only the ones under `tests/speech/`.

What it refuses, when the module is installed:

- `sounddevice`: every way of opening a stream or recording or playing a buffer. Opening a stream is
  what asks macOS for the microphone and starts the speaker, so it is the call that must never happen.
- `websockets`: every `connect` entry point, in the sync, asyncio and top-level APIs. (`websocket-client`
  is already refused by tests/conftest.py, which patches `websocket.create_connection`.)

A module that is not installed gets nothing: no stub is put in `sys.modules`, because tests of the
audio modules insert their own fake `sounddevice` and the "not installed" path expects an ImportError.
A test that needs a real call to get through patches it back itself, after this fixture.
"""

from __future__ import annotations

import importlib
import importlib.util

import pytest

SOUNDDEVICE_CALLS = (
    "RawInputStream",
    "RawOutputStream",
    "RawStream",
    "InputStream",
    "OutputStream",
    "Stream",
    "rec",
    "play",
    "playrec",
)
WEBSOCKETS_CALLS = (
    ("websockets", ("connect", "unix_connect")),
    ("websockets.sync.client", ("connect", "unix_connect")),
    ("websockets.asyncio.client", ("connect", "unix_connect")),
)


def _refusal(what: str):
    def refuse(*args, **kwargs):
        raise RuntimeError(f"a test reached the real machine through {what}; give the code a fake instead")

    return refuse


def installed(name: str) -> bool:
    """Whether `name` can be imported, without importing it (and so without side effects)."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def refuse_audio(monkeypatch, module, names) -> list[str]:
    """Replace each of `names` that `module` has with a function that raises. Returns the names replaced."""
    replaced = []
    for name in names:
        if hasattr(module, name):
            monkeypatch.setattr(module, name, _refusal(f"{module.__name__}.{name}"))
            replaced.append(name)
    return replaced


def refuse_all_installed(monkeypatch) -> list[str]:
    """Apply every refusal whose module is installed. Returns "module.name" for each, for the tests of the guard."""
    done: list[str] = []
    if installed("sounddevice"):
        module = importlib.import_module("sounddevice")
        done += [f"sounddevice.{name}" for name in refuse_audio(monkeypatch, module, SOUNDDEVICE_CALLS)]
    for module_name, names in WEBSOCKETS_CALLS:
        if installed(module_name.split(".")[0]) and installed_submodule(module_name):
            module = importlib.import_module(module_name)
            done += [f"{module_name}.{name}" for name in refuse_audio(monkeypatch, module, names)]
    return done


def installed_submodule(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError, AttributeError):
        return False


@pytest.fixture(autouse=True)
def no_real_audio(monkeypatch):
    refuse_all_installed(monkeypatch)

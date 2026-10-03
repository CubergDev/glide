"""The voice guard refuses what it says it refuses, tested on fake modules so it needs no audio package."""

from __future__ import annotations

import sys
import types

import pytest
from guards_voice import SOUNDDEVICE_CALLS, installed, no_real_audio, refuse_all_installed, refuse_audio  # noqa: F401


def fake_module(name: str, *names: str) -> types.ModuleType:
    module = types.ModuleType(name)
    for attr in names:
        setattr(module, attr, lambda *a, **k: "opened")
    return module


def test_every_way_of_opening_a_stream_is_refused(monkeypatch):
    module = fake_module("sounddevice", *SOUNDDEVICE_CALLS)
    replaced = refuse_audio(monkeypatch, module, SOUNDDEVICE_CALLS)
    assert set(replaced) == set(SOUNDDEVICE_CALLS)
    for name in SOUNDDEVICE_CALLS:
        with pytest.raises(RuntimeError, match=f"sounddevice.{name}"):
            getattr(module, name)(samplerate=16000)


def test_a_name_the_module_lacks_is_left_alone(monkeypatch):
    module = fake_module("sounddevice", "RawInputStream")
    assert refuse_audio(monkeypatch, module, SOUNDDEVICE_CALLS) == ["RawInputStream"]
    assert not hasattr(module, "rec")


def test_a_websocket_connect_is_refused_on_each_entry_point(monkeypatch):
    sync_client = fake_module("websockets.sync.client", "connect")
    asyncio_client = fake_module("websockets.asyncio.client", "connect")
    for module in (sync_client, asyncio_client):
        assert refuse_audio(monkeypatch, module, ("connect", "unix_connect")) == ["connect"]
        with pytest.raises(RuntimeError, match="connect"):
            module.connect("wss://example.invalid/")


def test_the_guard_leaves_sys_modules_alone_when_a_package_is_missing(monkeypatch):
    if installed("sounddevice"):
        pytest.skip("sounddevice is installed here; the point is the absent case")
    before = "sounddevice" in sys.modules
    refuse_all_installed(monkeypatch)
    assert ("sounddevice" in sys.modules) == before  # no stub was registered: ImportError stays an ImportError


def test_the_autouse_fixture_covers_an_installed_package():
    """If sounddevice or websockets is installed (an extra), opening a stream now fails instead of reaching hardware."""
    if installed("sounddevice"):
        import sounddevice

        with pytest.raises(RuntimeError, match="sounddevice"):
            sounddevice.RawInputStream(samplerate=16000, channels=1, dtype="int16")
    if installed("websockets"):
        from websockets.sync.client import connect

        with pytest.raises(RuntimeError, match="connect"):
            connect("wss://example.invalid/")

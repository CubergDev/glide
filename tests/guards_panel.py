"""Guard rows for `glide panel`: no test opens a browser, builds the real assistant, or sends a real provider probe.

The loopback listener needs no row (tests/conftest.py lets a test connect to a port it bound). What reaches past it is
refused here: `webbrowser.open`, `glide.panel.chat.real_assistant` (the assistant that can drive the screen) and
`glide.panel.server.real_probe` (spends tokens). A test supplies a fake through the server's own parameters.
"""

from __future__ import annotations

import pytest


def _refuse(what: str):
    def call(*args, **kwargs):
        raise RuntimeError(f"a test reached the real machine through {what}; use a fake")

    return call


@pytest.fixture(autouse=True)
def no_panel_machine_reach(monkeypatch):
    from glide.panel import chat, cli, server

    monkeypatch.setattr(cli, "webbrowser", type("W", (), {"open": staticmethod(_refuse("webbrowser.open"))}))
    monkeypatch.setattr(chat, "real_assistant", _refuse("glide.panel.chat.real_assistant (the real assistant)"))
    monkeypatch.setattr(server, "real_probe", _refuse("glide.panel.server.real_probe (spends tokens)"))

"""Guard rows for `glide setup`: no test opens a browser, starts Glide as a process, or runs a real provider probe.

The loopback listener itself needs no row: tests/conftest.py already lets a test connect to a port it bound. What reaches
past it is refused here: `webbrowser.open`, the process runner that starts Glide, and the doctor probe that spends tokens.
"""

from __future__ import annotations

import pytest


def _refuse(what: str):
    def call(*args, **kwargs):
        raise RuntimeError(f"a test reached the real machine through {what}; use a fake")

    return call


@pytest.fixture(autouse=True)
def no_setup_machine_reach(monkeypatch):
    import webbrowser

    from glide.providers import doctor
    from glide.setup import cli, server

    monkeypatch.setattr(webbrowser, "open", _refuse("webbrowser.open (a real browser)"))
    monkeypatch.setattr(cli, "webbrowser", type("W", (), {"open": staticmethod(_refuse("webbrowser.open"))}))
    monkeypatch.setattr(server, "run_process", _refuse("glide.setup.server.run_process (starts Glide)"))
    monkeypatch.setattr(doctor, "_probe", _refuse("the live doctor probe (spends tokens)"))

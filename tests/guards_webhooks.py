"""Refusals for the webhook package's calls that reach the machine or the network.

`tests/conftest.py` should import `no_webhook_machine_reach` (it is not edited here); `tests/webhooks/conftest.py`
already does, so the webhook tests are covered either way. Imports are lazy: the guard works with or without the
`webhooks` extra installed.
"""

from __future__ import annotations

import importlib.util

import pytest


def _refuse(what: str):
    def call(*args, **kwargs):
        from conftest import refusal

        raise refusal(what, "use a fake")

    return call


@pytest.fixture(autouse=True)
def no_webhook_machine_reach(monkeypatch):
    """No test starts the listener, drives the desktop, fetches Google's keys, or touches the OS keyring."""
    from glide.webhooks import cli, worker

    monkeypatch.setattr(cli, "serve", _refuse("glide.webhooks.cli.serve (the HTTP listener)"))
    monkeypatch.setattr(worker.ComputerExecutor, "execute", _refuse("the webhook desktop executor"))
    if importlib.util.find_spec("uvicorn"):
        import uvicorn

        monkeypatch.setattr(uvicorn, "run", _refuse("uvicorn.run"))
        monkeypatch.setattr(uvicorn.Server, "serve", _refuse("uvicorn.Server.serve"))
    if importlib.util.find_spec("jwt"):
        import jwt

        monkeypatch.setattr(jwt.PyJWKClient, "fetch_data", _refuse("the PyJWKClient key fetch"))
    if importlib.util.find_spec("keyring"):
        import keyring

        for name in ("get_password", "set_password", "delete_password"):
            monkeypatch.setattr(keyring, name, _refuse(f"keyring.{name}"))

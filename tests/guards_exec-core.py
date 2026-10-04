"""Guard for the structured engine's one door to the machine: `engine.make_backend`.

The engine reaches a browser or the desktop only through `glide.computer.execution.engine.make_backend`, which asks
the backends port for the adapter of the selected provider. A test that runs the engine without replacing it would
open a real browser connection or drive this computer, so the door is shut by default and a test that wants the
engine hands it a fake (`tests/execution_world.drive` does). The lower guards (conftest: sockets, processes, input,
capture) stay in force underneath; this one fails first and says where.

The filename has a hyphen, so conftest loads it by path (see `pytest_configure`).
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def no_real_execution_backend(monkeypatch):
    from glide.computer.execution import engine

    def refuse(browser: str):
        from conftest import refusal

        raise refusal("the engine's real browser or desktop backend", "give it a fake (execution_world.drive)")

    monkeypatch.setattr(engine, "make_backend", refuse)

"""Guards for the pet and point-to-ask: no test opens a real window, builds real audio, or reads the real screen.

An autouse fixture, registered by `tests/conftest.py` (`pytest_configure`) with the other per-owner guards. It adds to
what the shared table already refuses (`glide:pet` names such as `open_window`, `show_point_marker`; `pet_overlay`;
the platform adapters' `point_target` and `point_region`):

- Qt runs on its offscreen platform in every test, so a widget a test makes can never reach the Cocoa window server.
- `core._open_voice`, the one call that builds real audio hardware for the pet, and `pet_ui.show_pet`, which makes
  the QApplication and blocks in its event loop, and `point_ui.show_marker`, which shows a pin on the real screen.

Every refusal is imported lazily and skipped when its module is absent (no `ui` extra, or not yet ported).
"""

from __future__ import annotations

import importlib

import pytest


def _refuse(what: str):
    def call(*args, **kwargs):
        raise RuntimeError(f"a test reached the real machine through {what}; give the code a fake")

    return call


REFUSED = (
    ("glide.ui.core", "_open_voice", "the pet's real voice hardware"),
    ("glide.ui.pet_ui", "show_pet", "the pet's real window and event loop"),
    ("glide.ui.point_ui", "show_marker", "a pin on the real screen"),
)


@pytest.fixture(autouse=True)
def no_pet_machine_reach(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    for module_name, attribute, what in REFUSED:
        try:
            module = importlib.import_module(module_name)
        except ImportError:  # the ui extra is not installed, or the module is not in the tree yet
            continue
        monkeypatch.setattr(module, attribute, _refuse(f"{module_name}.{attribute} ({what})"))

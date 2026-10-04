"""The pet chooses the execution engine through the shared resolver, shows the active one, and its tasks use it."""

from __future__ import annotations

import pytest
from test_assistant_fakes import FakeLLM
from test_pet_core import make_core

from glide.assistant.tasks import ComputerTask
from glide.computer import browser_settings, runner
from glide.computer.platform_adapter import desktop
from glide.computer.runner import RunState
from glide.ui.core import PetEvent, PetView


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for name in ("GLIDE_ENGINE", "GLIDE_RESEARCH_CALLS", "GLIDE_BROWSER_PROVIDER", "GLIDE_BROWSER_ENDPOINT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(browser_settings, "_table", {})


def test_the_core_defaults_to_legacy_takes_a_known_engine_and_says_so(tmp_path):
    core, config = make_core(tmp_path, llm=FakeLLM())
    assert core.engine == "legacy"
    core.engine = "structured"
    assert core.engine == "structured" and config.engine_choice == "structured"
    assert [(e.type, e.data) for e in core.drain()] == [("engine", {"name": "structured"})]
    core.close()


@pytest.mark.parametrize("name", ["turbo", "", "Structured", None])
def test_the_core_refuses_anything_else_in_one_line_and_changes_nothing(tmp_path, name):
    core, config = make_core(tmp_path, llm=FakeLLM())
    with pytest.raises(ValueError) as caught:
        core.engine = name
    assert "\n" not in str(caught.value) and core.engine == "legacy" and not core.drain()
    assert getattr(config, "engine_choice", None) is None
    core.close()


def test_the_core_starts_where_the_environment_puts_it_and_survives_a_bad_one(tmp_path, monkeypatch):
    monkeypatch.setenv("GLIDE_ENGINE", "structured")
    core, _ = make_core(tmp_path, llm=FakeLLM())
    assert core.engine == "structured"
    monkeypatch.setenv("GLIDE_ENGINE", "turbo")
    assert core.engine == "legacy"
    core.close()


def test_a_task_the_pet_starts_runs_the_engine_it_chose(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(
        runner,
        "run",
        lambda cfg, ctx_factory, classifier_factory=None, control=None: seen.append(cfg.engine) or RunState(outcome="dry run"),
    )
    monkeypatch.setattr(desktop, "accessibility_trusted", lambda: True)
    core, config = make_core(tmp_path, llm=FakeLLM())
    for name in ("legacy", "structured"):
        core.engine = name
        task = ComputerTask("goal", act=False, config=config, folder=tmp_path / name)
        task.start()
        task.wait()
    assert seen == ["legacy", "structured"]
    core.close()


def test_the_view_follows_the_engine_event():
    view = PetView()
    assert view.engine == "legacy"
    view.apply(PetEvent("engine", {"name": "structured"}))
    assert view.engine == "structured" and view.lines[-1] == "engine: structured"


# -- the window ------------------------------------------------------------------------------------------

pytest.importorskip("PySide6.QtWidgets")

from pet_fakes import FakeCore, qt_application, window_factory  # noqa: E402


@pytest.fixture
def pet_window():
    yield from window_factory(qt_application())


def test_the_window_shows_the_active_engine_and_the_toggle_reaches_the_core(pet_window):
    core = FakeCore()
    window = pet_window(core)
    assert window.engine_badge.text() == "Engine: legacy" and window.badges.isHidden()
    window.engine_box.setCurrentText("structured")
    assert core.engine == "structured" and window.engine_badge.text() == "Engine: structured (opt-in)"
    assert not window.badges.isHidden() and window.badges.text() == "ENGINE: structured"
    window.engine_box.setCurrentText("legacy")
    assert core.engine == "legacy" and window.badges.isHidden()


def test_a_core_that_refuses_leaves_the_window_on_the_engine_in_force(pet_window):
    class Refusing(FakeCore):
        armed = False

        @property
        def engine(self):
            return "legacy"

        @engine.setter
        def engine(self, name):
            if self.armed:
                raise ValueError("The engine must be one of: legacy, structured")

    core = Refusing()
    window = pet_window(core)
    core.armed = True
    window.engine_box.setCurrentText("structured")
    assert window.engine_box.currentText() == "legacy" and window.engine_badge.text() == "Engine: legacy"
    assert window.view.lines[-1].startswith("The engine")


def test_the_pet_command_line_takes_the_engine_and_refuses_a_bad_one(tmp_path, monkeypatch):
    from glide.ui import pet

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("GLIDE_CONFIG", raising=False)
    opened = []
    monkeypatch.setattr(pet, "open_window", lambda core, config: (opened.append(core), 0)[1])
    assert (
        pet.main(["--workspace", str(tmp_path)]) == 0 and pet.main(["--workspace", str(tmp_path), "--engine", "structured"]) == 0
    )
    assert [core.engine for core in opened] == ["legacy", "structured"]
    for core in opened:
        core.close()
    with pytest.raises(SystemExit) as error:
        pet.main(["--workspace", str(tmp_path), "--engine", "turbo"])
    assert error.value.code == 2 and len(opened) == 2

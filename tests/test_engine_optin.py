"""The execution engine is chosen in one place (`computer.config.engine`, through `features.engine_for`) and every front end
reaches it: default legacy, `[computer] engine` in glide.toml, GLIDE_ENGINE over it, and --engine over both."""

from __future__ import annotations

import tomllib
from types import SimpleNamespace

import pytest
from test_assistant_cli import (
    EOF_MARK,
    Keys,
    acting_config,
    run,
)
from test_cli_tree import FakeLoop, doctor, example_block, isolated, no_extras, real_config  # noqa: F401

from glide import cli, features
from glide.computer import browser_settings, runner
from glide.computer import config as computer_config
from glide.computer.platform_adapter import desktop
from glide.computer.runner import RunState
from glide.providers.config import GlideConfig


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for name in ("GLIDE_ENGINE", "GLIDE_RESEARCH_CALLS", "GLIDE_BROWSER_PROVIDER", "GLIDE_BROWSER_ENDPOINT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(browser_settings, "_table", {})


@pytest.fixture
def loop_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(
        runner,
        "run",
        lambda cfg, ctx_factory, classifier_factory=None, control=None: calls.append(cfg) or RunState(outcome="dry run"),
    )
    monkeypatch.setattr(desktop, "accessibility_trusted", lambda: True)
    return calls


# -- the resolver ----------------------------------------------------------------------------------------


def test_the_default_is_legacy():
    assert computer_config.DEFAULT_ENGINE == "legacy" and computer_config.engine({}, {}) == "legacy"


def test_the_flag_beats_the_environment_beats_the_file_beats_the_default():
    table = {"engine": "structured"}
    assert computer_config.engine(table, {}) == "structured"
    assert computer_config.engine(table, {"GLIDE_ENGINE": "legacy"}) == "legacy"
    assert computer_config.engine(table, {"GLIDE_ENGINE": "legacy"}, flag="structured") == "structured"
    assert computer_config.engine({}, {"GLIDE_ENGINE": " structured "}) == "structured"
    assert computer_config.engine({}, {"GLIDE_ENGINE": ""}) == "legacy"


@pytest.mark.parametrize(
    ("table", "env", "flag"),
    [
        ({"engine": "turbo"}, {}, None),
        ({}, {"GLIDE_ENGINE": "turbo"}, None),
        ({}, {}, "turbo"),
        ({"engine": 1}, {}, None),
        ({"mode": "x"}, {}, None),
    ],
)
def test_a_bad_value_is_a_clear_one_line_error(table, env, flag):
    with pytest.raises(ValueError) as caught:
        computer_config.engine(table, env, flag)
    assert "\n" not in str(caught.value) and "turbo" not in str(caught.value)


def test_a_file_names_its_engine_and_the_example_is_valid(tmp_path):
    text = example_block("computer")
    assert tomllib.loads(text) == {"computer": {"engine": "legacy"}}
    path = tmp_path / "glide.toml"
    path.write_text('[computer]\nengine = "structured"\n')
    config = SimpleNamespace(source=str(path))
    assert features.engine_for(config) == "structured"
    assert features.engine_for(config, "legacy") == "legacy"
    config.engine_choice = "legacy"  # what the app's and the pet's toggle sets
    assert features.engine_for(config) == "legacy"


def test_the_computer_table_is_known_to_the_configuration():
    assert GlideConfig.from_toml('[computer]\nengine = "structured"\n', env={}).warnings == []


# -- every front end: the setting arrives in RunConfig -------------------------------------------------------


def test_ask_default_is_legacy_and_the_flag_arrives(monkeypatch, loop_calls):
    assert run(["ask", "open", "safari"], monkeypatch, acting_config())[0] == 0
    assert run(["ask", "--engine", "structured", "open", "safari"], monkeypatch, acting_config())[0] == 0
    legacy, structured = loop_calls
    assert legacy.engine == "legacy" and structured.engine == "structured"
    assert structured.execution_browser and structured.research_calls == computer_config.DEFAULT_RESEARCH_CALLS


def test_chat_takes_the_flag(monkeypatch, loop_calls):
    keys = Keys("open safari", EOF_MARK, settle=True)
    assert run(["chat", "--engine", "structured"], monkeypatch, acting_config(), keys=keys)[0] == 0
    assert [cfg.engine for cfg in loop_calls] == ["structured"]


def test_the_environment_arrives_without_a_flag(monkeypatch, loop_calls):
    monkeypatch.setenv("GLIDE_ENGINE", "structured")
    run(["ask", "open", "safari"], monkeypatch, acting_config())
    assert [cfg.engine for cfg in loop_calls] == ["structured"]


def test_voice_puts_the_flag_where_its_tasks_read_it(monkeypatch):
    seen = []
    monkeypatch.setattr(cli, "_load", lambda path: acting_config())
    monkeypatch.setattr(cli, "_voice_loop", lambda config, io, act: seen.append(config.engine_choice) or FakeLoop())
    assert cli.main(["voice", "--engine", "structured"]) == 0 and cli.main(["voice"]) == 0
    assert seen == ["structured", None]


def test_a_bad_engine_stops_every_front_end_in_one_line_before_anything_starts(monkeypatch, loop_calls):
    config = acting_config()
    err = []
    for argv in (
        ["ask", "--engine", "turbo", "x"],
        ["chat", "--engine", "turbo"],
        ["voice", "--engine", "turbo"],
        ["listen", "--engine", "turbo"],
    ):
        code, terminal = run(argv, monkeypatch, config)
        assert code == 2
        err += terminal.err.getvalue().strip().splitlines()
    assert len(err) == 4 and all(line.startswith("glide: The engine") for line in err) and not loop_calls


def test_a_bad_setting_in_the_browser_table_is_reported_by_the_task_not_run(monkeypatch, tmp_path, loop_calls):
    path = tmp_path / "bad.toml"
    path.write_text('[browser]\nprovider = "nope"\n')
    config = acting_config()
    config.source = str(path)
    code, terminal = run(["ask", "--engine", "structured", "open", "safari"], monkeypatch, config)
    assert code == 1 and not loop_calls and "task not configured" in terminal.out.getvalue()


# -- doctor and the run header ---------------------------------------------------------------------------------


def test_doctor_says_which_engine_is_active(isolated, monkeypatch, capsys, no_extras):  # noqa: F811
    _, out, _ = doctor(monkeypatch, real_config(isolated, ""), capsys)
    assert "engine: legacy" in out
    _, out, _ = doctor(monkeypatch, real_config(isolated, '\n[computer]\nengine = "structured"\n'), capsys)
    assert "engine: structured" in out
    monkeypatch.setenv("GLIDE_ENGINE", "turbo")
    code, out, _ = doctor(monkeypatch, real_config(isolated, ""), capsys)
    assert code == 1 and "engine: error: The engine" in out


def test_the_computer_run_header_names_the_engine(monkeypatch, capsys):
    from glide.computer import cli as computer_cli

    monkeypatch.setattr(computer_cli.config, "load_dotenv", lambda _: None)
    monkeypatch.setattr(computer_cli.config, "writer_vision", lambda: True)
    monkeypatch.setattr(computer_cli, "make_writer", lambda *a: None)
    monkeypatch.setattr(computer_cli, "desktop", SimpleNamespace(accessibility_trusted=lambda: True, abort_hint=lambda: "x"))
    seen = []
    monkeypatch.setattr(
        computer_cli, "run", lambda cfg, f, **kw: seen.append(cfg) or SimpleNamespace(outcome="done", failure="", answer=None)
    )
    config = SimpleNamespace(classifier=lambda: None, scrub=lambda t: t)
    assert computer_cli.main(["g", "--out", "x"], config) == 0
    assert "engine: legacy" in capsys.readouterr().out and seen[-1].engine == "legacy"
    assert computer_cli.main(["g", "--out", "x", "--engine", "structured"], config) == 0
    assert "engine: structured" in capsys.readouterr().out and seen[-1].engine == "structured"
    monkeypatch.setenv("GLIDE_ENGINE", "structured")
    assert computer_cli.main(["g", "--out", "x"], config) == 0 and seen[-1].engine == "structured"
    assert computer_cli.main(["g", "--out", "x", "--engine", "turbo"], config) == 2


def test_a_structured_run_records_its_engine_in_run_json_without_content(monkeypatch, tmp_path):
    import json

    from execution_world import Computer, Jev, Reasoner, drive, response

    from glide.computer.execution.contracts import Milestone

    step = Milestone("name", "Set the name", "field_value", target="Name", value="Quarterly report")
    drive(monkeypatch, tmp_path, Computer(), Reasoner([response(step)]), Jev("plan"), act=False)
    text = (tmp_path / "run.json").read_text()
    assert json.loads(text)["engine"] == "structured" and "Quarterly report" not in text

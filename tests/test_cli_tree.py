"""The one `glide` command tree, in process through `main(argv)`, over fakes.

Nothing here starts a server, a worker, an audio device, a subprocess or a run: every command that would is
reached through a seam this file replaces, and `tests/conftest.py` refuses the real thing if one slips through.
`no_extras` hides every optional package, so a command that imported one it should not have fails the test.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_assistant_fakes import FakeConfig, FakePlayer

from glide import cli, features
from glide.assistant.audio_io import AudioUnavailable
from glide.computer import browser_settings
from glide.computer import cli as computer_cli
from glide.computer import config as computer_config
from glide.computer.control import RunControl
from glide.mcp.config import McpSettings
from glide.memory.settings import MemorySettings, SettingsError
from glide.providers.config import ConfigError, GlideConfig
from glide.speech.session import build_voice
from glide.speech.settings import SpeechSettings

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = ROOT / "glide.toml.example"
OPTIONAL = ("fastapi", "uvicorn", "starlette", "jwt", "cryptography", "numpy", "onnxruntime", "sounddevice", "websockets", "mcp")
LAZY = (
    "glide.computer.cli",
    "glide.memory.cli",
    "glide.mcp.cli",
    "glide.webhooks.cli",
    "glide.webhooks.worker",
    "glide.speech.session",
    "glide.speech.audio",
    "glide.speech.vad",
)


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    """No .env, no glide.toml, no webhook file and no switch from the machine running the tests."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("glide.computer.config.load_dotenv", lambda path: None)
    for name in ("GLIDE_CONFIG", "GLIDE_MEMORY", "GLIDE_MEMORY_AUTO", "GLIDE_DATA_DIR", features.WEBHOOKS_ENV):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


@pytest.fixture
def no_extras(monkeypatch):
    for name in OPTIONAL:
        monkeypatch.setitem(sys.modules, name, None)  # an import of it raises ImportError


@pytest.fixture
def never_loads(monkeypatch):
    """Commands that are other modules' own must not load glide.toml's providers first."""

    def boom(path):
        raise AssertionError("the providers' configuration was loaded")

    monkeypatch.setattr(cli, "_load", boom)


def calls(monkeypatch, target: str, result: int = 0) -> list:
    seen: list = []
    monkeypatch.setattr(target, lambda *args, **kw: seen.append((args, kw)) or result)
    return seen


# -- help works with nothing installed, and imports nothing it does not need -----------------------------


def test_help_works_with_no_extras_and_imports_none_of_the_commands(no_extras, monkeypatch, capsys):
    for name in LAZY:
        monkeypatch.delitem(sys.modules, name, raising=False)
    for argv in (["--help"], ["webhooks", "--help"]):
        with pytest.raises(SystemExit) as caught:
            cli.main(argv)
        assert caught.value.code == 0
    out = capsys.readouterr().out
    for command in (
        "ask",
        "listen",
        "voice",
        "chat",
        "doctor",
        "status",
        "computer",
        "inspect",
        "memory",
        "mcp",
        "webhooks",
        "serve",
        "work",
    ):
        assert re.search(rf"^\s+{command}\s", out, re.M), command
    assert [name for name in LAZY if name in sys.modules] == []


@pytest.mark.parametrize(
    "argv",
    [
        ["memory", "--help"],
        ["mcp", "--help"],
        ["computer", "--help"],
        ["webhooks", "serve", "--help"],
        ["webhooks", "work", "--help"],
    ],
)
def test_help_of_each_command_is_that_commands_own_and_needs_no_configuration(argv, no_extras, never_loads, capsys):
    with pytest.raises(SystemExit) as caught:
        cli.main(argv)
    assert caught.value.code == 0
    assert f"usage: glide {' '.join(argv[:-1])}" in capsys.readouterr().out


def test_an_unknown_command_or_an_extra_word_on_a_plain_command_is_a_usage_error(capsys):
    for argv in (["frobnicate"], ["status", "extra"], ["webhooks"], ["webhooks", "frobnicate"]):
        with pytest.raises(SystemExit) as caught:
            cli.main(argv)
        assert caught.value.code == 2


# -- each command dispatches ---------------------------------------------------------------------------


def test_memory_and_mcp_get_everything_after_the_command_and_the_glide_toml_the_person_named(never_loads, monkeypatch):
    memory = calls(monkeypatch, "glide.memory.cli.main")
    mcp = calls(monkeypatch, "glide.mcp.cli.main")
    assert cli.main(["memory", "status"]) == 0
    assert cli.main(["--config", "a.toml", "memory", "remember", "k", "text", "--level", "user"]) == 0
    assert cli.main(["memory", "--config", "b.toml", "status"]) == 0
    assert cli.main(["--config", "a.toml", "mcp", "serve", "--user", "me"]) == 0
    assert [args[0] for args, _ in memory] == [
        ["status"],
        ["--config", "a.toml", "remember", "k", "text", "--level", "user"],
        ["--config", "b.toml", "status"],
    ]
    assert [args[0] for args, _ in mcp] == [["--config", "a.toml", "serve", "--user", "me"]]


def test_a_delegate_s_exit_status_is_the_commands(never_loads, monkeypatch):
    calls(monkeypatch, "glide.memory.cli.main", result=2)
    assert cli.main(["memory", "status"]) == 2


def test_webhooks_serve_defaults_its_file_from_the_environment_then_glide_toml_then_webhooks_json(
    isolated, never_loads, monkeypatch
):
    serve = calls(monkeypatch, "glide.webhooks.cli.main")
    monkeypatch.setattr(features, "missing", lambda modules: [])
    assert cli.main(["webhooks", "serve", "--check-config"]) == 0
    toml = isolated / "glide.toml"
    toml.write_text('[webhooks]\nconfig = "from-toml.json"\n')
    assert cli.main(["--config", str(toml), "webhooks", "serve", "--port", "9"]) == 0
    monkeypatch.setenv(features.WEBHOOKS_ENV, "from-env.json")
    assert cli.main(["--config", str(toml), "webhooks", "serve"]) == 0
    assert cli.main(["webhooks", "serve", "--config", "mine.json"]) == 0  # the person's own file is never doubled
    assert [args[0] for args, _ in serve] == [
        ["--config", "webhooks.json", "--check-config"],
        ["--config", "from-toml.json", "--port", "9"],
        ["--config", "from-env.json"],
        ["--config", "mine.json"],
    ]


def test_webhooks_serve_names_the_extra_when_it_is_missing_and_starts_nothing(no_extras, never_loads, monkeypatch, capsys):
    serve = calls(monkeypatch, "glide.webhooks.cli.main")
    assert cli.main(["webhooks", "serve"]) == 2
    err = capsys.readouterr().err
    assert serve == [] and "uv sync --extra webhooks" in err and "fastapi" in err and err.count("\n") == 1


def test_webhooks_serve_reports_a_wrong_webhooks_table_in_one_line(isolated, never_loads, monkeypatch, capsys):
    serve = calls(monkeypatch, "glide.webhooks.cli.main")
    monkeypatch.setattr(features, "missing", lambda modules: [])
    toml = isolated / "glide.toml"
    toml.write_text('[webhooks]\nconfigg = "x"\n')
    assert cli.main(["--config", str(toml), "webhooks", "serve"]) == 2
    assert serve == [] and "unknown key 'configg'" in capsys.readouterr().err


def test_webhooks_work_gets_the_glide_toml_as_its_config_and_needs_no_extra(no_extras, never_loads, monkeypatch):
    work = calls(monkeypatch, "glide.webhooks.worker.main")
    assert cli.main(["--config", "g.toml", "webhooks", "work", "--server", "s", "--agent", "a", "--once"]) == 0
    assert work[0][0][0] == ["--config", "g.toml", "--server", "s", "--agent", "a", "--once"]


def test_computer_loads_the_configuration_once_and_hands_it_over(monkeypatch):
    config = FakeConfig()
    loaded = []
    monkeypatch.setattr(cli, "_load", lambda path: loaded.append(path) or config)
    computer = calls(monkeypatch, "glide.computer.cli.main", result=0)
    assert cli.main(["--config", "g.toml", "computer", "open the calendar", "--steps", "3", "--act"]) == 0
    assert loaded == ["g.toml"] and config.closed
    assert computer == [((["open the calendar", "--steps", "3", "--act"], config), {})]
    assert cli.main(["computer", "--config", "h.toml", "go"]) == 0
    assert loaded == ["g.toml", "h.toml"]


def test_inspect_needs_no_providers_and_gets_its_words(never_loads, monkeypatch):
    inspect = calls(monkeypatch, "glide.computer.cli.inspect")
    assert cli.main(["inspect", "find the button", "--no-open", "--countdown", "0"]) == 0
    assert inspect == [((["find the button", "--no-open", "--countdown", "0"],), {})]


def test_the_old_command_names_are_the_same_commands(monkeypatch):
    seen = []
    monkeypatch.setattr(cli, "main", lambda argv: seen.append(argv) or 0)
    assert cli.computer_main(["a goal", "--act"]) == 0
    assert cli.inspect_main(["--no-open"]) == 0
    monkeypatch.setattr(sys, "argv", ["glide-computer", "x"])
    cli.computer_main()
    assert seen == [["computer", "a goal", "--act"], ["inspect", "--no-open"], ["computer", "x"]]


def test_every_entry_point_names_a_function_that_exists():
    scripts = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["scripts"]
    assert set(scripts) == {"glide", "glide-computer", "glide-inspect", "glide-webhooks", "glide-webhook-worker"}
    for name, target in scripts.items():
        module, _, function = target.partition(":")
        assert callable(getattr(__import__(module, fromlist=[function]), function)), name
    assert scripts["glide-computer"] == "glide.cli:computer_main" and scripts["glide-inspect"] == "glide.cli:inspect_main"


def test_memory_and_mcp_status_run_for_real_through_the_tree_and_create_nothing(isolated, never_loads, capsys):
    assert cli.main(["memory", "status"]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["enabled"] is False and status["exists"] is False
    assert not Path(status["database"]).parent.exists()
    assert cli.main(["mcp", "status"]) == 0
    assert json.loads(capsys.readouterr().out) == {"server_memory": "off", "client_servers": []}


def test_webhooks_serve_reaches_the_real_launcher_and_a_bad_file_starts_nothing(isolated, never_loads, capsys):
    for module in ("fastapi", "uvicorn", "jwt", "cryptography"):
        pytest.importorskip(module)
    (isolated / "webhooks.json").write_text("{}")
    with pytest.raises(SystemExit) as caught:
        cli.main(["webhooks", "serve", "--check-config"])
    assert caught.value.code == 2 and "Could not configure the webhook service" in capsys.readouterr().err


# -- glide computer, for real, over fakes ---------------------------------------------------------------


def computer_run(monkeypatch, config, *flags, outcome="dry run", trusted=True):
    seen = {}

    def fake_run(cfg, ctx_factory, **kwargs):
        seen.update(cfg=cfg, ctx_factory=ctx_factory, **kwargs)
        return SimpleNamespace(outcome=outcome, failure="", answer=None)

    monkeypatch.setattr(computer_cli, "run", fake_run)
    monkeypatch.setattr(computer_cli.desktop, "accessibility_trusted", lambda: trusted)
    monkeypatch.setattr(cli, "_load", lambda path: config)
    return cli.main(["computer", "a goal", *flags]), seen


@pytest.mark.parametrize(("flags", "recorded"), [([], False), (["--record-content"], True)])
def test_computer_runs_over_the_provider_chains_with_a_run_control(monkeypatch, tmp_path, flags, recorded):
    config = FakeConfig(writer="the writer")
    monkeypatch.setattr(computer_cli, "make_writer", lambda given: given.writer())
    code, seen = computer_run(monkeypatch, config, "--out", str(tmp_path / "run"), *flags)
    assert code == 0
    assert seen["classifier_factory"] == config.classifier  # the chain, with failover and visible switches, not a bare client
    assert isinstance(seen["control"], RunControl) and seen["cfg"].record_content is recorded and seen["cfg"].act is False


def test_computer_says_when_there_is_no_writer_and_still_runs(monkeypatch, capsys):
    monkeypatch.setattr(computer_cli, "make_writer", lambda given: None)
    code, seen = computer_run(monkeypatch, FakeConfig())
    assert code == 0 and "writer disabled" in capsys.readouterr().out and seen["cfg"]


def test_computer_act_needs_accessibility_and_an_aborted_run_exits_130(monkeypatch, capsys):
    monkeypatch.setattr(computer_cli, "make_writer", lambda given: None)
    code, seen = computer_run(monkeypatch, FakeConfig(), "--act", trusted=False)
    assert code == 2 and seen == {} and "Accessibility" in capsys.readouterr().err
    assert computer_run(monkeypatch, FakeConfig(), outcome="aborted (stopped)")[0] == 130


def test_computer_with_no_usable_classifier_says_so_and_scrubs_the_message(monkeypatch, capsys):
    config = FakeConfig(secret="sk-test-SECRET-9876543210")
    monkeypatch.setattr(computer_cli, "make_writer", lambda given: None)

    def fail(cfg, ctx_factory, **kwargs):
        raise ConfigError("no usable classifier provider. sk-test-SECRET-9876543210")

    monkeypatch.setattr(computer_cli, "run", fail)
    monkeypatch.setattr(computer_cli.desktop, "accessibility_trusted", lambda: True)
    monkeypatch.setattr(cli, "_load", lambda path: config)
    assert cli.main(["computer", "a goal"]) == 2
    err = capsys.readouterr().err
    assert "no usable classifier provider" in err and "SECRET" not in err


# -- glide voice -------------------------------------------------------------------------------------------


class FakeLoop:
    def __init__(self, failure=None, interrupt=False):
        self.assistant = SimpleNamespace(stop=lambda: self.stops.append("task"), close=lambda: self.stops.append("assistant"))
        self.failure, self.interrupt, self.stops, self.ran = failure, interrupt, [], False

    def run(self):
        self.ran = True
        if self.interrupt:
            raise KeyboardInterrupt

    def stop(self):
        self.stops.append("loop")


def voice(monkeypatch, loop, *flags):
    built = []
    monkeypatch.setattr(cli, "_load", lambda path: FakeConfig())
    monkeypatch.setattr(cli, "_voice_loop", lambda config, io, act: built.append((io, act)) or loop)
    monkeypatch.setattr(cli, "_player", lambda on_error: pytest.fail("a second speaker was opened"))
    return cli.main(["voice", *flags]), built


def test_voice_builds_the_loop_without_a_speaker_of_its_own_and_cleans_up(monkeypatch, capsys):
    loop = FakeLoop()
    code, built = voice(monkeypatch, loop)
    assert code == 0 and loop.ran and built[0][1] is False and built[0][0].player is None
    assert loop.stops == ["loop", "assistant"]
    assert "ACT MODE" not in capsys.readouterr().err
    code, built = voice(monkeypatch, FakeLoop(), "--act")
    assert code == 0 and built[0][1] is True and "ACT MODE" in capsys.readouterr().err


def test_voice_leaves_cleanly_on_ctrl_c_and_exits_one_when_the_microphone_failed(monkeypatch):
    loop = FakeLoop(interrupt=True)
    assert voice(monkeypatch, loop)[0] == 0 and loop.stops == ["task", "loop", "assistant"]
    assert voice(monkeypatch, FakeLoop(failure="microphone lost"))[0] == 1


def test_voice_without_the_speech_extra_is_one_actionable_line(no_extras, monkeypatch, capsys):
    config = FakeConfig()
    config.voice = SpeechSettings()
    monkeypatch.setattr(cli, "_load", lambda path: config)
    assert cli.main(["voice"]) == 2  # the real `build_voice`: no sound device can be had
    err = capsys.readouterr().err
    assert "glide voice:" in err and "uv sync --extra speech" in err and "Traceback" not in err
    assert config.calls.llm == 0 and config.calls.tts == 0  # nothing was asked of a provider


def test_voice_reports_a_detector_that_cannot_be_built(monkeypatch, capsys):
    from glide.speech.vad import VadError

    def boom(config, io, act):
        raise VadError("Silero needs numpy and onnxruntime (the speech extra)")

    monkeypatch.setattr(cli, "_load", lambda path: FakeConfig())
    monkeypatch.setattr(cli, "_voice_loop", boom)
    assert cli.main(["voice"]) == 2 and "uv sync --extra speech" in capsys.readouterr().err
    monkeypatch.setattr(cli, "_voice_loop", lambda *a: (_ for _ in ()).throw(AudioUnavailable("no PortAudio")))
    assert cli.main(["voice"]) == 2


def test_the_speech_table_of_a_real_file_is_what_build_voice_is_given(monkeypatch):
    config = GlideConfig.from_toml('[speech]\nsilence_ms = 800\nmerge_window_s = 1.0\nidle_s = 5.0\nvad = "energy"\n', env={})
    device = FakePlayer()
    loop = build_voice(FakeConfig(), config.voice, device=device, vad=lambda frame: 0.0)
    assert loop.assistant.io.player is device
    assert loop._merge_frames == 32 and loop._idle_s == 5.0 and loop._detector.silence_frames == 25
    loop.assistant.close()


# -- glide doctor: the feature lines ---------------------------------------------------------------------

PROVIDERS = """
[providers.alpha]
kind = "openai_compat"
base_url = "http://localhost:1/v1"
api_key_env = "ALPHA_API_KEY"

[llm.fast]
chain = ["alpha:m1"]
[llm.smart]
chain = ["alpha:m1"]
[stt]
chain = ["alpha:m1"]
[tts]
chain = ["macos_say"]
[classifier]
chain = ["llm.fast"]
"""


def real_config(tmp_path, extra: str) -> GlideConfig:
    """A real configuration read from a file in `tmp_path`, with fake adapters in place of the vendors."""
    path = tmp_path / "glide.toml"
    path.write_text(PROVIDERS + extra)

    def fake(spec, model, key, options):
        return SimpleNamespace(name=spec.name, model=model, sample_rate=22050)

    builders = {(job, kind): fake for job, kind in (("llm", "openai_compat"), ("stt", "openai_compat"), ("tts", "macos_say"))}
    builders[("classifier", "typesafe")] = fake
    return GlideConfig.from_toml(path.read_text(), env={"ALPHA_API_KEY": "k" * 20}, source=str(path), builders=builders)


def doctor(monkeypatch, config, capsys):
    monkeypatch.setattr(cli, "_load", lambda path: config)
    code = cli.main(["doctor"])
    out = capsys.readouterr().out
    return code, out, {m.group(1): m.group(2) for m in re.finditer(r"^  (voice|memory|webhooks|mcp)\s+(.*)$", out, re.M)}


def test_doctor_reports_voice_memory_webhooks_and_mcp_and_exits_zero_when_all_is_off(isolated, monkeypatch, capsys, no_extras):
    code, out, lines = doctor(monkeypatch, real_config(isolated, ""), capsys)
    assert code == 0 and "ready" in out
    assert set(lines) == {"voice", "memory", "webhooks", "mcp"}
    assert lines["voice"].startswith("vad auto: loudness") and "uv sync --extra speech" in lines["voice"]
    assert lines["memory"].startswith("off") and lines["webhooks"].startswith("off")
    assert "none started" in lines["mcp"] and "server_memory off" in lines["mcp"]


def test_doctor_shows_what_is_switched_on_and_creates_nothing(isolated, monkeypatch, capsys):
    (isolated / "webhooks.json").write_text(json.dumps({"enabled": True, "sources": []}))
    extra = '[memory]\nenabled = true\ndata_dir = "mem"\n[[mcp.servers]]\nname = "files"\ncommand = "some-server"\n'
    code, _, lines = doctor(monkeypatch, real_config(isolated, extra), capsys)
    assert code == 0
    assert lines["memory"].startswith("on, auto_capture off") and "database mem/memory/memory.sqlite" in lines["memory"]
    assert not (isolated / "mem").exists()  # reading the settings creates no directory and no database
    assert "enabled in webhooks.json" in lines["webhooks"]
    assert "files (none started)" in lines["mcp"]


def test_doctor_uses_the_webhooks_file_the_table_and_the_variable_name(isolated, monkeypatch, capsys):
    (isolated / "other.json").write_text(json.dumps({"enabled": False}))
    _, _, lines = doctor(monkeypatch, real_config(isolated, '[webhooks]\nconfig = "other.json"\n'), capsys)
    assert lines["webhooks"].startswith("off (enabled is not true) in other.json")
    (isolated / "env.json").write_text("not json")
    monkeypatch.setenv(features.WEBHOOKS_ENV, "env.json")
    code, _, lines = doctor(monkeypatch, real_config(isolated, ""), capsys)
    assert code == 1 and lines["webhooks"].startswith("error:") and "env.json" in lines["webhooks"]


@pytest.mark.parametrize(
    ("table", "feature"),
    [
        ("[memory]\nenabled = 1\n", "memory"),
        ('[mcp]\nserver_memory = "everything"\n', "mcp"),
        ('[webhooks]\ncontent = "x"\n', "webhooks"),
        ('[speech]\nvad = "silero"\nvad_model_path = "m.onnx"\nvad_model_sha256 = "' + "0" * 64 + '"\n', "voice"),
    ],
)
def test_doctor_shows_a_wrong_setting_as_an_error_line_and_exits_one(isolated, monkeypatch, capsys, table, feature):
    code, out, lines = doctor(monkeypatch, real_config(isolated, table), capsys)
    assert code == 1 and lines[feature].startswith("error:"), out


def test_doctor_checks_the_voice_model_from_the_file_and_never_fetches_it(isolated, monkeypatch, capsys):
    import glide.speech.vad as vad

    monkeypatch.setattr(vad, "install_model", lambda *a, **k: pytest.fail("a model was downloaded"))
    monkeypatch.setattr(vad, "_download", lambda url: pytest.fail("a model was downloaded"))
    model = isolated / "vad.onnx"
    digest = hashlib.sha256(b"model bytes").hexdigest()
    table = f'[speech]\nvad_model_path = "{model}"\nvad_model_sha256 = "{digest}"\nvad_model_url = "https://example.invalid/vad.onnx"\n'
    config = real_config(isolated, table)
    code, _, lines = doctor(monkeypatch, config, capsys)
    assert code == 0 and "is missing; fetch it with glide.speech.vad.install_model; auto falls back to loudness" in lines["voice"]
    model.write_bytes(b"something else")
    assert "does not match the configured checksum; auto falls back" in doctor(monkeypatch, config, capsys)[2]["voice"]
    model.write_bytes(b"model bytes")
    assert f"model ok ({model})" in doctor(monkeypatch, config, capsys)[2]["voice"]


def test_a_key_in_a_feature_line_is_never_printed(isolated, monkeypatch, capsys):
    monkeypatch.setenv("SOME_API_KEY", "sk-test-SECRET-9876543210")
    (isolated / "webhooks.json").write_text("{}")
    monkeypatch.setenv(features.WEBHOOKS_ENV, str(isolated / "webhooks.json"))
    _, out, _ = doctor(monkeypatch, real_config(isolated, ""), capsys)
    assert "SECRET" not in out


# -- the example file's commented tables, uncommented, through each table's own reader --------------------


def example_block(table: str) -> str:
    """The commented example that starts at `# [table]` in glide.toml.example, uncommented, to the blank line that ends it."""
    lines, found = [], False
    for line in EXAMPLE.read_text().splitlines():
        found = found or line.startswith(f"# [{table}]")
        if found and not line.startswith("#"):
            break
        if found:
            lines.append(line[2:])  # "# key = 1" and a bare "#" alike
    assert lines, f"no commented [{table}] example"
    return "\n".join(lines)


def test_the_commented_speech_example_is_valid_for_both_readers():
    text = example_block("speech")
    text = text.replace("<64 hexadecimal digits>", "ab" * 32)
    config = GlideConfig.from_toml(text, env={})
    assert config.warnings == [] and config.voice.silence_ms == 700 and config.voice.vad == "auto"
    assert config.speech.vad_model()[2] == "ab" * 32


def test_the_commented_memory_example_is_valid_for_memory(isolated):
    settings = MemorySettings.from_mapping(tomllib.loads(example_block("memory"))["memory"], {})
    assert settings.enabled is True and settings.auto_capture is False and settings.data_dir is not None


def test_the_commented_mcp_example_is_valid_for_mcp():
    text = (
        example_block("mcp").replace("<NAME OF A VARIABLE>", "SOME_TOKEN").replace("<a local MCP server command>", "some-server")
    )
    settings = McpSettings.from_mapping(tomllib.loads(text)["mcp"])
    assert (
        settings.server_memory == "off"
        and settings.servers[0].name == "files"
        and settings.servers[0].env_names == ("SOME_TOKEN",)
    )


def test_the_commented_webhooks_example_is_valid_for_the_webhooks_table(isolated):
    path = isolated / "glide.toml"
    path.write_text(example_block("webhooks"))
    assert features.webhooks_file({}, path) == Path("webhooks.json")
    with pytest.raises(SettingsError, match="must be a path"):
        path.write_text('[webhooks]\nconfig = ""\n')
        features.webhooks_file({}, path)


def test_the_commented_browser_and_research_examples_are_valid_for_their_readers():
    text = (
        example_block("browser")
        .replace("<address of the page a search starts from>", "https://search.example.test/")
        .replace("<loopback address and port of the browser's debugging port>", "http://127.0.0.1:9")
        .replace("<loopback address and port of Obscura>", "http://127.0.0.1:10")
    )
    settings = browser_settings.resolve(tomllib.loads(text)["browser"], {})
    assert settings.chain == ("cdp", "obscura") and settings.target == "new" and settings.search_url
    assert settings.session == "glide"
    assert computer_config.research_budget(tomllib.loads(example_block("research"))["research"], {}) == 12


def test_the_example_file_has_no_url_but_the_placeholder_and_no_real_server():
    text = EXAMPLE.read_text()
    assert set(re.findall(r"https?://[^\s\"']+", text)) <= {
        "http://localhost:11434/v1",
        "https://example.invalid/vad.onnx",
    }


# -- D12: one name --------------------------------------------------------------------------------------------

OLD_NAMES = re.compile(
    "|".join(
        ("CLICKER" + "_", "PERMIT" + "_", "clicker" + "-", r"\b" + "clicker" + r"\b", "permit" + "_", r"\b" + "permit" + r"\b")
    ),
    re.I,
)


def test_no_old_product_or_environment_names_are_left_in_the_package_the_tests_or_the_config():
    found = []
    files = [*(ROOT / "glide").rglob("*"), *(ROOT / "tests").rglob("*"), ROOT / "pyproject.toml", EXAMPLE]
    own = {Path(__file__), ROOT / "tests" / "webhooks" / "test_webhooks_literals.py"}  # each names the old names to refuse them
    for path in files:
        if path.suffix in (".py", ".md", ".toml", ".example", ".json", ".txt") and path not in own:
            for number, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                if OLD_NAMES.search(line):
                    found.append(f"{path.relative_to(ROOT)}:{number}: {line.strip()[:80]}")
    assert found == []

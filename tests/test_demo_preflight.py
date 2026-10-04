"""The pure parts of scripts/demo_preflight.py. Nothing here reads the real environment, a real .env or the network."""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("demo_preflight", ROOT / "scripts" / "demo_preflight.py")
preflight = importlib.util.module_from_spec(_spec)
sys.modules["demo_preflight"] = preflight
_spec.loader.exec_module(preflight)

SECRET = "sk-test-value-that-must-never-print"


def test_dotenv_names_keeps_names_and_drops_values():
    text = f"# comment\nOPENAI_API_KEY={SECRET}\nexport GEMINI_API_KEY=\nBAD LINE\nQUOTED='x'  # note\n"
    found = preflight.dotenv_names(text)
    assert found == {"OPENAI_API_KEY": True, "GEMINI_API_KEY": False, "QUOTED": True}
    assert SECRET not in repr(found)


def test_effective_env_prefers_the_process_environment():
    env = preflight.effective_env({"A": "from-process"}, "A=from-file\nB=from-file\n")
    assert env == {"A": "from-process", "B": "from-file"}


def test_names_set_returns_booleans_only():
    state = preflight.names_set(["A", "B", "C"], {"A": SECRET, "B": "   "})
    assert state == {"A": True, "B": False, "C": False}


def test_extras_report_uses_the_finder_and_imports_nothing():
    asked = []

    def find(name):
        asked.append(name)
        return None if name in {"livekit", "numpy"} else object()

    lacking = preflight.extras_report(find)
    assert lacking["aec"] == ["livekit"] and lacking["speech"] == ["numpy"] and lacking["ui"] == []
    assert "numpy" in asked


def test_browser_commands_are_loopback_only_and_use_a_throwaway_profile():
    command = preflight.chrome_command(9333, "brave")
    assert "--remote-debugging-port=9333" in command
    assert "--remote-debugging-address=127.0.0.1" in command
    assert "--user-data-dir=" in command and "Brave Browser" in command
    assert preflight.verify_command(9333) == "curl -s http://127.0.0.1:9333/json/version"


def test_port_of_accepts_only_loopback():
    assert preflight.port_of("http://127.0.0.1:9222") == 9222
    assert preflight.port_of("http://example.com:9222") is None


def test_tools_use_which_only():
    lines = preflight.check_tools(lambda name: "/bin/" + name if name == "uv" else None)
    assert [line.status for line in lines] == [preflight.OK, preflight.INFO, preflight.INFO, preflight.INFO]


def test_blocking_and_render():
    lines = [preflight.Line(preflight.OK, "a", "fine"), preflight.Line(preflight.FAIL, "b", "broken")]
    assert preflight.blocking(lines) and not preflight.blocking(lines[:1])
    assert "broken" in preflight.render(lines)


def test_run_never_prints_a_key_value(tmp_path):
    toml = tmp_path / "glide.toml"
    toml.write_text('[llm.fast]\nchain = ["openai:some-model"]\n', encoding="utf-8")
    lines = preflight.run(str(toml), tmp_path / "missing.env", 9222, environ={"OPENAI_API_KEY": SECRET, "HOME": str(tmp_path)})
    text = preflight.render(lines)
    assert SECRET not in text
    assert "OPENAI_API_KEY" in text  # the name is shown


def test_a_bad_config_is_reported_not_raised(tmp_path):
    lines = preflight.run(str(tmp_path / "nope.toml"), tmp_path / "x.env", 9222, environ={"HOME": str(tmp_path)})
    assert any(line.status == preflight.FAIL and line.topic == "glide.toml" for line in lines)

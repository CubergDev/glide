"""[mcp] settings: nothing is started or connected by reading them; secrets are named, never held."""

import pytest

from glide.mcp import McpServerSpec, McpSettings
from glide.memory.settings import SettingsError


def test_defaults_offer_nothing():
    settings = McpSettings.from_mapping({})
    assert settings.server_memory == "off" and settings.servers == ()
    assert McpSettings.from_mapping(None) == settings


def test_servers_parse_with_env_names_only():
    settings = McpSettings.from_mapping(
        {
            "server_memory": "read",
            "servers": [{"name": "files", "command": "some-server", "args": ["--flag"], "env": ["SOME_TOKEN"], "timeout_s": 5}],
        }
    )
    assert settings.server_memory == "read"
    assert settings.servers == (McpServerSpec("files", "some-server", ("--flag",), ("SOME_TOKEN",), 5.0),)


@pytest.mark.parametrize(
    "table",
    [
        {"server_memory": "all"},
        {"servers": "files"},
        {"servers": ["files"]},
        {"servers": [{"name": "bad name", "command": "x"}]},
        {"servers": [{"name": "a", "command": ""}]},
        {"servers": [{"name": "a"}]},
        {"servers": [{"name": "a", "command": "x"}, {"name": "a", "command": "y"}]},
        {"servers": [{"name": "a", "command": "x", "args": [1]}]},
        {"servers": [{"name": "a", "command": "x", "env": ["TOKEN=abc123"]}]},
        {"servers": [{"name": "a", "command": "x", "env": ["lower-case-with-dash"]}]},
        {"servers": [{"name": "a", "command": "x", "timeout_s": 0}]},
        {"servers": [{"name": "a", "command": "x", "timeout_s": True}]},
        {"servers": [{"name": "a", "command": "x", "timeout_s": 3600.5}]},
        {"servers": [{"name": "a", "command": "x", "timeout_s": -1}]},
        {"servers": [{"name": "a", "command": "x", "timeout_s": "60"}]},
        {"servers": [{"name": "a", "command": "x", "url": "elsewhere"}]},
        {"surprise": 1},
    ],
)
def test_bad_tables_are_errors(table):
    with pytest.raises(SettingsError):
        McpSettings.from_mapping(table)


def test_an_env_value_is_never_echoed():
    with pytest.raises(SettingsError) as caught:
        McpSettings.from_mapping({"servers": [{"name": "a", "command": "x", "env": ["TOKEN=sk-abcdefghijklmnopqrstuvwxyz"]}]})
    assert "sk-" not in str(caught.value)


def test_load_reads_the_toml_table(tmp_path):
    config = tmp_path / "glide.toml"
    config.write_text('[mcp]\nserver_memory = "write"\n[[mcp.servers]]\nname = "files"\ncommand = "some-server"\n')
    settings = McpSettings.load({}, config=config)
    assert settings.server_memory == "write" and settings.servers[0].name == "files"
    empty = tmp_path / "empty.toml"
    empty.write_text("")
    assert McpSettings.load({}, config=empty) == McpSettings()


from mcp_guard_loader import no_real_memory_or_mcp  # noqa: E402, F401  (autouse guard)


def test_the_longest_allowed_timeout_is_an_hour():
    assert (
        McpSettings.from_mapping({"servers": [{"name": "a", "command": "x", "timeout_s": 3600}]}).servers[0].timeout_s == 3600.0
    )

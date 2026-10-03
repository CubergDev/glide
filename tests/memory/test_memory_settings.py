"""Opt-in switches: default off, strict parsing, environment over file, nothing created by reading."""

from pathlib import Path

import pytest

from glide.memory.settings import MemorySettings, SettingsError, default_data_dir, find_config, read_table


def test_default_is_off_and_creates_nothing(tmp_path):
    settings = MemorySettings.from_mapping({}, {})
    assert (settings.enabled, settings.auto_capture, settings.data_dir) == (False, False, None)
    path = settings.database_path({"GLIDE_DATA_DIR": str(tmp_path / "data")})
    assert path == tmp_path / "data" / "memory" / "memory.sqlite"
    assert not (tmp_path / "data").exists()


def test_file_values_and_environment_override():
    table = {"enabled": True, "auto_capture": True}
    assert MemorySettings.from_mapping(table, {}).enabled is True
    assert MemorySettings.from_mapping(table, {"GLIDE_MEMORY": "0"}).enabled is False
    assert MemorySettings.from_mapping({}, {"GLIDE_MEMORY": "1", "GLIDE_MEMORY_AUTO": "on"}).auto_capture is True
    assert MemorySettings.from_mapping(table, {"GLIDE_MEMORY_AUTO": "off"}).auto_capture is False


@pytest.mark.parametrize(
    "table", [{"enabled": "yes"}, {"enabled": 1}, {"auto_capture": "maybe"}, {"data_dir": 5}, {"enable": True}]
)
def test_bad_settings_are_errors_not_guesses(table):
    with pytest.raises(SettingsError):
        MemorySettings.from_mapping(table, {})


def test_bad_environment_value_is_an_error():
    with pytest.raises(SettingsError, match="GLIDE_MEMORY"):
        MemorySettings.from_mapping({}, {"GLIDE_MEMORY": "perhaps"})


def test_data_dir_from_table_wins_over_default(tmp_path):
    settings = MemorySettings.from_mapping({"data_dir": str(tmp_path / "mine")}, {"GLIDE_DATA_DIR": str(tmp_path / "other")})
    assert settings.memory_dir({}) == tmp_path / "mine" / "memory"


def test_default_data_dir_uses_home_and_env(tmp_path):
    assert default_data_dir({"GLIDE_DATA_DIR": str(tmp_path / "x")}) == tmp_path / "x"
    chosen = default_data_dir({}, home=tmp_path)
    assert chosen.is_relative_to(tmp_path) and chosen.name == "glide"


def test_config_discovery_order_and_errors(tmp_path):
    here, home = tmp_path / "here", tmp_path / "home"
    (home / ".config" / "glide").mkdir(parents=True)
    here.mkdir()
    (home / ".config" / "glide" / "glide.toml").write_text("[memory]\nenabled = true\n")
    assert find_config({}, cwd=here, home=home) == home / ".config" / "glide" / "glide.toml"
    (here / "glide.toml").write_text("[memory]\nenabled = false\n")
    assert find_config({}, cwd=here, home=home) == here / "glide.toml"
    named = tmp_path / "named.toml"
    named.write_text("[memory]\nenabled = true\n")
    assert find_config({"GLIDE_CONFIG": str(named)}, cwd=here, home=home) == named
    with pytest.raises(SettingsError, match="does not exist"):
        find_config({"GLIDE_CONFIG": str(tmp_path / "missing.toml")}, cwd=here, home=home)
    assert find_config({}, cwd=tmp_path, home=tmp_path / "nowhere") is None


def test_read_table_and_load(tmp_path):
    config = tmp_path / "glide.toml"
    config.write_text("[memory]\nenabled = true\nauto_capture = false\n[other]\nx = 1\n")
    assert read_table("memory", config) == {"enabled": True, "auto_capture": False}
    assert read_table("memory", None) == {}
    settings = MemorySettings.load({}, config=config)
    assert settings.enabled is True
    config.write_text("[memory\n")
    with pytest.raises(SettingsError, match="not valid TOML"):
        read_table("memory", config)
    config.write_text("memory = 3\n")
    with pytest.raises(SettingsError, match="must be a table"):
        read_table("memory", config)


def test_settings_error_never_echoes_file_content(tmp_path):
    config = tmp_path / "glide.toml"
    config.write_text('[memory]\nsecret_token = "sk-should-not-appear-anywhere"\n')
    with pytest.raises(SettingsError) as caught:
        MemorySettings.load({}, config=config)
    assert "sk-should-not" not in str(caught.value)
    assert isinstance(Path(str(config)), Path)

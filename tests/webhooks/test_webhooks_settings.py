"""D5 opt-in defaults and the configuration boundary. Needs only pydantic."""

import json

import pytest
from pydantic import ValidationError

from glide.webhooks.secret_sources import DictSecrets, KeyringSecrets
from glide.webhooks.settings import ServerSettings, Source, load_settings
from webhooks.helpers import make_settings, settings_data


def test_everything_is_off_by_default():
    data = settings_data()
    for key in ("enabled", "record_content", "database"):
        data.pop(key)
    for source in data["sources"]:
        source.pop("enabled")
    settings = ServerSettings.model_validate_json(json.dumps(data))
    assert settings.enabled is False and settings.record_content is False
    assert all(source.enabled is False for source in settings.sources)
    assert settings.database == ".glide-webhooks/queue.sqlite3"


def test_default_allowed_hosts_do_not_include_a_test_host():
    assert "testserver" not in make_settings().allowed_hosts


def test_providers_cannot_enable_computer_actions():
    data = settings_data()
    data["sources"][0]["allow_actions"] = True  # github
    with pytest.raises(ValidationError):
        ServerSettings.model_validate_json(json.dumps(data))


def test_secrets_come_from_the_named_source_and_must_be_long_enough(monkeypatch):
    source = Source.model_validate_json(json.dumps(settings_data()["sources"][0]))
    assert source.secrets(DictSecrets({"GITHUB_WEBHOOK_SECRET": "x" * 32})) == ("x" * 32,)
    with pytest.raises(ValueError, match="missing or too short"):
        source.secrets(DictSecrets({"GITHUB_WEBHOOK_SECRET": "short"}))
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", "e" * 40)
    assert source.secrets() == ("e" * 40,)


def test_secret_env_names_are_validated():
    data = settings_data()["sources"][0]
    data["key_envs"] = ["lower-case"]
    with pytest.raises(ValidationError):
        Source.model_validate_json(json.dumps(data))


def test_load_settings_error_does_not_echo_the_file(tmp_path):
    path = tmp_path / "webhooks.json"
    path.write_text('{"enabled": true, "sources": [], "marker-private-text": 1}')
    with pytest.raises(ValueError) as error:
        load_settings(path)
    assert "marker-private-text" not in str(error.value)


def test_load_settings_rejects_duplicate_keys(tmp_path):
    path = tmp_path / "webhooks.json"
    path.write_text('{"enabled": false, "enabled": true}')
    with pytest.raises(ValueError):
        load_settings(path)


def test_keyring_adapter_is_an_interface_with_a_fake_backend():
    class Fake:
        def __init__(self):
            self.calls = []

        def get_password(self, service, name):
            self.calls.append((service, name))
            return "value-from-fake" if name == "KNOWN" else None

    fake = Fake()
    store = KeyringSecrets("glide-test", backend=fake)
    assert store.get("KNOWN") == "value-from-fake" and store.get("OTHER") == ""
    assert fake.calls == [("glide-test", "KNOWN"), ("glide-test", "OTHER")]


def test_keyring_failure_does_not_leak_the_account_name():
    class Broken:
        def get_password(self, service, name):
            raise RuntimeError("backend says: " + name)

    with pytest.raises(ValueError) as error:
        KeyringSecrets("svc", backend=Broken()).get("SECRET_ACCOUNT")
    assert "SECRET_ACCOUNT" not in str(error.value)

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


# -- characterisation: every configuration rule the file enforces ---------------------------------------------------------


def source_data(index):
    return settings_data()["sources"][index]  # 0 github, 1 gmail, 2 outlook, 3 standard


def parse(data):
    return Source.model_validate_json(json.dumps(data))


@pytest.mark.parametrize(
    "index,drop",
    [
        (0, "key_envs"),
        (0, "repositories"),
        (1, "mailbox"),
        (1, "subscription"),
        (1, "audience"),
        (1, "service_account"),
        (2, "key_envs"),
        (2, "mailbox_id"),
        (2, "tenant_id"),
        (2, "subscription_id"),
        (2, "subscription_expires_at"),
        (3, "key_envs"),
    ],
)
def test_each_provider_needs_every_binding_it_is_checked_against(index, drop):
    data = source_data(index)
    data.pop(drop)
    with pytest.raises(ValidationError):
        parse(data)


def test_a_subscription_expiry_must_be_a_timestamp_with_its_zone():
    data = source_data(2)
    data["subscription_expires_at"] = "2030-01-01T00:00:00"
    with pytest.raises(ValidationError):
        parse(data)
    data["subscription_expires_at"] = "tomorrow"
    with pytest.raises(ValidationError):
        parse(data)
    data["subscription_expires_at"] = "2030-01-01T00:00:00Z"
    assert parse(data).subscription_expires_at.endswith("Z")


def test_only_a_standard_source_may_allow_actions():
    for index in (0, 1, 2):
        data = source_data(index)
        data["allow_actions"] = True
        with pytest.raises(ValidationError):
            parse(data)
    mcp = {"id": "m", "provider": "mcp", "agent_id": "team", "allow_actions": True}
    with pytest.raises(ValidationError):
        parse(mcp)
    assert parse(source_data(3)).allow_actions is True
    assert all(parse(source_data(i)).allow_actions is False for i in (0, 1, 2))


def test_a_source_is_closed_and_bounded():
    for change in (
        {"extra": 1},
        {"id": "Bad Id"},
        {"agent_id": "../x"},
        {"provider": "slack"},
        {"key_envs": ["A", "B", "C", "D", "E"]},
        {"key_envs": ["A" * 101]},
        {"gmail_max_age_seconds": 299},
        {"gmail_max_age_seconds": 604801},
        {"enabled": "yes"},
    ):
        with pytest.raises(ValidationError):
            parse({**source_data(0), **change})
    assert parse({"id": "m", "provider": "mcp", "agent_id": "team"}).enabled is False


def test_several_secrets_must_all_be_long_enough_and_a_missing_one_fails():
    source = parse({**source_data(0), "key_envs": ["FIRST", "SECOND"]})
    both = DictSecrets({"FIRST": "a" * 32, "SECOND": "b" * 32})
    assert source.secrets(both) == ("a" * 32, "b" * 32)
    for values in ({"FIRST": "a" * 32}, {"FIRST": "a" * 32, "SECOND": "b" * 31}):
        with pytest.raises(ValueError, match="missing or too short"):
            source.secrets(DictSecrets(values))
    assert parse({**source_data(0), "key_envs": ["FIRST"]}).secrets(DictSecrets({"FIRST": "é" * 16})) == (
        "é" * 16,
    )  # bytes, not characters
    with pytest.raises(ValueError, match="missing or too short"):
        parse({**source_data(0), "key_envs": ["FIRST"]}).secrets(DictSecrets({"FIRST": "é" * 15}))


def test_server_settings_reject_duplicates_and_out_of_range_limits():
    base = settings_data()
    for change in (
        {"sources": [base["sources"][0], base["sources"][0]]},
        {"sources": []},
        {"agent_auth": {**base["agent_auth"], "keys": base["agent_auth"]["keys"] * 2}},
        {"agent_auth": {**base["agent_auth"], "keys": []}},
        {"agent_auth": {**base["agent_auth"], "max_token_seconds": 3601}},
        {"agent_auth": {**base["agent_auth"], "max_token_seconds": 59}},
        {"timestamp_tolerance": 301},
        {"timestamp_tolerance": 29},
        {"lease_seconds": 14},
        {"lease_seconds": 301},
        {"body_limit": 1023},
        {"body_limit": 1048577},
        {"max_pending": 0},
        {"allowed_hosts": []},
        {"allowed_hosts": ["ok.example", "*"]},
        {"surprise": True},
    ):
        with pytest.raises(ValidationError):
            ServerSettings.model_validate_json(json.dumps({**base, **change}))
    ServerSettings.model_validate_json(json.dumps({**base, "timestamp_tolerance": 300, "lease_seconds": 300, "body_limit": 1024}))


def test_a_file_that_cannot_be_read_or_parsed_is_one_fixed_error(tmp_path):
    for path in (tmp_path / "missing.json", tmp_path):
        with pytest.raises(ValueError, match="Invalid webhook configuration"):
            load_settings(path)
    broken = tmp_path / "broken.json"
    broken.write_text("not json marker-private-text")
    with pytest.raises(ValueError) as error:
        load_settings(broken)
    assert "marker-private-text" not in str(error.value)
    good = tmp_path / "good.json"
    good.write_text(json.dumps(settings_data()))
    assert load_settings(good).enabled is True


def test_secret_sources_return_empty_for_an_unset_name(monkeypatch):
    from glide.webhooks.secret_sources import EnvSecrets

    monkeypatch.delenv("GLIDE_FIXTURE_UNSET", raising=False)
    monkeypatch.setenv("GLIDE_FIXTURE_SET", "value")
    assert EnvSecrets().get("GLIDE_FIXTURE_UNSET") == "" and EnvSecrets().get("GLIDE_FIXTURE_SET") == "value"
    assert DictSecrets().get("x") == "" and DictSecrets({"x": "y"}).get("x") == "y"
    with pytest.raises(ValueError):
        KeyringSecrets("")


@pytest.mark.parametrize(
    "field,value",
    [
        ("repositories", ["owner/repo/"]),
        ("repositories", ["team/.."]),
        ("repositories", ["noslash"]),
        ("allowed_senders", ["bad login"]),
        ("allowed_senders", ["x/y"]),
    ],
)
def test_github_allowlist_entries_that_could_never_match_are_refused(field, value):
    """PR7-4175624965: a typo here used to pass --check-config and reject every callback at run time."""
    data = source_data(0)
    data[field] = value
    with pytest.raises(ValidationError):
        parse(data)
    data[field] = ["team/repo"] if field == "repositories" else ["octocat", "dependabot[bot]"]
    assert parse(data)


def test_uncertain_retention_defaults_to_fourteen_days_and_is_bounded():
    assert make_settings().uncertain_retention_days == 14
    for days, ok in ((1, True), (90, True), (0, False), (91, False)):
        data = {**settings_data(), "uncertain_retention_days": days}
        if ok:
            assert ServerSettings.model_validate_json(json.dumps(data)).uncertain_retention_days == days
        else:
            with pytest.raises(ValidationError):
                ServerSettings.model_validate_json(json.dumps(data))


@pytest.mark.parametrize(("env", "expected"), [("3", 3), ("90", 90), (" 1 ", 1)])
def test_the_environment_overrides_the_retention(tmp_path, monkeypatch, env, expected):
    path = tmp_path / "w.json"
    path.write_text(json.dumps({**settings_data(), "uncertain_retention_days": 20}))
    monkeypatch.setenv("GLIDE_WEBHOOK_UNCERTAIN_RETENTION_DAYS", env)
    assert load_settings(path).uncertain_retention_days == expected
    monkeypatch.delenv("GLIDE_WEBHOOK_UNCERTAIN_RETENTION_DAYS")
    assert load_settings(path).uncertain_retention_days == 20


@pytest.mark.parametrize("env", ["0", "91", "-5", "soon", "1.5"])
def test_a_bad_environment_retention_is_refused(tmp_path, monkeypatch, env):
    path = tmp_path / "w.json"
    path.write_text(json.dumps(settings_data()))
    monkeypatch.setenv("GLIDE_WEBHOOK_UNCERTAIN_RETENTION_DAYS", env)
    with pytest.raises(ValueError, match="Invalid webhook configuration"):
        load_settings(path)

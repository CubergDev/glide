"""The launcher: opt-in, loopback by default, and `--check-config` starts nothing."""

import base64
import json
from pathlib import Path

import pytest

from webhooks.helpers import settings_data

pytest.importorskip("fastapi")
pytest.importorskip("jwt")
pytest.importorskip("cryptography")
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from glide.webhooks import cli


@pytest.fixture
def config(tmp_path, monkeypatch):
    pem = (
        ed25519.Ed25519PrivateKey.generate()
        .public_key()
        .public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
        .decode()
    )
    monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", "g" * 40)
    monkeypatch.setenv("GRAPH_CLIENT_STATE", "o" * 40)
    monkeypatch.setenv("STANDARD_WEBHOOK_KEY", "whsec_" + base64.b64encode(b"a" * 32).decode())

    def write(**changes):
        path = tmp_path / "webhooks.json"
        path.write_text(json.dumps(settings_data(pem=pem, database=str(tmp_path / "q.sqlite3"), **changes)))
        return path

    return write


def test_check_config_is_offline_and_never_serves(config, capsys):
    assert cli.main(["--config", str(config()), "--check-config"]) == 0
    assert "No server or agent started" in capsys.readouterr().out


def test_a_disabled_configuration_does_not_start(config, capsys):
    with pytest.raises(SystemExit) as error:
        cli.main(["--config", str(config(enabled=False))])  # serve is guarded, so reaching it would raise RuntimeError instead
    assert error.value.code == 2


@pytest.mark.expect_refusals
def test_serving_requires_the_listener_which_the_guard_refuses_in_tests(config):
    with pytest.raises(RuntimeError, match="real machine"):
        cli.main(["--config", str(config())])


def test_public_bind_needs_behind_proxy(config):
    with pytest.raises(SystemExit) as error:
        cli.main(["--config", str(config()), "--host", "0.0.0.0"])
    assert error.value.code == 2


def test_missing_secret_is_a_clean_error(config, monkeypatch):
    monkeypatch.delenv("GITHUB_WEBHOOK_SECRET")
    with pytest.raises(SystemExit) as error:
        cli.main(["--config", str(config()), "--check-config"])
    assert error.value.code == 2


def test_importing_the_package_starts_nothing():
    import glide.webhooks
    import glide.webhooks.settings

    source = Path(glide.webhooks.__file__).read_text(encoding="utf-8")
    assert "import uvicorn" not in source and "import fastapi" not in source


# -- characterisation: the launcher's argument rules and its one fixed failure message ------------------------------------


@pytest.mark.parametrize("port", ["0", "65536", "-1"])
def test_the_port_must_be_a_real_port(config, port):
    with pytest.raises(SystemExit) as error:
        cli.main(["--config", str(config()), "--port", port, "--check-config"])
    assert error.value.code == 2


@pytest.mark.parametrize("host", ["example.com", "0.0.0.0", "192.0.2.1", "::", "not a host"])
def test_any_non_loopback_host_needs_the_proxy_flag(config, host):
    with pytest.raises(SystemExit) as error:
        cli.main(["--config", str(config()), "--host", host, "--check-config"])
    assert error.value.code == 2
    assert cli.main(["--config", str(config()), "--host", "127.0.0.1", "--check-config"]) == 0


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "127.9.9.9"])
def test_loopback_hosts_are_local_without_the_flag(config, host, capsys):
    assert cli.main(["--config", str(config()), "--host", host, "--check-config"]) == 0


@pytest.mark.expect_refusals
def test_a_public_bind_with_the_proxy_flag_reaches_the_listener_which_the_guard_refuses(config):
    with pytest.raises(RuntimeError, match="real machine"):
        cli.main(["--config", str(config()), "--host", "0.0.0.0", "--behind-proxy"])


def test_the_listener_receives_the_validated_settings_host_and_port(config, monkeypatch):
    seen = []
    monkeypatch.setattr(cli, "serve", lambda settings, host, port: seen.append((settings.enabled, host, port)))
    assert cli.main(["--config", str(config()), "--port", "9123"]) == 0
    assert seen == [(True, "127.0.0.1", 9123)]


def test_a_failing_listener_or_a_missing_file_is_one_fixed_message(config, monkeypatch, tmp_path, capsys):
    def busy(*args):
        raise OSError("address in use marker-private-text")

    monkeypatch.setattr(cli, "serve", busy)
    for argv in (["--config", str(config())], ["--config", str(tmp_path / "absent.json")]):
        with pytest.raises(SystemExit) as error:
            cli.main(argv)
        assert error.value.code == 2
    err = capsys.readouterr().err
    assert "Could not configure the webhook service" in err and "marker-private-text" not in err


def test_the_default_file_comes_from_the_environment(config, monkeypatch):
    monkeypatch.setenv("GLIDE_WEBHOOK_CONFIG", str(config()))
    assert cli.main(["--check-config"]) == 0
    monkeypatch.setenv("GLIDE_WEBHOOK_CONFIG", str(config(enabled=False)))
    with pytest.raises(SystemExit):
        cli.main(["--check-config"])


@pytest.mark.parametrize("host", ["::1", "::", "2001:db8::1"])
def test_an_ipv6_bind_is_refused_with_a_reason(config, capsys, host):
    """PR7-4175586405: allowed_hosts cannot name an IPv6 literal, so a ::1 listener would refuse every request."""
    with pytest.raises(SystemExit) as error:
        cli.main(["--config", str(config()), "--host", host, "--behind-proxy", "--check-config"])
    assert error.value.code == 2 and "IPv6" in capsys.readouterr().err

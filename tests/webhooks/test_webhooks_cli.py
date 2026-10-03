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

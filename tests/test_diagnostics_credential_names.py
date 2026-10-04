"""R2 audit: prefixed credential names, auth schemes and declared key variables are scrubbed from diagnostics."""

from __future__ import annotations

import pytest

from glide.computer import diagnostics as d
from glide.providers.config import load_config


@pytest.mark.parametrize(
    "text",
    [
        "OPENAI_API_KEY=sk-proj-abcdef123456",
        "client_secret=s3cr3tvalue",
        "access_token=ya29.abcdefg",
        "refresh_token: rt_abcdefg",
        "Authorization: Basic dXNlcjpwYXNz",
        "Authorization: Token abcdefgh",
        "Proxy-Authorization: Basic Zm9vOmJhcg==",
        "MY_OPENAI_KEY=sk-live-ABCDEF123456",
    ],
)
def test_a_prefixed_credential_or_auth_scheme_value_is_removed(text):
    secret = text.split(None, 1)[-1].split("=", 1)[-1].split()[-1].strip()
    for out in (d.scrub_text(text), d.redact(text)):
        assert secret not in out and d.REDACTED in out


def test_ordinary_text_with_key_or_token_words_is_kept():
    assert d.scrub_text("press the key: Enter then token_count=5") == "press the key: Enter then token_count=5"


def test_a_key_variable_the_config_declares_is_removed_whatever_its_name(monkeypatch, tmp_path):
    monkeypatch.setenv("MY_OPENAI_KEY", "sk-live-ABCDEF123456")
    monkeypatch.setenv("GROQ_KEY", "gsk_ZZZZZZZZZZZZ")
    monkeypatch.setenv("ODDNAME", "plainvalue-9988")
    assert "sk-live-ABCDEF123456" not in d.scrub_text("x sk-live-ABCDEF123456 y")  # a KEY suffix alone is enough
    assert "gsk_ZZZZZZZZZZZZ" not in d.redact("gsk_ZZZZZZZZZZZZ")
    path = tmp_path / "glide.toml"
    path.write_text('[providers.odd]\nkind = "openai_compat"\nbase_url = "https://x.example/v1"\napi_key_env = "ODDNAME"\n')
    load_config(path)
    assert d.scrub_text("v plainvalue-9988 v") == f"v {d.REDACTED} v"

"""A credential written as a header in glide.toml is refused when the file loads (keys come from the environment only)."""

from __future__ import annotations

import pytest

from glide.providers.config import ConfigError, GlideConfig

SECRET = "sk-written-in-the-file-0123456789"


def load(text: str) -> GlideConfig:
    return GlideConfig.from_toml(text, env={})


@pytest.mark.parametrize(
    "name", ["Authorization", "authorization", "X-Api-Key", "api-key", "x-goog-api-key", "X-Auth-Token", "Cookie", "X-Secret"]
)
@pytest.mark.parametrize("key", ["extra_headers", "headers"])
def test_a_credential_header_in_a_providers_options_is_refused_without_echoing_it(name, key):
    with pytest.raises(ConfigError) as caught:
        load(
            f'[providers.p]\nkind = "openai_compat"\nbase_url = "https://p.example/v1"\n[providers.p.options]\n{key} = {{ "{name}" = "{SECRET}" }}\n'
        )
    message = str(caught.value)
    assert SECRET not in message and name in message and "api_key_env" in message and "[providers.p]" in message


def test_a_credential_header_in_a_slots_options_is_refused_too():
    text = f"""
[providers.p]
kind = "openai_compat"
base_url = "https://p.example/v1"

[llm.fast]
chain = [{{ provider = "p", model = "m", options = {{ extra_headers = {{ Authorization = "Bearer {SECRET}" }} }} }}]
"""
    with pytest.raises(ConfigError) as caught:
        load(text)
    assert SECRET not in str(caught.value) and "[llm.fast]" in str(caught.value)


def test_headers_that_are_not_credentials_are_kept():
    config = load(
        '[providers.p]\nkind = "openai_compat"\nbase_url = "https://p.example/v1"\n'
        '[providers.p.options]\nextra_headers = { "HTTP-Referer" = "https://example.org", "X-Title" = "Glide" }\n'
    )
    assert config.providers["p"].options["extra_headers"]["X-Title"] == "Glide"

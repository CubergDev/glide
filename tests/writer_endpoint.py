"""A writer over the provider chains, aimed at a local test endpoint. No network beyond 127.0.0.1."""

from __future__ import annotations

from glide.computer.writer import make_writer
from glide.providers.config import GlideConfig


def config_for(url: str, *, env: dict | None = None, key_env: str = "", extra: str = "") -> GlideConfig:
    """A configuration whose fast and smart chains are one `openai_compat` provider at `url`.

    The endpoint is configuration, as in a real glide.toml: nothing about it is in the code under test.
    """
    key = f'api_key_env = "{key_env}"' if key_env else ""
    toml = f"""
    [providers.local]
    kind = "openai_compat"
    base_url = "{url}/v1"
    {key}

    [llm.fast]
    chain = ["local:fast-model"]

    [llm.smart]
    chain = ["local:smart-model"]
    {extra}
    """
    return GlideConfig.from_toml(toml, env=env or {})


def writer_for(url: str, **kw):
    return make_writer(config_for(url, **kw))

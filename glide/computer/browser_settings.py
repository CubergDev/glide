"""Which browser provider drives a task, and how to reach it. Resolving never connects to or launches anything.

In glide.toml:

    [browser]
    provider = "cdp"            # native (the default), cdp, obscura or playwright
    target = "new"              # a tab id, or "new" for a tab of its own; empty means the one open tab
    fallback = ["obscura"]      # optional, default none: tried in order when the provider is unavailable, see providers.py
    search_url = "https://..."  # optional: where a search starts when a task names no site. There is no built-in one.

    [browser.cdp]
    endpoint = "http://127.0.0.1:9222"
    [browser.obscura]
    endpoint = "http://127.0.0.1:9333"
    [browser.playwright]
    session = "glide"           # an already opened Playwright CLI session
    cli = "/path/to/playwright-cli"

The environment wins over the file: GLIDE_BROWSER_PROVIDER, GLIDE_BROWSER_ENDPOINT (for the selected cdp or obscura
provider), GLIDE_BROWSER_TARGET, GLIDE_PLAYWRIGHT_SESSION, GLIDE_PLAYWRIGHT_CLI and GLIDE_SEARCH_URL. With no provider named, one with an
endpoint means cdp and anything else means the native desktop. Endpoints must be loopback HTTP origins: a browser is
reached only on this machine and only where the user pointed Glide at it. Nothing here holds a key.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from .execution.contracts import safe_url

PROVIDERS = ("native", "cdp", "obscura", "playwright")
ENDPOINT_PROVIDERS = ("cdp", "obscura")
DEFAULT_SESSION = "glide"
_TABLE_KEYS = ("provider", "target", "fallback", "search_url", *ENDPOINT_PROVIDERS, "playwright")
_SESSION = re.compile(r"[A-Za-z0-9_-]{1,64}")


@dataclass(frozen=True)
class Connection:
    provider: str
    endpoint: str
    target: str
    session: str


@dataclass(frozen=True)
class Settings:
    provider: str
    fallback: tuple[str, ...]
    target: str
    endpoints: Mapping[str, str]
    session: str
    cli: str
    search_url: str  # where a search starts when a task names no site; "" when none is configured
    source: str  # where the provider came from: "environment", "glide.toml" or "default"

    def connection(self, provider: str | None = None) -> Connection:
        provider = provider or self.provider
        return Connection(provider, self.endpoints.get(provider, ""), self.target, self.session)

    @property
    def chain(self) -> tuple[str, ...]:
        return (self.provider, *self.fallback)


def provider_name(value: str) -> str:
    return "playwright" if value == "playwright-cli" else value


def loopback_origin(endpoint: str) -> tuple[str, int]:
    """The normalized origin and port of a loopback HTTP endpoint, or ValueError. The message never quotes the value."""
    try:
        parsed = urlsplit(endpoint)
        port = parsed.port
    except ValueError:
        raise ValueError("Use an approved loopback HTTP browser endpoint") from None
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost"}
        or not port
        or parsed.username
        or parsed.password
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Use an approved loopback HTTP browser endpoint")
    return f"http://127.0.0.1:{port}", port


def validate_provider(provider: str, endpoint: str, target: str, session: str = DEFAULT_SESSION) -> None:
    provider = provider_name(provider)
    if provider not in PROVIDERS:
        raise ValueError("Choose native, cdp, playwright or obscura as the browser provider")
    if len(target) > 256 or any(c.isspace() or ord(c) < 32 for c in target):
        raise ValueError("Use an explicit browser target ID or new")
    if endpoint:
        loopback_origin(endpoint)
    elif provider in ENDPOINT_PROVIDERS:
        raise ValueError(f"The {provider} provider needs an approved loopback HTTP endpoint")
    if not _SESSION.fullmatch(session):
        raise ValueError("Playwright session names must use 1-64 letters, digits, underscores or hyphens")


def _text(table: Mapping[str, Any], key: str, where: str) -> str:
    value = table.get(key, "")
    if not isinstance(value, str):
        raise ValueError(f"{where} {key} must be text")
    return value.strip()


def _only(table: Mapping[str, Any], allowed: tuple[str, ...], where: str) -> None:
    for key in table:
        if key not in allowed:
            raise ValueError(f"{where} has an unknown key {key!r} (known: {', '.join(allowed)})")


def _subtable(table: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    sub = table.get(name, {})
    if not isinstance(sub, Mapping):
        raise ValueError(f"[browser.{name}] must be a table")
    _only(sub, ("session", "cli") if name == "playwright" else ("endpoint",), f"[browser.{name}]")
    return sub


def resolve(table: Mapping[str, Any] | None = None, env: Mapping[str, str] | None = None) -> Settings:
    """The settings for one run: `env` over the `[browser]` table over the defaults. ValueError when they cannot work."""
    env = os.environ if env is None else env
    table = {} if table is None else table
    _only(table, _TABLE_KEYS, "[browser]")
    subtables = {name: _subtable(table, name) for name in (*ENDPOINT_PROVIDERS, "playwright")}
    endpoints = {name: _text(subtables[name], "endpoint", f"[browser.{name}]") for name in ENDPOINT_PROVIDERS}
    override = env.get("GLIDE_BROWSER_ENDPOINT", "").strip()

    named_by_env = provider_name(env.get("GLIDE_BROWSER_PROVIDER", "").strip())
    named_by_file = provider_name(_text(table, "provider", "[browser]"))
    provider = named_by_env or named_by_file or ("cdp" if override or endpoints["cdp"] else "native")
    source = "environment" if named_by_env else "glide.toml" if named_by_file else "default"
    if override and provider in endpoints:
        endpoints[provider] = override  # an endpoint left over from another provider constrains nothing

    fallback = table.get("fallback", [])
    if not isinstance(fallback, list) or not all(isinstance(item, str) for item in fallback):
        raise ValueError("[browser] fallback must be a list of provider names")
    settings = Settings(
        provider=provider,
        fallback=tuple(provider_name(item.strip()) for item in fallback),
        target=env.get("GLIDE_BROWSER_TARGET", "") or _text(table, "target", "[browser]"),
        endpoints=endpoints,
        session=env.get("GLIDE_PLAYWRIGHT_SESSION", "")
        or _text(subtables["playwright"], "session", "[browser.playwright]")
        or DEFAULT_SESSION,
        cli=env.get("GLIDE_PLAYWRIGHT_CLI", "") or _text(subtables["playwright"], "cli", "[browser.playwright]"),
        search_url=env.get("GLIDE_SEARCH_URL", "").strip() or _text(table, "search_url", "[browser]"),
        source=source,
    )
    _check(settings)
    return settings


def _check(settings: Settings) -> None:
    if settings.search_url and not safe_url(settings.search_url):
        raise ValueError("The search address (search_url or GLIDE_SEARCH_URL) must be an http or https address")
    chain = settings.chain
    for name in chain:
        validate_provider(name, settings.endpoints.get(name, ""), settings.target, settings.session)
    if settings.fallback:
        if settings.provider == "native":
            raise ValueError("The native desktop has no fallback; fallback applies to cdp, obscura and playwright")
        if "native" in settings.fallback:
            raise ValueError("Native desktop control is never a fallback; select it explicitly as the provider")
        if len(set(chain)) != len(chain):
            raise ValueError("The browser fallback list repeats a provider")


# The `[browser]` table of the loaded glide.toml, installed once at start-up. The environment still overrides it.
_table: Mapping[str, Any] = {}


def use(table: Mapping[str, Any] | None) -> Settings:
    """Install the `[browser]` table of glide.toml for this process and return what it resolves to (ValueError if bad)."""
    global _table
    settings = resolve(table)
    _table = {} if table is None else dict(table)
    return settings


def current() -> Settings:
    return resolve(_table)


def description() -> str:
    settings = current()
    target = settings.target or "single tab"
    if settings.provider == "native":
        from .config import browser

        return f"Native desktop · {browser()}"
    if settings.provider == "playwright":
        return f"Playwright CLI · session {settings.session} · target {target}"
    return f"{'CDP' if settings.provider == 'cdp' else 'Obscura'} · {settings.endpoints[settings.provider]} · target {target}"


def add_arguments(parser) -> None:
    parser.add_argument("--browser-provider", choices=sorted((*PROVIDERS, "playwright-cli")), help="browser execution provider")
    parser.add_argument("--browser-endpoint", help="loopback HTTP endpoint of the selected cdp or obscura provider")
    parser.add_argument("--browser-target", help="selected target ID or new (creates a tab, not a browser)")
    parser.add_argument("--playwright-session", help="name of an already opened Playwright CLI session")


def apply_arguments(args) -> None:
    """Command-line choices become the environment for this process, then are checked together with the file."""
    for name, key in (
        ("browser_provider", "GLIDE_BROWSER_PROVIDER"),
        ("browser_endpoint", "GLIDE_BROWSER_ENDPOINT"),
        ("browser_target", "GLIDE_BROWSER_TARGET"),
        ("playwright_session", "GLIDE_PLAYWRIGHT_SESSION"),
    ):
        value = getattr(args, name, None)
        if value is not None:
            os.environ[key] = provider_name(value) if name == "browser_provider" else value
    current()

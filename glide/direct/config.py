"""The addresses and aliases `glide.direct` builds requests from: the shipped `defaults.toml`, plus an optional
override that the caller hands in (the `[direct]` table of glide.toml). This module reads no environment variable, opens
no connection, and the existing config loader is not touched: wiring passes `tomllib.load(f).get("direct")` to
`load_settings`, or a path to `load_settings_file`."""

from __future__ import annotations

import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from string import Formatter
from typing import Any
from urllib.parse import urlsplit

MAX_TEMPLATE = 2048
MAP_TEMPLATES = {
    "directions": {"origin", "destination"},
    "directions_to": {"destination"},
    "place": {"query"},
}
_TICKER = re.compile(r"[A-Z0-9]{1,6}(?:[.-][A-Z0-9]{1,3})?")
_KEYS = {"sites", "search", "maps", "tickers", "weather", "quote"}


class DirectConfigError(ValueError):
    """A bad `[direct]` table. The message names the key and never echoes the value."""


@dataclass(frozen=True)
class Settings:
    sites: dict[str, str]  # spoken name -> address
    search: dict[str, str]  # spoken name -> template with {query}
    default_search: str
    maps: dict[str, str]  # MAP_TEMPLATES name -> template
    weather: str  # template with {city}
    quote: str  # template with {ticker}
    tickers: dict[str, str]  # spoken company name -> ticker symbol


def _name(key: object, where: str) -> str:
    if not isinstance(key, str) or not re.fullmatch(r"[a-z0-9]+(?: [a-z0-9]+)*", key.casefold().strip()):
        raise DirectConfigError(f"{where}: a name must be words and digits")
    return key.casefold().strip()


def _http(url: object, where: str) -> str:
    """An absolute http(s) address with a host and no credentials; the host never holds a placeholder."""
    if not isinstance(url, str) or not 0 < len(url) <= MAX_TEMPLATE or any(c.isspace() or ord(c) < 32 for c in url):
        raise DirectConfigError(f"{where}: not a usable address")
    try:
        parts = urlsplit(url)
        bad_port = parts.port is not None and not 0 < parts.port <= 65535
    except ValueError:
        raise DirectConfigError(f"{where}: not a usable address") from None
    if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.password or bad_port:
        raise DirectConfigError(f"{where}: only http and https addresses are accepted")
    if "{" in parts.netloc or "}" in parts.netloc:
        raise DirectConfigError(f"{where}: a placeholder may not be in the host")
    return url


def _template(url: object, fields: set[str], where: str) -> str:
    text = _http(url, where)
    try:
        found = [name for _, name, spec, conv in Formatter().parse(text) if name is not None and not spec and not conv]
        total = sum(1 for _, name, _, _ in Formatter().parse(text) if name is not None)
    except ValueError:
        raise DirectConfigError(f"{where}: unbalanced braces") from None
    if set(found) != fields or len(found) != total:
        raise DirectConfigError(f"{where}: must use exactly {', '.join('{' + f + '}' for f in sorted(fields))}")
    return text


def _table(value: object, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise DirectConfigError(f"{where}: must be a table")
    return value


def _merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    unknown = set(override) - _KEYS
    if unknown:
        raise DirectConfigError("unknown key in [direct]: " + ", ".join(sorted(map(str, unknown))))
    merged = dict(base)
    for table in ("sites", "search", "maps", "tickers"):
        if table in override:
            merged[table] = {**base[table], **{_name(k, table): v for k, v in _table(override[table], table).items()}}
    for scalar in ("weather", "quote"):
        if scalar in override:
            merged[scalar] = override[scalar]
    return merged


def _build(data: Mapping[str, Any]) -> Settings:
    sites = {_name(k, "sites"): _http(v, f"sites.{k}") for k, v in data["sites"].items()}
    search = dict(data["search"])
    default = search.pop("default", None)
    templates = {_name(k, "search"): _template(v, {"query"}, f"search.{k}") for k, v in search.items()}
    if not isinstance(default, str) or default.casefold() not in templates:
        raise DirectConfigError("search.default: must name a search entry")
    maps = {}
    for key, value in data["maps"].items():
        if key not in MAP_TEMPLATES:
            raise DirectConfigError(f"maps: unknown entry {key!r}; expected one of {', '.join(MAP_TEMPLATES)}")
        maps[key] = _template(value, MAP_TEMPLATES[key], f"maps.{key}")
    if set(maps) != set(MAP_TEMPLATES):
        raise DirectConfigError("maps: directions, directions_to and place are all required")
    tickers = {}
    for key, value in data["tickers"].items():
        if not isinstance(value, str) or not _TICKER.fullmatch(value.upper()):
            raise DirectConfigError(f"tickers.{key}: not a ticker symbol")
        tickers[_name(key, "tickers")] = value.upper()
    return Settings(
        sites=sites,
        search=templates,
        default_search=default.casefold(),
        maps=maps,
        weather=_template(data["weather"], {"city"}, "weather"),
        quote=_template(data["quote"], {"ticker"}, "quote"),
        tickers=tickers,
    )


def _defaults() -> dict[str, Any]:
    return tomllib.loads(resources.files("glide.direct").joinpath("defaults.toml").read_text(encoding="utf-8"))


def load_settings(direct: Mapping[str, Any] | None = None) -> Settings:
    """The shipped defaults with `direct` (the `[direct]` table of glide.toml, or None) laid over them."""
    if direct is None:
        return _build(_defaults())
    return _build(_merge(_defaults(), _table(direct, "[direct]")))


def load_settings_file(path: str | Path) -> Settings:
    """`load_settings` for a glide.toml path. A missing file gives the defaults; a bad one raises `DirectConfigError`."""
    try:
        data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return load_settings()
    except (tomllib.TOMLDecodeError, UnicodeDecodeError, OSError) as error:
        raise DirectConfigError(f"cannot read {Path(path).name}: {type(error).__name__}") from None
    return load_settings(data.get("direct"))

"""Turn a clear request into a URL, with no network and no browser: open a site, search a site, directions, a place,
the weather, a stock quote. Every address comes from `Settings` (defaults.toml and the `[direct]` override), never from
this file. The words of the request are data: they are matched, then substituted URL-encoded into one placeholder that
the configuration guarantees is after the host, so a query can neither pick the destination nor add a parameter."""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import quote, urlsplit

from .config import Settings

MAX_TEXT = 300
MAX_PART = 200  # one query, city or place

_POLITE = re.compile(r"^(?:(?:hey |ok )?glide[, ]+)?(?:(?:please|can you|could you|would you)\s+)*", re.IGNORECASE)
_TRAILING = re.compile(r"[\s.,!?]*(?:\s+please)?[\s.,!?]*$", re.IGNORECASE)
_OPEN = re.compile(
    r"(?:open|go to|visit|take me to|pull up)\s+(?:the\s+)?(?P<what>.+?)(?:\s+(?:website|site|page))?", re.IGNORECASE
)
_SEARCH_SITE = re.compile(r"(?:search|find|look up|look)\s+(?P<site>[\w ]+?)\s+for\s+(?P<q>.+)", re.IGNORECASE)
_SEARCH_ON = re.compile(r"(?:search|look up|find)\s+(?:for\s+)?(?P<q>.+?)\s+(?:on|in|at)\s+(?P<site>[\w ]+)", re.IGNORECASE)
_SEARCH_FOR = re.compile(r"(?:search|look up)\s+for\s+(?P<q>.+)", re.IGNORECASE)
_SEARCH_ANY = re.compile(r"search\s+(?!for\b)(?P<q>.+)", re.IGNORECASE)
_DIRECTIONS = re.compile(
    r"(?:(?:get |give me |show me )?directions|navigate|route)(?:\s+from\s+(?P<from_>.+?))?\s+to\s+(?P<to_>.+?)", re.IGNORECASE
)
_DIRECTIONS_FROM = re.compile(r"(?:get |give me |show me )?directions\s+from\s+(?P<o>.+?)\s+to\s+(?P<d>.+)", re.IGNORECASE)
_HOW_TO_GET = re.compile(r"how (?:do i|can i|to) get to\s+(?P<d>.+?)\s+from\s+(?P<o>.+)", re.IGNORECASE)
_PLACE_ON_MAP = re.compile(
    r"(?:show|find|locate)\s+(?:me\s+)?(?:the\s+)?(?P<q>.+?)\s+on\s+(?:the\s+|google\s+)?maps?", re.IGNORECASE
)
_PLACE_MAP_OF = re.compile(r"(?:a\s+)?map of\s+(?:the\s+)?(?P<q>.+)", re.IGNORECASE)
_WEATHER = re.compile(
    r"(?:what(?:'s| is)|how(?:'s| is)|show me|tell me)?\s*(?:the\s+)?weather\s+(?:in|for|at)\s+(?P<city>.+)", re.IGNORECASE
)
_QUOTE = (
    re.compile(
        r"(?:what(?:'s| is)\s+)?(?:the\s+)?(?:stock\s+)?(?:price|quote)\s+(?:of|for)\s+(?:the\s+)?(?:stock\s+)?(?P<t>[\w.-]+)",
        re.IGNORECASE,
    ),
    re.compile(r"(?:stock\s+)?quote\s+(?:of|for)\s+(?P<t>[\w.-]+)", re.IGNORECASE),
    re.compile(r"(?P<t>[\w.-]+)\s+stock(?:\s+price|\s+quote)?", re.IGNORECASE),
)
_CITY = re.compile(r"(?=.*[^\W\d_])[^\W\d_][^\W\d_ .,'-]*(?:[ .,'-]+[^\W\d_ .,'-]+)*[.]?")
_TICKER = re.compile(r"[A-Za-z]{1,5}(?:[.-][A-Za-z]{1,2})?")
_HOST = re.compile(r"(?:www\.)[a-z0-9-]+(?:\.[a-z0-9-]+)+(?:[/?#]\S*)?", re.IGNORECASE)

# What the plan step says for each kind of request. It never repeats the user's words.
GOALS = {
    "open": "Open the requested site",
    "search": "Open the search results the user asked for",
    "directions": "Open the requested directions",
    "place": "Show the requested place on a map",
    "weather": "Open the requested weather page",
    "quote": "Open the requested stock quote",
}


@dataclass(frozen=True)
class Compiled:
    kind: str
    url: str


def _clean(value: str | None) -> str | None:
    """User text as a single bounded line, or None. Control characters are refused, not stripped."""
    if value is None:
        return None
    value = value.strip().rstrip(".,;:!?")
    if not value or len(value) > MAX_PART or any(ord(c) < 32 or ord(c) == 127 for c in value):
        return None
    return value


def _fill(template: str, **values: str) -> str:
    return template.format(**{k: quote(v, safe="") for k, v in values.items()})


def _safe(kind: str, url: str) -> Compiled | None:
    try:
        parts = urlsplit(url)
        ok = parts.scheme in {"http", "https"} and parts.hostname and not parts.username and not parts.password
    except ValueError:
        return None
    return Compiled(kind, url) if ok and not any(c.isspace() or ord(c) < 32 for c in url) else None


def _open(text: str, settings: Settings) -> Compiled | None:
    match = _OPEN.fullmatch(text)
    if not match:
        return None
    what = match["what"].strip()
    url = settings.sites.get(what.casefold())
    if url is None and re.fullmatch(r"https?://\S+", what, re.IGNORECASE):
        url = what
    elif url is None and _HOST.fullmatch(what):
        url = "https://" + what
    return _safe("open", url) if url else None


def _search(text: str, settings: Settings) -> Compiled | None:
    lowered = text.casefold()
    if not lowered.startswith(("search", "look", "find")):
        return None
    for pattern in (_SEARCH_SITE, _SEARCH_ON):
        match = pattern.fullmatch(text)
        if match:
            site = match["site"].strip().casefold()
            site = site.removeprefix("the ").removesuffix(" website").removesuffix(" site")
            template, query = settings.search.get(site), _clean(match["q"])
            if template and query:
                return _safe("search", _fill(template, query=query))
    # No known site named: the default engine, only for "search for X" or a bare "search X" with no site-like "for".
    match = _SEARCH_FOR.fullmatch(text) or (None if " for " in lowered else _SEARCH_ANY.fullmatch(text))
    query = _clean(match["q"]) if match else None
    return _safe("search", _fill(settings.search[settings.default_search], query=query)) if query else None


def _directions(text: str, settings: Settings) -> Compiled | None:
    origin = destination = None
    for pattern in (_DIRECTIONS_FROM, _HOW_TO_GET, _DIRECTIONS):
        match = pattern.fullmatch(text)
        if match:
            groups = match.groupdict()
            origin = groups.get("o") or groups.get("from_")
            destination = groups.get("d") or groups.get("to_")
            break
    origin, destination = _clean(origin) if origin else None, _clean(destination)
    if not destination:
        return None
    if origin:
        return _safe("directions", _fill(settings.maps["directions"], origin=origin, destination=destination))
    return _safe("directions", _fill(settings.maps["directions_to"], destination=destination))


def _place(text: str, settings: Settings) -> Compiled | None:
    match = _PLACE_ON_MAP.fullmatch(text) or _PLACE_MAP_OF.fullmatch(text)
    query = _clean(match["q"]) if match else None
    return _safe("place", _fill(settings.maps["place"], query=query)) if query else None


def _weather(text: str, settings: Settings) -> Compiled | None:
    match = _WEATHER.fullmatch(text)
    city = _clean(match["city"]) if match else None
    if not city or not _CITY.fullmatch(city):
        return None
    return _safe("weather", _fill(settings.weather, city=city))


def _quote(text: str, settings: Settings) -> Compiled | None:
    for pattern in _QUOTE:
        match = pattern.fullmatch(text)
        if match:
            word = match["t"].strip(".-")
            if not _TICKER.fullmatch(word):
                return None
            # A lowercase word is a company name, so it must be a configured one; capitals are a ticker as typed.
            symbol = settings.tickers.get(word.casefold()) if not word.isupper() else word
            return _safe("quote", _fill(settings.quote, ticker=symbol)) if symbol else None
    return None


def compile_request(text: str, settings: Settings) -> Compiled | None:
    """The URL for a clear request, or None. Never raises on odd input and reaches nothing outside this process."""
    if not isinstance(text, str) or not 0 < len(text) <= MAX_TEXT:
        return None
    text = _POLITE.sub("", text.strip())
    text = _TRAILING.sub("", text)
    for step in (_weather, _quote, _directions, _place, _search, _open):
        found = step(text, settings)
        if found:
            return found
    return None

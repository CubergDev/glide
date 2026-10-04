"""Tunables, the site catalog, and environment loading."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

MIN_OCR_CONFIDENCE = 0.3
MAX_OPTIONS = 255  # TypeSafe Choice ceiling
ABORT_CORNER_PX = 4
DEFAULT_MIN_CONFIDENCE = 0.4
DEFAULT_STEPS = 100
DEFAULT_DELAY = 2.0
DEFAULT_READINESS_TIMEOUT = 10.0  # seconds a page or form may take to become usable before the run reports it
DEFAULT_HANDOFFS = 10  # each one is a call to the answer model, a few seconds and a few cents
DEFAULT_RESEARCH_CALLS = 24  # model calls one research task may use, each a few seconds and a few cents
MAX_RESEARCH_CALLS = 32  # the most any setting may allow
DEFAULT_BROWSER = "Google Chrome"
ENGINES = ("legacy", "structured")
DEFAULT_ENGINE = "legacy"  # the structured engine is not live-qualified yet; flipping this is the user's call after live checks
ENGINE_ENV = "GLIDE_ENGINE"

# Sites the classifier can pick by name. Anything else goes through the writer.
SITES: dict[str, str] = {
    "github": "https://github.com/",
    "gmail": "https://mail.google.com/",
    "google_calendar": "https://calendar.google.com/",
    "launchdarkly": "https://app.launchdarkly.com/",
    "linear": "https://linear.app/",
    "notion": "https://www.notion.so/",
    "slack": "https://app.slack.com/",
    "typesafe_console": "https://console.typesafe.ai/",
}


def _chooses_configuration(name: str) -> bool:
    return name == "GLIDE_CONFIG" or name.startswith("GLIDE_PIN_")


def load_dotenv(path: Path) -> None:
    """Set KEY=VALUE lines from a .env file into the environment unless already set.

    A .env sits in the project directory, so it is not the user's say about WHICH configuration is used: it cannot set
    GLIDE_CONFIG (a file named there would count as the user's own and skip the trusted-hosts rule of a project-local
    glide.toml) or GLIDE_PIN_* (which slot gets a key). Those come from the real environment or a flag.
    """
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if _chooses_configuration(key.strip()):
            continue
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def browser() -> str:
    return os.environ.get("GLIDE_BROWSER", DEFAULT_BROWSER)


def writer_vision() -> bool:
    """Whether the answer model gets the screenshot. Off for a model that reads text only."""
    raw = os.environ.get("GLIDE_WRITER_VISION", "").strip().lower() or "true"
    if raw not in ("true", "false", "1", "0", "yes", "no"):
        raise ValueError(f"GLIDE_WRITER_VISION must be true or false, not {raw!r}")
    return raw in ("true", "1", "yes")


def email() -> str | None:
    return os.environ.get("GLIDE_EMAIL") or None


def research_budget(table: Mapping[str, Any] | None = None, env: Mapping[str, str] | None = None) -> int:
    """Model calls one research task may use: GLIDE_RESEARCH_CALLS over `calls` in the `[research]` table over the
    default. A ValueError names the setting, never its value."""
    env = os.environ if env is None else env
    table = {} if table is None else table
    for key in table:
        if key != "calls":
            raise ValueError(f"[research] has an unknown key {key!r} (known: calls)")
    named = env.get("GLIDE_RESEARCH_CALLS", "").strip()
    calls = table.get("calls", DEFAULT_RESEARCH_CALLS)
    if named:
        calls = int(named) if named.isdecimal() else None
    if type(calls) is not int or not 1 <= calls <= MAX_RESEARCH_CALLS:
        raise ValueError(
            f"The research budget ([research] calls or GLIDE_RESEARCH_CALLS) must be a whole number from 1 to {MAX_RESEARCH_CALLS}"
        )
    return calls


def engine(table: Mapping[str, Any] | None = None, env: Mapping[str, str] | None = None, flag: str | None = None) -> str:
    """The execution engine: `flag` (--engine, or the app's or pet's setting) over GLIDE_ENGINE over `engine` in the
    `[computer]` table over the default. The one resolver every front end uses. A ValueError names the setting, one line."""
    env = os.environ if env is None else env
    table = {} if table is None else table
    for key in table:
        if key != "engine":
            raise ValueError(f"[computer] has an unknown key {key!r} (known: engine)")
    chosen = flag if flag is not None else env.get(ENGINE_ENV, "").strip() or table.get("engine", DEFAULT_ENGINE)
    if chosen not in ENGINES:
        raise ValueError(f"The engine (--engine, {ENGINE_ENV} or [computer] engine) must be one of: {', '.join(ENGINES)}")
    return chosen

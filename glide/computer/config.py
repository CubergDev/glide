"""Tunables, the site catalog, and environment loading."""

from __future__ import annotations

import os
from pathlib import Path

MIN_OCR_CONFIDENCE = 0.3
MAX_OPTIONS = 255  # TypeSafe Choice ceiling
ABORT_CORNER_PX = 4
DEFAULT_MIN_CONFIDENCE = 0.4
DEFAULT_STEPS = 100
DEFAULT_DELAY = 2.0
DEFAULT_READINESS_TIMEOUT = 10.0  # seconds a page or form may take to become usable before the run reports it
DEFAULT_HANDOFFS = 10  # each one is a call to the answer model, a few seconds and a few cents
DEFAULT_BROWSER = "Google Chrome"

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


def load_dotenv(path: Path) -> None:
    """Set KEY=VALUE lines from a .env file into the environment unless already set."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
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

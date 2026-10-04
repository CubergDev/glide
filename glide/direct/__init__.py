"""Answers and destinations that need no model: arithmetic, and requests that are just a URL.

`resolve(text, settings)` returns a `Direct` for clear arithmetic ("what is 15% of 240": an answer to say) or for a
clear request to open a site, search a site, get directions, show a place, check the weather or a stock quote (a URL to
open, or the same as one plan step), and None for everything else, so the caller carries on with its normal route.
Importing this package, and calling `resolve`, reaches no network, no browser and no machine: it only builds strings.
Opening the URL is the caller's job, through the approved path. Addresses and aliases are configuration
(`defaults.toml`, overridden by the `[direct]` table of glide.toml passed to `load_settings`).

    python -m glide.direct "search amazon for headphones"
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .calc import evaluate
from .config import DirectConfigError, Settings, load_settings, load_settings_file
from .requests import GOALS, compile_request

__all__ = ["Direct", "DirectConfigError", "Settings", "load_settings", "load_settings_file", "resolve"]


@dataclass(frozen=True)
class Direct:
    kind: str  # "answer", or one of requests.GOALS
    answer: str = field(default="", repr=False)  # the result to say, for "answer"
    url: str = field(default="", repr=False)  # where to go, for the rest; it holds the user's words, so it is not logged

    def plan_step(self, step_id: str = "direct") -> dict[str, str | int] | None:
        """The same request as one `url` milestone in the shape the structured planner returns; None for an answer."""
        if not self.url:
            return None
        return {"id": step_id, "goal": GOALS[self.kind], "effect": "url", "target": "", "value": self.url, "quantity": 1}


def resolve(text: str, settings: Settings | None = None) -> Direct | None:
    """The direct answer or destination for `text`, or None when the request is not clear enough to be one."""
    answer = evaluate(text)
    if answer is not None:
        return Direct("answer", answer=answer)
    compiled = compile_request(text, settings or load_settings())
    return Direct(compiled.kind, url=compiled.url) if compiled else None

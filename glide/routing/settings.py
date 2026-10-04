"""The `[routing]` table of glide.toml, validated once. Nothing here names a vendor, a model or a voice.

```toml
[routing]
min_confidence = 0.6        # a non-acting route (answer, reason, clarify) needs at least this, after calibration
act_min_confidence = 0.8    # execute and research drive the machine, so they need more
act_min_margin = 0.2        # ...and the classifier must put this much more on them than on the runner-up
stop_min_confidence = 0.9   # a stop the MODEL heard (the phrase list is exact and needs no number)
escalate_to_reason = true   # sure it is not an action, unsure how deep: the frontier model, not a guess
history_turns = 4           # earlier turns the router may read (0-12)
history_chars = 300         # of each
memory_hints = false        # use hints the front end passes from memory; the front end passes none unless memory is on
speculative_fast = false    # ask the fast model at the same time as the classifier (one extra call, less waiting)
confirm_acting = false      # two keys to act: the fast model must read an execute/research request the same way
fast_timeout_s = 15.0       # the fast model's JSON call
max_clarifications = 1      # questions put to the user for one request
calibration_file = ""       # a reliability table from a run on your own providers (docs/ROUTER.md); "" = none yet
```

Every number above is a DEFAULT, named as one in this file, and none of them is a claim about a model: how often a
route is right at a given confidence is measured on the providers you configured (`calibration_file`), not assumed.
An unknown key is an error (a misspelt threshold would otherwise be ignored without a word).
"""

from __future__ import annotations

import math
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from glide.providers.config import ConfigError

DEFAULT_MIN_CONFIDENCE = 0.6
DEFAULT_ACT_MIN_CONFIDENCE = 0.8
DEFAULT_ACT_MIN_MARGIN = 0.2
DEFAULT_STOP_MIN_CONFIDENCE = 0.9
DEFAULT_HISTORY_TURNS = 4
DEFAULT_HISTORY_CHARS = 300
DEFAULT_FAST_TIMEOUT_S = 15.0
MAX_HISTORY_TURNS = 12
# A classifier that ranks its top 3 options cannot report less than about 1/3 for its top one (providers/classifier.py),
# so a floor at or under that would never fire.
MIN_FLOOR = 0.4


@dataclass(frozen=True)
class RoutingSettings:
    min_confidence: float = DEFAULT_MIN_CONFIDENCE
    act_min_confidence: float = DEFAULT_ACT_MIN_CONFIDENCE
    act_min_margin: float = DEFAULT_ACT_MIN_MARGIN
    stop_min_confidence: float = DEFAULT_STOP_MIN_CONFIDENCE
    escalate_to_reason: bool = True
    history_turns: int = DEFAULT_HISTORY_TURNS
    history_chars: int = DEFAULT_HISTORY_CHARS
    memory_hints: bool = False
    speculative_fast: bool = False
    confirm_acting: bool = False
    fast_timeout_s: float = DEFAULT_FAST_TIMEOUT_S
    max_clarifications: int = 1
    calibration_file: str = ""
    utterance_chars: int = 1000
    max_hints: int = 5
    hint_chars: int = 160

    def __post_init__(self) -> None:
        def number(name: str, low: float, high: float) -> None:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
                raise ConfigError(f"[routing] {name} must be a number")
            if not low <= value <= high:
                raise ConfigError(f"[routing] {name} must be from {low} to {high}")

        number("min_confidence", MIN_FLOOR, 1.0)
        number("act_min_confidence", MIN_FLOOR, 1.0)
        number("stop_min_confidence", MIN_FLOOR, 1.0)
        number("act_min_margin", 0.0, 1.0)
        number("fast_timeout_s", 0.5, 120.0)
        for name, high in (("history_turns", MAX_HISTORY_TURNS), ("max_clarifications", 3)):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value <= high:
                raise ConfigError(f"[routing] {name} must be a whole number from 0 to {high}")
        for name in ("history_chars", "utterance_chars", "max_hints", "hint_chars"):
            value = getattr(self, name)
            if type(value) is not int or not 0 <= value <= 20000:
                raise ConfigError(f"[routing] {name} must be a whole number from 0 to 20000")
        for name in ("escalate_to_reason", "memory_hints", "speculative_fast", "confirm_acting"):
            if type(getattr(self, name)) is not bool:
                raise ConfigError(f"[routing] {name} must be true or false")
        if not isinstance(self.calibration_file, str):
            raise ConfigError("[routing] calibration_file must be a path or empty")
        if self.act_min_confidence < self.min_confidence:
            raise ConfigError("[routing] act_min_confidence must not be below min_confidence: acting is never the easier call")

    @classmethod
    def from_table(cls, table: Mapping[str, Any] | None) -> RoutingSettings:
        """Settings from the `[routing]` table, as already parsed. Unknown keys are refused, naming the known ones."""
        table = dict(table or {})
        known = {f.name for f in fields(cls)}
        for key in table:
            if key not in known:
                raise ConfigError(f"[routing] has an unknown key {key!r} (known: {', '.join(sorted(known))})")
        return cls(**table)

    @classmethod
    def load(cls, path: str | Path | None) -> RoutingSettings:
        """Settings from a glide.toml path (defaults when there is no file or no `[routing]` table)."""
        if path is None:
            return cls()
        try:
            data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError:
            raise ConfigError(f"{path} is not valid TOML") from None
        except (OSError, UnicodeDecodeError):
            raise ConfigError(f"cannot read {path}") from None
        table = data.get("routing", {})
        if not isinstance(table, dict):
            raise ConfigError("[routing] must be a table")
        return cls.from_table(table)

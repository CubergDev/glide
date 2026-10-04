"""The `[panel]` table of glide.toml: switches only the control panel reads. Every one defaults to the safe side.

```toml
[panel]
computer = false       # the Chat tab may start computer tasks (each still needs a click, and a real run a second one)
files = false          # the Files tab may plan and run moves
voice = false          # the panel shows the voice command and its status (voice itself runs in a terminal)
point_ask = false      # the panel shows the point-and-ask status (it runs from the app or the pet)
record_content = false # detailed recording: the panel keeps a local chat log of utterances and replies
retention_days = 30    # a chat log entry older than this is deleted when the log is next written
```
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

SWITCHES = ("computer", "files", "voice", "point_ask", "record_content")
KEYS = (*SWITCHES, "retention_days")
MAX_RETENTION_DAYS = 365


@dataclass(frozen=True)
class PanelSettings:
    computer: bool = False
    files: bool = False
    voice: bool = False
    point_ask: bool = False
    record_content: bool = False
    retention_days: int = 30

    @classmethod
    def from_mapping(cls, table: Mapping | None = None) -> PanelSettings:
        table = {} if table is None else table
        for key in table:
            if key not in KEYS:
                raise ValueError(f"[panel] has an unknown key {key!r} (known: {', '.join(KEYS)})")
        out: dict = {}
        for key in SWITCHES:
            if key in table:
                if not isinstance(table[key], bool):
                    raise ValueError(f"[panel] {key} must be true or false")
                out[key] = table[key]
        if "retention_days" in table:
            days = table["retention_days"]
            if type(days) is not int or not 1 <= days <= MAX_RETENTION_DAYS:
                raise ValueError(f"[panel] retention_days must be a whole number from 1 to {MAX_RETENTION_DAYS}")
            out["retention_days"] = days
        return cls(**out)

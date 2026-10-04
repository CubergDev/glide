"""What a glide.toml that is not the user's own may not set.

A glide.toml found in the current directory can be shipped by a repository. Besides the key-host rule in
`providers.config`, it must never LOWER a safety property on the user's behalf: it may not turn the per-task yes of
hands-free speech off, opt the user into storing what they say, name a webhook service file, or add MCP clients or
memory sharing. The user's own file, `--config` and $GLIDE_CONFIG are the user's say and are honoured.

Every reader of a project-local file goes through `strip`, so the rule has one home. Nothing here reads a file.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

# table -> the keys a project-local file may not set (None: no key of that table, the table is ignored)
REFUSED: Mapping[str, tuple[str, ...] | None] = {
    "speech": ("confirm_tasks", "confirm_phrase", "confirm_timeout_s"),
    "memory": None,
    "webhooks": None,
    "mcp": None,
}


def strip_table(name: str, table: Mapping[str, Any]) -> tuple[dict[str, Any], tuple[str, ...]]:
    """`table` without the keys a project-local file may not set, and the names that were left out."""
    if name not in REFUSED:
        return dict(table), ()
    refused = REFUSED[name]
    if refused is None:
        return {}, tuple(f"[{name}] {key}" for key in table)
    return {k: v for k, v in table.items() if k not in refused}, tuple(f"[{name}] {k}" for k in table if k in refused)


def strip(data: Mapping[str, Any]) -> tuple[dict[str, Any], tuple[str, ...]]:
    """The parsed file without the settings a project-local file may not make, and the settings that were left out."""
    out: dict[str, Any] = {}
    left: tuple[str, ...] = ()
    for name, value in data.items():
        if name in REFUSED and isinstance(value, Mapping):
            value, dropped = strip_table(name, value)
            left += dropped
            if REFUSED[name] is None:
                continue
        out[name] = value
    return out, left


def notice(source: str, left: tuple[str, ...]) -> str:
    return (
        f"{source}: read from the current directory, so it cannot set {', '.join(left)}; "
        "put them in your own configuration (~/.config/glide/glide.toml) or pass --config"
    )

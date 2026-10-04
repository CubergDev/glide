"""MCP settings from the [mcp] table of glide.toml. Reading them starts nothing and connects to nothing.

    [mcp]
    server_memory = "off"        # "off" (default), "read" or "write": what `glide mcp serve` offers from memory.
                                 # Anything but "off" also needs [memory] enabled = true.

    [[mcp.servers]]              # MCP servers Glide may use as a client. None by default.
    name = "files"               # letters, digits, . _ - ; becomes the tool namespace mcp:files/...
    command = "some-server"      # a local command, started over stdio only after the user approves that command
    args = ["--flag"]
    env = ["SOME_TOKEN"]         # NAMES of environment variables to pass on. Values are never written anywhere.
    timeout_s = 60

Keys, tokens and URLs are never stored in this table: a server that needs a secret names the variable.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from glide.memory.settings import SettingsError, locate_config, read_table

_NAME = re.compile(r"[A-Za-z0-9_.-]{1,128}\Z")
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}\Z")
SERVER_MEMORY = ("off", "read", "write")
_KNOWN = ("server_memory", "servers")
_KNOWN_SERVER = ("name", "command", "args", "env", "timeout_s")


@dataclass(frozen=True)
class McpServerSpec:
    name: str
    command: str
    args: tuple[str, ...] = ()
    env_names: tuple[str, ...] = ()
    timeout_s: float = 60.0


def _server_spec(entry: Any, seen: set[str]) -> McpServerSpec:
    """One `[[mcp.servers]]` table, checked key by key. Messages name the setting, never a value read from disk."""
    if not isinstance(entry, dict):
        raise SettingsError("each [[mcp.servers]] entry must be a table")
    for key in entry:
        if key not in _KNOWN_SERVER:
            raise SettingsError(f"[[mcp.servers]] has an unknown key {key!r} (known: {', '.join(_KNOWN_SERVER)})")
    name, command = entry.get("name"), entry.get("command")
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise SettingsError("[[mcp.servers]] name must be letters, digits, . _ - (at most 128)")
    if name in seen:
        raise SettingsError(f"[[mcp.servers]] name {name!r} is listed twice")
    seen.add(name)
    if not isinstance(command, str) or not command.strip() or "\x00" in command:
        raise SettingsError(f"[[mcp.servers]] {name}: command must be a nonempty string")
    args, env = entry.get("args", []), entry.get("env", [])
    if not isinstance(args, list) or any(not isinstance(a, str) or "\x00" in a for a in args):
        raise SettingsError(f"[[mcp.servers]] {name}: args must be a list of strings")
    if not isinstance(env, list) or any(not isinstance(e, str) or not _ENV_NAME.fullmatch(e) for e in env):
        raise SettingsError(f"[[mcp.servers]] {name}: env must list environment variable NAMES, not values")
    timeout = entry.get("timeout_s", 60.0)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout <= 3600:
        raise SettingsError(f"[[mcp.servers]] {name}: timeout_s must be between 0 and 3600")
    return McpServerSpec(name, command, tuple(args), tuple(env), float(timeout))


@dataclass(frozen=True)
class McpSettings:
    server_memory: str = "off"
    servers: tuple[McpServerSpec, ...] = field(default_factory=tuple)

    @classmethod
    def from_mapping(cls, table: Mapping[str, Any] | None = None) -> McpSettings:
        table = {} if table is None else table
        for key in table:
            if key not in _KNOWN:
                raise SettingsError(f"[mcp] has an unknown key {key!r} (known: {', '.join(_KNOWN)})")
        mode = table.get("server_memory", "off")
        if mode not in SERVER_MEMORY:
            raise SettingsError(f"[mcp] server_memory must be one of {', '.join(SERVER_MEMORY)}")
        raw = table.get("servers", [])
        if not isinstance(raw, list):
            raise SettingsError("[mcp] servers must be an array of tables ([[mcp.servers]])")
        seen: set[str] = set()
        return cls(mode, tuple(_server_spec(entry, seen) for entry in raw))

    @classmethod
    def load(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        config: str | os.PathLike[str] | None = None,
        cwd: Path | None = None,
        home: Path | None = None,
    ) -> McpSettings:
        env = os.environ if environ is None else environ
        found, foreign = locate_config(env, path=config, cwd=cwd, home=home)
        return cls.from_mapping(read_table("mcp", found, foreign=foreign))

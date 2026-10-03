"""Opt-in switches and the private data directory for memory. Reading settings touches no database.

Memory persists text the user typed or spoke, so it is OFF unless the user turns it on:

    [memory]                # in glide.toml
    enabled = true          # default false. GLIDE_MEMORY=1 / 0 in the environment overrides the file.
    auto_capture = false    # default false. true also saves sentences such as "I prefer concise answers".
                            # GLIDE_MEMORY_AUTO=1 / 0 overrides the file.
    data_dir = "..."        # default: GLIDE_DATA_DIR, else the platform's per-user data directory.

With `enabled` false nothing is created, read or written: not the directory, not the database.
"""

from __future__ import annotations

import os
import sys
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CONFIG_NAME = "glide.toml"
_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off", ""})
_KNOWN = ("enabled", "auto_capture", "data_dir")


class SettingsError(ValueError):
    """A settings problem the user can fix. The message names the setting, never a value read from disk."""


def parse_switch(value: object, name: str, *, text: bool = True) -> bool:
    """A boolean from a TOML bool or (when `text`) an environment string; anything else is an error, not a guess."""
    if isinstance(value, bool):
        return value
    if text and isinstance(value, str):
        text = value.strip().lower()
        if text in _TRUE:
            return True
        if text in _FALSE:
            return False
    raise SettingsError(f"{name} must be true or false")


def default_data_dir(environ: Mapping[str, str], home: Path | None = None) -> Path:
    """GLIDE_DATA_DIR, else the platform's per-user data directory for glide."""
    named = (environ.get("GLIDE_DATA_DIR") or "").strip()
    if named:
        return Path(named).expanduser()
    base = home if home is not None else Path(environ.get("HOME") or Path.home())
    if sys.platform == "darwin":
        return base / "Library" / "Application Support" / "glide"
    if sys.platform.startswith("win"):
        local = environ.get("LOCALAPPDATA")
        return (Path(local) if local else base / "AppData" / "Local") / "glide"
    share = environ.get("XDG_DATA_HOME")
    return (Path(share) if share else base / ".local" / "share") / "glide"


def find_config(
    environ: Mapping[str, str], *, path: str | os.PathLike[str] | None = None, cwd: Path | None = None, home: Path | None = None
) -> Path | None:
    """The glide.toml `load_config` would pick: `path`, $GLIDE_CONFIG, ./glide.toml, ~/.config/glide/glide.toml."""
    if path is not None:
        chosen = Path(path).expanduser()
        if not chosen.is_file():
            raise SettingsError(f"the config file {chosen} does not exist")
        return chosen
    if named := (environ.get("GLIDE_CONFIG") or "").strip():
        chosen = Path(named).expanduser()
        if not chosen.is_file():
            raise SettingsError(f"GLIDE_CONFIG names {chosen}, which does not exist")
        return chosen
    base = home if home is not None else (Path(h) if (h := environ.get("HOME")) else None)
    for candidate in ((cwd or Path.cwd()) / CONFIG_NAME, base / ".config" / "glide" / CONFIG_NAME if base else None):
        if candidate is not None and candidate.is_file():
            return candidate
    return None


def read_table(name: str, config: Path | None) -> Mapping[str, Any]:
    """One top-level table of glide.toml ({} when there is no file or no such table)."""
    if config is None:
        return {}
    try:
        data = tomllib.loads(config.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError:
        raise SettingsError(f"{config} is not valid TOML") from None
    except (OSError, UnicodeDecodeError):
        raise SettingsError(f"cannot read {config}") from None
    table = data.get(name, {})
    if not isinstance(table, dict):
        raise SettingsError(f"[{name}] must be a table")
    return table


@dataclass(frozen=True)
class MemorySettings:
    enabled: bool = False
    auto_capture: bool = False
    data_dir: Path | None = None

    @classmethod
    def from_mapping(cls, table: Mapping[str, Any] | None = None, environ: Mapping[str, str] | None = None) -> MemorySettings:
        """Settings from the [memory] table, with GLIDE_MEMORY and GLIDE_MEMORY_AUTO over it. Unknown keys are errors."""
        table = {} if table is None else table
        env = os.environ if environ is None else environ
        for key in table:
            if key not in _KNOWN:
                raise SettingsError(f"[memory] has an unknown key {key!r} (known: {', '.join(_KNOWN)})")
        enabled = parse_switch(table.get("enabled", False), "[memory] enabled", text=False)
        auto = parse_switch(table.get("auto_capture", False), "[memory] auto_capture", text=False)
        if "GLIDE_MEMORY" in env:
            enabled = parse_switch(env["GLIDE_MEMORY"], "GLIDE_MEMORY")
        if "GLIDE_MEMORY_AUTO" in env:
            auto = parse_switch(env["GLIDE_MEMORY_AUTO"], "GLIDE_MEMORY_AUTO")
        data_dir = table.get("data_dir")
        if data_dir is not None and (not isinstance(data_dir, str) or not data_dir.strip()):
            raise SettingsError("[memory] data_dir must be a path")
        return cls(enabled, auto, Path(data_dir).expanduser() if data_dir else None)

    @classmethod
    def load(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        config: str | os.PathLike[str] | None = None,
        cwd: Path | None = None,
        home: Path | None = None,
    ) -> MemorySettings:
        env = os.environ if environ is None else environ
        return cls.from_mapping(read_table("memory", find_config(env, path=config, cwd=cwd, home=home)), env)

    def memory_dir(self, environ: Mapping[str, str] | None = None, home: Path | None = None) -> Path:
        """Where memory lives: <data dir>/memory. Computing it creates nothing."""
        env = os.environ if environ is None else environ
        return (self.data_dir or default_data_dir(env, home)) / "memory"

    def database_path(self, environ: Mapping[str, str] | None = None, home: Path | None = None) -> Path:
        return self.memory_dir(environ, home) / "memory.sqlite"

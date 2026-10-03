"""Open memory only after the user opted in, in a private data directory.

`MemoryService.open(settings)` returns None when memory is off and then does nothing at all: it creates no
directory and opens no database. When on, the data directory and its `memory` subdirectory are 0700 and the
database files are 0600 (see store.py). Everything else in this package is a library that never decides
by itself to persist anything.
"""

from __future__ import annotations

import os
import stat
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

from .catalog import Catalog
from .contracts import Model, Policy, Tool
from .harness import Harness
from .settings import MemorySettings, SettingsError
from .store import Store


def _private_dir(path: Path, *, tighten: bool) -> None:
    """Create `path` as 0700. An existing one must be a real directory this user owns; `tighten` fixes its mode."""
    if not path.exists() and not path.is_symlink():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.mkdir(mode=0o700)
        os.chmod(path, 0o700)  # mkdir's mode is masked by the umask
        return
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode):
        raise SettingsError(f"{path} must be a directory (not a symlink or file)")
    if hasattr(os, "geteuid") and info.st_uid != os.geteuid():
        raise SettingsError(f"{path} is owned by another user")
    if tighten:
        os.chmod(path, 0o700)


class MemoryService:
    """The opened store and harness. Close it (or use `with`) when the process is done."""

    def __init__(self, store: Store, harness: Harness, settings: MemorySettings, database: Path):
        self.store, self.harness, self.settings, self.database = store, harness, settings, database

    @classmethod
    def open(
        cls,
        settings: MemorySettings,
        *,
        environ: Mapping[str, str] | None = None,
        home: Path | None = None,
        catalog_root: Path | None = None,
        enabled_plugins: frozenset[str] = frozenset(),
        enabled_skills: frozenset[str] = frozenset(),
        tools: tuple[Tool, ...] = (),
        models: tuple[Model, ...] = (),
        policy: Policy | None = None,
    ) -> MemoryService | None:
        if not settings.enabled:
            return None
        base = settings.memory_dir(environ, home)
        data_dir = base.parent
        existed = data_dir.exists()
        _private_dir(data_dir, tighten=not existed)
        _private_dir(base, tighten=True)
        root = catalog_root
        if root is None:
            root = base / "catalog"
            _private_dir(root, tighten=True)
        policy = policy or Policy()
        policy = replace(policy, auto_memory=settings.auto_capture)  # the setting is the only switch
        store = Store(base / "memory.sqlite")
        try:
            catalog = Catalog(root, enabled_plugins=enabled_plugins, enabled_skills=enabled_skills)
            harness = Harness(store, catalog, tools=tools, models=models, policy=policy)
        except BaseException:
            store.close()
            raise
        return cls(store, harness, settings, base / "memory.sqlite")

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> MemoryService:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

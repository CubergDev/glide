"""A throwaway home folder for the file planner tests. Everything lives under the test's tmp_path: the strict guard
protects the machine, not the disk, and `os.rename`, `os.mkdir` and `os.rmdir` are not in it."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class World:
    home: Path
    root: Path  # <home>/Downloads
    runs: Path  # the manifest folder

    def make(self, *names: str) -> None:
        for name in names:
            (self.root / name).write_text(name)

    def listing(self) -> list[str]:
        """Every path under root, relative, folders with a trailing slash: the whole visible state."""
        found = []
        for base, dirs, files in os.walk(self.root):
            rel = Path(base).relative_to(self.root)
            found += [f"{(rel / d).as_posix()}/" for d in dirs if not (Path(base) / d).is_symlink()]
            found += [(rel / f).as_posix() for f in files]
        return sorted(p.removeprefix("./") for p in found)


def build_world(tmp_path: Path) -> World:
    home = (tmp_path / "home").resolve()
    root = home / "Downloads"
    root.mkdir(parents=True)
    runs = tmp_path / "runs"
    return World(home, root, runs)

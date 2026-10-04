"""The path rules every file move answers to: which folder may be touched, which names are acceptable, and how a
colliding name is made unique. Pure checks on strings and the disk's current state; nothing here moves anything."""

from __future__ import annotations

import os
from pathlib import Path


class Refused(Exception):
    """A plan, manifest or path that the file planner will not act on. Raised before anything is touched."""


def check_name(name: object, what: str = "name") -> str:
    """One path component: no separator, no NUL, no leading dot (so no `.` or `..` and nothing hidden), not empty."""
    if not isinstance(name, str) or not name:
        raise Refused(f"{what} must be a non-empty string")
    if "/" in name or "\\" in name or "\0" in name:
        raise Refused(f"{what} may not contain a slash or a NUL")
    if name.startswith("."):
        raise Refused(f"{what} may not start with a dot")
    return name


def real_home(home: Path | str | None) -> str:
    resolved = os.path.realpath(str(home) if home is not None else str(Path.home()))
    if resolved == os.path.dirname(resolved):
        raise Refused("the home folder cannot be the top of the file system")
    return resolved


def _off_limits(path: str, base: str) -> bool:
    """Whether `path` is below `base` through a hidden folder or a Library folder."""
    if not path.startswith(base + os.sep):
        return False
    return any(part.startswith(".") or part.casefold() == "library" for part in path[len(base) + 1 :].split(os.sep))


def check_root(root: Path | str, home: Path | str | None = None) -> str:
    """The real path of a folder Glide may reorganize: an existing folder strictly below the user's home, never the
    home itself, never through a hidden or Library folder (as written or once symlinks are resolved)."""
    given = os.path.abspath(str(root))
    base = real_home(home)
    real = os.path.realpath(given)
    if not os.path.isdir(real):
        raise Refused("the folder to organize does not exist")
    if not real.startswith(base + os.sep):
        raise Refused("the folder must be inside the home folder, and not the home folder itself")
    raw_home = os.path.abspath(str(home) if home is not None else str(Path.home()))
    if _off_limits(real, base) or _off_limits(given, raw_home) or _off_limits(given, base):
        raise Refused("hidden and Library folders are never touched")
    return real


def unique_name(folder: str, name: str, taken: set[str]) -> str:
    """`name`, or `stem (2).ext`, `stem (3).ext` ... the first that is not on disk in `folder` and not in `taken`
    (lower-cased names already chosen in this run). The result is added to `taken`."""
    stem, dot, ext = name.rpartition(".") if "." in name[1:] else (name, "", "")
    n, candidate = 1, name
    while candidate.casefold() in taken or os.path.lexists(os.path.join(folder, candidate)):
        n += 1
        candidate = f"{stem} ({n}){dot}{ext}"
    taken.add(candidate.casefold())
    return candidate

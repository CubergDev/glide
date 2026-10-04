"""plan() and preview(): decide what would move where, and describe it. Read-only: no file or folder is created."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path

from .model import Move, Plan
from .safety import Refused, check_name, check_root, unique_name

ORGANIZE_BY_TYPE = "organize by type"
NAMED_FOLDERS = "sort into named folders"
OTHER = "Other"

# Folder name -> extensions. Configuration: pass your own table to `plan`, or `--categories file.json` on the command line.
DEFAULT_CATEGORIES: Mapping[str, tuple[str, ...]] = {
    "Documents": (".pdf", ".doc", ".docx", ".txt", ".rtf", ".md", ".odt", ".pages", ".epub"),
    "Images": (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp", ".heic", ".svg"),
    "Spreadsheets": (".xls", ".xlsx", ".csv", ".ods", ".numbers"),
    "Presentations": (".ppt", ".pptx", ".odp", ".key"),
    "Audio": (".mp3", ".wav", ".aac", ".flac", ".m4a", ".ogg", ".aiff"),
    "Video": (".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"),
    "Archives": (".zip", ".tar", ".gz", ".tgz", ".bz2", ".7z", ".rar", ".dmg"),
    "Code": (".py", ".js", ".ts", ".java", ".c", ".h", ".cpp", ".go", ".rs", ".rb", ".sh", ".json", ".html", ".css", ".swift"),
}


def _extension(name: str) -> str:
    return os.path.splitext(name)[1].casefold()


def _by_type(name: str, table: Mapping[str, Sequence[str]]) -> str:
    ext = _extension(name)
    for folder, extensions in table.items():
        if ext and ext in (e.casefold() for e in extensions):
            return folder
    return OTHER


def _by_keywords(name: str, folders: Mapping[str, Sequence[str]]) -> str | None:
    stem = os.path.splitext(name)[0].casefold()
    for folder, keywords in folders.items():
        for word in keywords:
            if (word.startswith(".") and _extension(name) == word.casefold()) or (
                not word.startswith(".") and word.casefold() in stem
            ):
                return folder
    return None


def _checked_table(table: Mapping[str, Sequence[str]]) -> None:
    for folder, words in table.items():
        check_name(folder, "folder name")
        if not words or not all(isinstance(w, str) and w for w in words):
            raise Refused(f"folder {folder!r} needs at least one non-empty keyword or extension")


def plan(
    root: Path | str,
    intent: str,
    *,
    categories: Mapping[str, Sequence[str]] | None = None,
    folders: Mapping[str, Sequence[str]] | None = None,
    home: Path | str | None = None,
) -> Plan:
    """What `intent` would do to the direct, non-hidden, regular files in `root`. Moves nothing.

    `organize by type` sorts by extension into `categories` (default `DEFAULT_CATEGORIES`, anything else into Other).
    `sort into named folders` takes `folders` (folder -> keywords; a keyword starting with a dot is an extension, any
    other is matched inside the file name, first folder wins) and leaves files that match none where they are."""
    real = check_root(root, home)
    if intent == ORGANIZE_BY_TYPE:
        table = dict(DEFAULT_CATEGORIES if categories is None else categories)
        _checked_table(table)
        choose = lambda name: _by_type(name, table)  # noqa: E731
    elif intent == NAMED_FOLDERS:
        if not folders:
            raise Refused("sorting into named folders needs the folders and their keywords")
        _checked_table(folders)
        choose = lambda name: _by_keywords(name, folders)  # noqa: E731
    else:
        raise Refused(f"unknown intent {intent!r}")

    moves: list[Move] = []
    taken: dict[str, set[str]] = {}
    with os.scandir(real) as entries:
        files = sorted(
            (e for e in entries if not e.name.startswith(".") and e.is_file(follow_symlinks=False)), key=lambda e: e.name
        )
    for entry in files:
        folder = choose(entry.name)
        if folder is None:
            continue
        check_name(folder, "folder name")
        target = os.path.join(real, folder)
        name = unique_name(target, check_name(entry.name), taken.setdefault(folder, set()))
        info = entry.stat(follow_symlinks=False)
        moves.append(Move(entry.path, os.path.join(target, name), (info.st_ino, info.st_size, info.st_mtime_ns)))
    bare = Plan(real, intent, tuple(moves), "")
    return Plan(real, intent, bare.moves, bare.compute_hash())


def preview(plan: Plan, examples: int = 3) -> str:
    """A dry run in words: how many files go to each folder. Nothing is touched, and nothing is created."""
    if not plan.moves:
        return f"Plan {plan.plan_hash}: nothing to move in {plan.root}."
    per_folder: dict[str, list[str]] = {}
    for move in plan.moves:
        per_folder.setdefault(os.path.basename(os.path.dirname(move.destination)), []).append(os.path.basename(move.source))
    lines = [f"Plan {plan.plan_hash}: {_files(len(plan.moves))} in {plan.root}, '{plan.intent}'."]
    for folder, names in per_folder.items():
        shown = ", ".join(names[:examples]) + (", ..." if len(names) > examples else "")
        lines.append(f"  {folder}/  {_files(len(names))}  ({shown})")
    lines.append("Nothing has been moved. Only an approval carrying the hash above runs this plan.")
    return "\n".join(lines)


def _files(count: int) -> str:
    return f"{count} file" if count == 1 else f"{count} files"

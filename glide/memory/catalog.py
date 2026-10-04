"""Opt-in Markdown registry; frontmatter is strict JSON, not general YAML.

Only flat ``skills/*.md`` and ``plugins/*.md`` files are read. Manifests declare
metadata and instructions, never executable entrypoints. A reload publishes its
complete validated snapshot at once, preserving the old snapshot on failure.
"""

import json
import os
import stat
from pathlib import Path
from threading import RLock

STAGES = frozenset({"classify", "write", "answer", "handoff", "refine"})
MAX_MANIFEST_BYTES = 65536
MAX_BODY_BYTES = 32768


def _identifier(value: object, limit: int = 128) -> bool:
    return isinstance(value, str) and 0 < len(value) <= limit and not any(char.isspace() or ord(char) < 32 for char in value)


def _object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _parse(raw: bytes, kind: str, label: str) -> dict:
    if len(raw) > MAX_MANIFEST_BYTES:
        raise ValueError(f"manifest too large: {label}")
    try:
        lines = raw.decode("utf-8").splitlines(keepends=True)
        if not lines or lines[0].strip() != "---":
            raise ValueError("missing JSON frontmatter")
        end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
        item = json.loads("".join(lines[1:end]), object_pairs_hook=_object)
        body = "".join(lines[end + 1 :]).strip()
    except (UnicodeError, StopIteration, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON frontmatter: {label}") from error
    fields = (
        {"id", "description", "tools", "keywords", "stages"} if kind == "skills" else {"id", "description", "skills", "tools"}
    )
    if not isinstance(item, dict) or set(item) != fields:
        raise ValueError(f"required/unknown fields in {label}: expected {sorted(fields)}")
    if not _identifier(item["id"]):
        raise ValueError(f"invalid id: {label}")
    if not isinstance(item["description"], str) or not item["description"].strip():
        raise ValueError(f"invalid description: {label}")
    for field in sorted(fields - {"id", "description"}):
        value = item[field]
        if not isinstance(value, list):
            raise ValueError(f"{field} must be a list of nonempty strings: {label}")
        if field == "keywords":
            valid = all(isinstance(entry, str) and entry.strip() and len(entry) <= 256 for entry in value)
        else:
            valid = all(_identifier(entry, 512 if field == "tools" else 128) for entry in value)
        if not valid:
            raise ValueError(f"{field} must be a list of nonempty strings: {label}")
        if len(value) != len(set(value)):
            raise ValueError(f"duplicate {field}: {label}")
    if kind == "skills":
        if not item["stages"] or not set(item["stages"]) <= STAGES:
            raise ValueError(f"unknown or empty stages: {label}")
        if not body or len(body.encode("utf-8")) > MAX_BODY_BYTES:
            raise ValueError(f"empty or oversized instructions: {label}")
        item["body"] = body
    elif len(body.encode("utf-8")) > MAX_BODY_BYTES:
        raise ValueError(f"plugin body too large: {label}")
    return {key: tuple(value) if isinstance(value, list) else value for key, value in item.items()}


# Opening every step relative to a directory descriptor, refusing symlinks, needs POSIX. Elsewhere (Windows) the
# same checks are made on paths: no symlink or junction, regular files only. That path cannot close a swap between
# the check and the read, so the descriptor route is used wherever it exists.
_DIR_FD = hasattr(os, "O_NOFOLLOW") and hasattr(os, "O_DIRECTORY") and os.open in os.supports_dir_fd


def _add(records: dict[str, dict], raw: bytes, kind: str, name: str) -> None:
    item = _parse(raw, kind, f"{kind}/{name}")
    if item["id"] in records:
        raise ValueError(f"duplicate {kind} id: {item['id']}")
    records[item["id"]] = item


def _read_directory_fd(root_fd: int, kind: str) -> dict[str, dict]:
    """Relative descriptor opens prevent manifests or their directory escaping root."""
    flags = os.O_RDONLY | os.O_NOFOLLOW
    try:
        directory_fd = os.open(kind, flags | os.O_DIRECTORY, dir_fd=root_fd)
    except FileNotFoundError:
        return {}
    records: dict[str, dict] = {}
    try:
        for name in sorted(os.listdir(directory_fd)):
            if not name.endswith(".md"):
                continue
            file_fd = os.open(name, flags | os.O_NONBLOCK, dir_fd=directory_fd)
            with os.fdopen(file_fd, "rb") as stream:
                if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                    raise ValueError(f"manifest is not a regular file: {kind}/{name}")
                _add(records, stream.read(MAX_MANIFEST_BYTES + 1), kind, name)
    finally:
        os.close(directory_fd)
    return records


def _is_link(path: Path) -> bool:
    return path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction())


def _read_directory_path(root: Path, kind: str) -> dict[str, dict]:
    """The same contract by path, for platforms without descriptor-relative opens."""
    directory = root / kind
    if _is_link(directory):
        raise OSError("catalog directory is a link")
    if not directory.exists():
        return {}
    if not directory.is_dir():
        raise OSError("catalog entry is not a directory")
    records: dict[str, dict] = {}
    for path in sorted(directory.iterdir()):
        if not path.name.endswith(".md"):
            continue
        if _is_link(path):
            raise OSError("manifest is a link")
        if not stat.S_ISREG(path.lstat().st_mode):
            raise ValueError(f"manifest is not a regular file: {kind}/{path.name}")
        with path.open("rb") as stream:
            _add(records, stream.read(MAX_MANIFEST_BYTES + 1), kind, path.name)
    return records


def _read_manifests(root: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    if not _DIR_FD:
        return _read_directory_path(root, "skills"), _read_directory_path(root, "plugins")
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        return _read_directory_fd(root_fd, "skills"), _read_directory_fd(root_fd, "plugins")
    finally:
        os.close(root_fd)


class Catalog:
    def __init__(
        self,
        root: str | Path,
        *,
        enabled_plugins: frozenset[str] = frozenset(),
        enabled_skills: frozenset[str] = frozenset(),
    ):
        for value in (enabled_plugins, enabled_skills):
            if not isinstance(value, frozenset) or any(not _identifier(item) for item in value):
                raise ValueError("enablement must be a frozenset of valid ids")
        self.root = Path(root).resolve(strict=True)
        self.enabled_plugins = enabled_plugins
        self.enabled_skills = enabled_skills
        self._snapshot: tuple[tuple[dict, ...], frozenset[str], int] = ((), frozenset(), 0)
        self._lock = RLock()
        self.reload()

    def reload(self) -> None:
        with self._lock:
            try:
                skills, plugins = _read_manifests(self.root)
            except OSError as error:
                raise ValueError("catalog paths must be contained regular files/directories without symlinks") from error
            if set(skills) & set(plugins):
                raise ValueError("skill and plugin ids must be globally unique")
            for plugin in plugins.values():
                unknown = set(plugin["skills"]) - skills.keys()
                if unknown:
                    raise ValueError(f"unknown skill references in {plugin['id']}: {sorted(unknown)}")
            if self.enabled_plugins - plugins.keys() or self.enabled_skills - skills.keys():
                raise ValueError("unknown explicitly enabled plugin or skill")
            active = set(self.enabled_skills)
            tool_ids = {tool for skill_id in self.enabled_skills for tool in skills[skill_id]["tools"]}
            for plugin_id in self.enabled_plugins:
                active.update(plugins[plugin_id]["skills"])
                tool_ids.update(plugins[plugin_id]["tools"])
            self._snapshot = (tuple(skills[key] for key in sorted(active)), frozenset(tool_ids), self._snapshot[2] + 1)

    @property
    def revision(self) -> int:
        with self._lock:
            return self._snapshot[2]

    def snapshot_with_revision(self) -> tuple[list[dict], frozenset[str], int]:
        with self._lock:
            skills, tool_ids, revision = self._snapshot
        return (
            [{key: list(value) if isinstance(value, tuple) else value for key, value in skill.items()} for skill in skills],
            tool_ids,
            revision,
        )

    def skills(self) -> list[dict]:
        return self.snapshot_with_revision()[0]

    def tool_ids(self) -> frozenset[str]:
        with self._lock:
            return self._snapshot[1]

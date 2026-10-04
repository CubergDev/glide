"""R2 audit: files that hold goals, mail excerpts, local config and recorded runs are never staged by `git add -A`.

The test reads `.gitignore` itself (the offline guard refuses subprocesses), so it checks the patterns, not git."""

from fnmatch import fnmatch
from pathlib import Path

import pytest

LINES = [
    line.strip()
    for line in (Path(__file__).resolve().parent.parent / ".gitignore").read_text().splitlines()
    if line.strip() and not line.startswith("#")
]
PRIVATE = (
    ".glide-webhooks/queue.sqlite3",
    ".glide-webhooks/queue.sqlite3-wal",
    ".glide-webhooks/queue.sqlite3-shm",
    ".glide-webhooks/queue.sqlite3.key",
    "glide.toml",
    "webhooks.toml",
    "events.jsonl",
    "diagnostic.json",
    "failure.md",
)


def ignored(path: str) -> bool:
    parts = path.split("/")
    for pattern in LINES:
        if pattern.endswith("/") and pattern[:-1] in parts[:-1]:
            return True
        if not pattern.endswith("/") and fnmatch(parts[-1], pattern):
            return True
    return False


@pytest.mark.parametrize("path", PRIVATE)
def test_private_runtime_state_is_ignored(path):
    assert ignored(path), f"{path} is not in .gitignore"

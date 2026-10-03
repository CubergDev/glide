"""AGENTS.md recording rules for memory: nothing without opt-in, and no utterance or URL in what is kept besides what the user saves."""

import json
import re
import sqlite3
from pathlib import Path

from glide.memory import Catalog, Harness, MemoryService, MemorySettings, Model, Scope, Store

SCOPE = Scope("alice", "project", "session")
ROOT = Path(__file__).resolve().parents[2] / "glide"


def test_planning_a_turn_stores_ids_not_the_goal_or_memory_text(tmp_path):
    (tmp_path / "catalog").mkdir()
    with Store(tmp_path / "s.sqlite") as store:
        harness = Harness(store, Catalog(tmp_path / "catalog"), models=(Model("label", frozenset({"write"}), 32768),))
        store.remember(SCOPE, "style", "PRIVATE-MEMORY-TEXT concise")
        harness.prepare(SCOPE, "UTTERANCE-TEXT draft a reply about https://example.com/private", stage="write")
        events = json.dumps(store.events(SCOPE))
    assert "UTTERANCE-TEXT" not in events and "PRIVATE-MEMORY-TEXT" not in events and "example.com" not in events


def test_the_event_table_holds_only_allowlisted_metadata(tmp_path):
    from glide.memory import Event, SQLiteEventSink

    with Store(tmp_path / "s.sqlite") as store:
        sink = SQLiteEventSink(store, default_scope=SCOPE)
        sink(
            Event(
                "1",
                "mcp.call_finished",
                "mcp:files",
                {
                    "name": "read",
                    "arguments": {"path": "/Users/x/secret.txt"},
                    "result": "TEXT",
                    "message": "LOG",
                    "status": "success",
                    "duration_ms": 5,
                },
                1.0,
                SCOPE,
                "call-1",
            )
        )
        saved = json.dumps(store.events(SCOPE))
    assert "secret.txt" not in saved and "TEXT" not in saved and "LOG" not in saved
    assert '"name": "read"' in saved or '"name":"read"' in saved


def test_nothing_is_written_when_memory_is_off(tmp_path):
    for settings in (MemorySettings(data_dir=tmp_path / "a"), MemorySettings.from_mapping({"data_dir": str(tmp_path / "b")}, {})):
        assert MemoryService.open(settings) is None
    assert not (tmp_path / "a").exists() and not (tmp_path / "b").exists()


def test_stored_tables_are_exactly_the_documented_ones(tmp_path):
    with Store(tmp_path / "s.sqlite") as store:
        store.remember(SCOPE, "k", "text")
    db = sqlite3.connect(tmp_path / "s.sqlite")
    tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    db.close()
    assert tables == {"memories", "events", "outcomes", "harness", "overlays", "proposals"}


def test_no_endpoint_or_model_literals_in_the_packages():
    """D6: endpoints and model ids are configuration, never code."""
    url = re.compile(r"https?://")
    offenders = []
    for package in ("memory", "mcp"):
        for path in (ROOT / package).rglob("*.py"):
            for number, line in enumerate(path.read_text().splitlines(), 1):
                if url.search(line):
                    offenders.append(f"{path.relative_to(ROOT)}:{number}")
    assert offenders == []

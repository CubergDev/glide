"""Review findings for glide.memory: forget really deletes, run ids are opaque, a bad manifest does not lock the store."""

import io
import json
import os
import sqlite3
from pathlib import Path

import pytest

from glide.memory import MemoryService, MemorySettings, Scope, Store, cli
from glide.memory import catalog as catalog_module
from glide.memory.catalog import Catalog

SCOPE = Scope("alice", "project", "session")
MARKER = "zebra-marker-7f3a9c-do-not-keep"


def wal_bytes(path: Path) -> bytes:
    wal = Path(str(path) + "-wal")
    return wal.read_bytes() if wal.exists() else b""


def test_forget_removes_the_text_from_the_write_ahead_log_too(tmp_path):
    """PR5-4175258147: with journal_mode=WAL the deleted row's text lingers in old frames until a checkpoint."""
    database = tmp_path / "memory.sqlite"
    with Store(database) as store:
        identifier = store.remember(SCOPE, "note", f"I like {MARKER}")
        assert MARKER.encode() in wal_bytes(database)  # the precondition the finding relies on
        assert store.forget(SCOPE, identifier) is True
        assert MARKER.encode() not in wal_bytes(database)
        assert MARKER.encode() not in database.read_bytes()
        assert store.forget(SCOPE, identifier) is False  # nothing left to forget, nothing to truncate


def test_forget_says_so_when_a_reader_keeps_the_log_from_being_emptied(tmp_path):
    database = tmp_path / "memory.sqlite"
    with Store(database) as store:
        store._db.execute("PRAGMA busy_timeout=0")  # do not wait for the reader below
        identifier = store.remember(SCOPE, "note", f"I like {MARKER}")
        reader = sqlite3.connect(database, isolation_level=None)
        try:
            reader.execute("BEGIN")
            reader.execute("SELECT count(*) FROM memories").fetchall()  # holds a read snapshot
            with pytest.raises(sqlite3.OperationalError, match="write-ahead log"):
                store.forget(SCOPE, identifier)
            assert store.memories(SCOPE) == []  # the row is gone either way
        finally:
            reader.close()


@pytest.mark.parametrize(
    "run_id", ["https://example.com/run/1", "/Users/me/runs/run.json", "C:\\runs\\1", "two words", "x" * 129]
)
def test_a_run_id_that_is_a_url_or_path_is_refused(tmp_path, run_id):
    """PR5-4175597265"""
    with Store(tmp_path / "s.sqlite") as store:
        with pytest.raises(ValueError, match="opaque"):
            store.record_outcome(SCOPE, run_id, True, "Use short steps")
        assert store.outcomes(SCOPE) == []
        store.record_outcome(SCOPE, "run:" + "ab" * 32, True, "Use short steps")  # what ingest_run produces
        store.record_outcome(SCOPE, "run-1", True, "Use short steps")


def make_catalog(root: Path) -> None:
    (root / "skills").mkdir(parents=True)
    (root / "plugins").mkdir()
    manifest = {"id": "mail", "description": "d", "tools": [], "keywords": ["email"], "stages": ["handoff"]}
    (root / "skills" / "mail.md").write_text("---\n" + json.dumps(manifest) + "\n---\nSummarize the mail.")


def test_the_catalog_loads_where_directory_descriptors_do_not_exist(tmp_path, monkeypatch):
    """PR5-4175258144: Windows has no O_NOFOLLOW/O_DIRECTORY/dir_fd; the path-based reader gives the same answer."""
    make_catalog(tmp_path)
    expected = Catalog(tmp_path, enabled_skills=frozenset({"mail"})).skills()
    monkeypatch.setattr(catalog_module, "_DIR_FD", False)
    for flag in ("O_NOFOLLOW", "O_DIRECTORY"):
        monkeypatch.delattr(os, flag, raising=False)
    assert Catalog(tmp_path, enabled_skills=frozenset({"mail"})).skills() == expected


@pytest.mark.skipif(os.name != "posix", reason="symlinks need POSIX here")
def test_the_portable_catalog_reader_still_refuses_links_and_bad_manifests(tmp_path, monkeypatch):
    monkeypatch.setattr(catalog_module, "_DIR_FD", False)
    make_catalog(tmp_path)
    outside = tmp_path.parent / "outside.md"
    outside.write_text("---\n{}\n---\nbody")
    (tmp_path / "skills" / "escape.md").symlink_to(outside)
    with pytest.raises(ValueError, match="symlinks"):
        Catalog(tmp_path)
    (tmp_path / "skills" / "escape.md").unlink()
    (tmp_path / "skills" / "bad.md").write_text("not frontmatter")
    with pytest.raises(ValueError, match="frontmatter"):
        Catalog(tmp_path)
    (tmp_path / "skills" / "bad.md").unlink()
    (tmp_path / "skills").rename(tmp_path / "real")
    (tmp_path / "skills").symlink_to(tmp_path / "real", target_is_directory=True)
    with pytest.raises(ValueError, match="symlinks"):
        Catalog(tmp_path)


def test_a_malformed_manifest_does_not_stop_store_only_commands(tmp_path):
    """PR5-4175632619: `glide memory remember/recall/forget` do not read the catalog."""
    data = tmp_path / "data"
    env = {"GLIDE_DATA_DIR": str(data), "GLIDE_MEMORY": "1"}
    catalog = data / "memory" / "catalog"
    (catalog / "skills").mkdir(parents=True)
    (catalog / "skills" / "broken.md").write_text("this is not a manifest")

    def run(*args):
        out, err = io.StringIO(), io.StringIO()
        code = cli.main(["--user", "u", "--project", "p", *args], environ=env, home=tmp_path / "home", out=out, err=err)
        return code, out.getvalue(), err.getvalue()

    code, out, err = run("remember", "style", "concise")
    assert code == 0, err
    identifier = json.loads(out)
    assert run("recall")[0] == 0
    assert json.loads(run("forget", identifier)[1]) == {"forgotten": True}


def test_a_malformed_manifest_surfaces_when_the_harness_is_needed(tmp_path):
    data = tmp_path / "data"
    settings = MemorySettings(enabled=True, data_dir=data)
    with MemoryService.open(settings, home=tmp_path) as service:
        (service.database.parent / "catalog" / "skills").mkdir(parents=True)
        (service.database.parent / "catalog" / "skills" / "broken.md").write_text("nope")
        with pytest.raises(ValueError, match="frontmatter"):
            service.harness  # noqa: B018


def test_a_subscriber_that_raises_keyboard_interrupt_does_not_swallow_it():
    """audit2 finding 7: publish turned a Ctrl-C inside a subscriber into an errors entry and carried on."""
    from glide.memory.events import EventBus

    bus = EventBus()

    def interrupted(event):
        raise KeyboardInterrupt

    bus.subscribe(interrupted)
    with pytest.raises(KeyboardInterrupt):
        bus.publish("k", "s", {})
    assert bus.errors and bus.errors[-1]["error_type"] == "KeyboardInterrupt"

    def broken(event):
        raise ValueError

    other = EventBus()
    other.subscribe(broken)
    other.publish("k", "s", {})  # an ordinary bug is still only recorded
    assert other.errors[-1]["error_type"] == "ValueError"

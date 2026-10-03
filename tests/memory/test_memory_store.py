"""SQLite integration tests for this extension; no host/provider/device calls."""

import os
import sqlite3
import stat
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from glide.memory.contracts import Scope
from glide.memory.store import Store, validate_text


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "state.sqlite"
        self.scope = Scope("alice", "project-a", "session-a")
        self.store = Store(self.path)

    def tearDown(self):
        self.store.close()
        self.directory.cleanup()

    def evidence(self, *, scope=None, run="run-1"):
        return self.store.record_outcome(scope or self.scope, run, True, "Use concise steps")

    def test_reopen_preserves_memories_evidence_and_harness(self):
        memory = self.store.remember(self.scope, "style", "Keep replies concise", source="tool", confidence=0.7)
        evidence = self.evidence()
        proposal = self.store.propose(self.scope, "prompt:lesson", "Use concise steps", [evidence])
        self.assertEqual(self.store.apply(self.scope, proposal, expected_revision=0), 1)
        event = self.store.event(self.scope, "completed", {"proposal_id": proposal})
        self.store.close()
        self.store = Store(self.path)
        self.assertEqual(self.store.memories(self.scope)[0]["id"], memory)
        self.assertEqual(self.store.outcomes(self.scope)[0]["id"], evidence)
        self.assertEqual(self.store.proposals(self.scope)[0]["status"], "applied")
        self.assertEqual(self.store.overlays_with_revision(self.scope), (1, {"prompt:lesson": "Use concise steps"}))
        self.assertEqual(self.store.events(self.scope)[0]["id"], event)

    def test_failed_snapshot_query_does_not_poison_connection(self):
        with sqlite3.connect(self.path) as db:
            db.execute("ALTER TABLE overlays RENAME TO displaced_overlays")
        try:
            with self.assertRaises(sqlite3.OperationalError):
                self.store.overlays_with_revision(self.scope)
        finally:
            with sqlite3.connect(self.path) as db:
                db.execute("ALTER TABLE displaced_overlays RENAME TO overlays")
        memory = self.store.remember(self.scope, "style", "Connection remains usable")
        self.assertEqual(self.store.memories(self.scope)[0]["id"], memory)
        self.assertEqual(self.store.overlays_with_revision(self.scope), (0, {}))

    def test_migration_failure_rolls_back_all_new_schema(self):
        path = Path(self.directory.name) / "collision.sqlite"
        with sqlite3.connect(path) as db:
            db.execute("CREATE TABLE events (original TEXT)")
        with self.assertRaises(sqlite3.OperationalError):
            Store(path)
        with sqlite3.connect(path) as db:
            self.assertEqual(db.execute("PRAGMA user_version").fetchone()[0], 0)
            names = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            self.assertEqual(names, {"events"})

    def test_newer_schema_is_refused_without_version_change(self):
        path = Path(self.directory.name) / "future.sqlite"
        with sqlite3.connect(path) as db:
            db.execute("PRAGMA user_version=2")
        with self.assertRaisesRegex(ValueError, "newer"):
            Store(path)
        with sqlite3.connect(path) as db:
            self.assertEqual(db.execute("PRAGMA user_version").fetchone()[0], 2)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'").fetchone()[0], 0)

    def test_memory_precedence_access_upsert_and_forget(self):
        user = self.store.remember(self.scope, "style", "User style", level="user")
        project = self.store.remember(self.scope, "style", "Project style", level="project")
        session = self.store.remember(self.scope, "style", "Session style", level="session")
        other_session = Scope("alice", "project-a", "session-b")
        other_project = Scope("alice", "project-b", "session-a")
        other_user = Scope("bob", "project-a", "session-a")
        self.assertEqual(self.store.memories(self.scope)[0]["id"], session)
        self.assertEqual(self.store.memories(other_session)[0]["id"], project)
        self.assertEqual(self.store.memories(other_project)[0]["id"], user)
        self.assertEqual(self.store.memories(other_user), [])
        self.assertFalse(self.store.forget(other_session, session))
        self.assertFalse(self.store.forget(other_project, project))
        self.assertFalse(self.store.forget(other_user, user))
        self.assertEqual(self.store.remember(self.scope, "style", "Updated session", level="session"), session)
        self.assertTrue(self.store.forget(self.scope, session))
        self.assertEqual(self.store.memories(self.scope)[0]["text"], "Project style")
        self.assertTrue(self.store.forget(self.scope, project))
        self.assertEqual(self.store.memories(self.scope)[0]["text"], "User style")
        self.assertTrue(self.store.forget(other_project, user))
        self.assertEqual(self.store.memories(self.scope), [])
        with sqlite3.connect(self.path) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM memories").fetchone()[0], 0)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM events").fetchone()[0], 0)

    def test_ttl_expires_override_and_purges_on_next_memory_write(self):
        with patch("glide.memory.store.time.time", return_value=1000.0):
            self.store.remember(self.scope, "style", "Fallback", level="user")
            expired = self.store.remember(self.scope, "style", "Temporary", ttl_seconds=1)
            self.assertEqual(self.store.memories(self.scope)[0]["text"], "Temporary")
        with patch("glide.memory.store.time.time", return_value=1002.0):
            self.assertEqual(self.store.memories(self.scope)[0]["text"], "Fallback")
            self.store.remember(self.scope, "another", "New memory")
        with sqlite3.connect(self.path) as db:
            self.assertIsNone(db.execute("SELECT id FROM memories WHERE id=?", (expired,)).fetchone())

    def test_invalid_memory_numbers_are_rejected(self):
        for value in (-1, 2, True, float("nan"), float("inf"), 10**1000, "0.5"):
            with self.subTest(confidence_type=type(value).__name__), self.assertRaises(ValueError):
                self.store.remember(self.scope, "key", "Value", confidence=value)
        for value in (-1, 0, True, float("nan"), float("inf"), 10**1000, "1"):
            with self.subTest(ttl_type=type(value).__name__), self.assertRaises(ValueError):
                self.store.remember(self.scope, "key", "Value", ttl_seconds=value)
        self.assertEqual(self.store.memories(self.scope), [])

    def test_first_outcome_is_idempotent_and_project_scoped(self):
        first = self.store.record_outcome(self.scope, "run-1", False, "First verified result")
        repeat = self.store.record_outcome(self.scope, "run-1", True, "Changed result")
        self.assertEqual(first, repeat)
        rows = self.store.outcomes(self.scope)
        self.assertEqual(rows, [{"id": first, "run_id": "run-1", "success": False, "summary": "First verified result"}])
        same_project_session = Scope("alice", "project-a", "session-b")
        self.assertEqual(self.store.outcomes(same_project_session), rows)
        other = Scope("alice", "project-b", "session-a")
        self.assertNotEqual(self.store.record_outcome(other, "run-1", True, "Other result"), first)
        self.assertEqual(len(self.store.outcomes(other)), 1)

    def test_evidence_must_be_distinct_existing_and_same_project(self):
        own = self.evidence()
        other = self.evidence(scope=Scope("bob", "project-a", "session-a"), run="other")
        for ids in ([], ["missing"], [other], [own, own], [own] * 65):
            with self.subTest(evidence_count=len(ids)), self.assertRaises(ValueError):
                self.store.propose(self.scope, "prompt:lesson", "Instruction", ids)
        for target in ("policy:settings", "code:module", "tool:id", "model:id", "prompt:", "skill:with space"):
            with self.subTest(target=target), self.assertRaises(ValueError):
                self.store.propose(self.scope, target, "Instruction", [own])
        self.assertEqual(self.store.proposals(self.scope), [])
        self.assertEqual(self.store.revision(self.scope), 0)

    def test_failed_apply_and_rollback_leave_no_partial_change(self):
        proposal = self.store.propose(self.scope, "prompt:lesson", "Instruction", [self.evidence()])
        with self.assertRaisesRegex(ValueError, "revision"):
            self.store.apply(self.scope, proposal, expected_revision=1)
        self.assertEqual(self.store.overlays_with_revision(self.scope), (0, {}))
        self.assertEqual(self.store.proposals(self.scope)[0]["status"], "draft")
        self.assertEqual(self.store.apply(self.scope, proposal, expected_revision=0), 1)
        with self.assertRaisesRegex(ValueError, "revision"):
            self.store.rollback(self.scope, proposal, expected_revision=0)
        self.assertEqual(self.store.overlays_with_revision(self.scope), (1, {"prompt:lesson": "Instruction"}))
        self.assertEqual(self.store.proposals(self.scope)[0]["status"], "applied")
        self.assertEqual(self.store.rollback(self.scope, proposal, expected_revision=1), 2)
        self.assertEqual(self.store.overlays_with_revision(self.scope), (2, {}))
        with self.assertRaises(ValueError):
            self.store.rollback(self.scope, proposal, expected_revision=2)
        self.assertEqual(self.store.revision(self.scope), 2)

    def test_stale_draft_and_historical_text_can_be_reproposed(self):
        evidence = [self.evidence()]
        a = self.store.propose(self.scope, "prompt:lesson", "A", evidence)
        self.assertEqual(self.store.propose(self.scope, "prompt:lesson", "A", evidence), a)
        b = self.store.propose(self.scope, "prompt:lesson", "B", evidence)
        self.store.apply(self.scope, b, expected_revision=0)
        with self.assertRaisesRegex(ValueError, "target changed"):
            self.store.apply(self.scope, a, expected_revision=1)
        fresh_a = self.store.propose(self.scope, "prompt:lesson", "A", evidence)
        self.assertNotEqual(fresh_a, a)
        self.store.apply(self.scope, fresh_a, expected_revision=1)
        self.assertEqual(self.store.propose(self.scope, "prompt:lesson", "A", evidence), fresh_a)
        fresh_b = self.store.propose(self.scope, "prompt:lesson", "B", evidence)
        self.assertNotEqual(fresh_b, b)
        self.store.apply(self.scope, fresh_b, expected_revision=2)
        self.assertEqual(self.store.overlays_with_revision(self.scope), (3, {"prompt:lesson": "B"}))

    def test_stacked_rollback_restores_ancestry_across_other_target_edits(self):
        evidence = [self.evidence()]
        a = self.store.propose(self.scope, "prompt:lesson", "A", evidence)
        self.store.apply(self.scope, a, expected_revision=0)
        b = self.store.propose(self.scope, "prompt:lesson", "B", evidence)
        self.store.apply(self.scope, b, expected_revision=1)
        c = self.store.propose(self.scope, "skill:other", "C", evidence)
        self.store.apply(self.scope, c, expected_revision=2)
        with self.assertRaisesRegex(ValueError, "target changed"):
            self.store.rollback(self.scope, a, expected_revision=3)
        self.assertEqual(self.store.rollback(self.scope, b, expected_revision=3), 4)
        self.assertEqual(self.store.overlays(self.scope), {"prompt:lesson": "A", "skill:other": "C"})
        self.assertEqual(self.store.rollback(self.scope, a, expected_revision=4), 5)
        self.assertEqual(self.store.overlays_with_revision(self.scope), (5, {"skill:other": "C"}))

    def test_concurrent_connections_deduplicate_proposals(self):
        evidence = [self.evidence()]
        barrier = threading.Barrier(4)

        def propose():
            with Store(self.path) as store:
                barrier.wait(timeout=5)
                return store.propose(self.scope, "prompt:lesson", "Shared instruction", evidence)

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(propose) for _ in range(4)]
            ids = [future.result(timeout=15) for future in futures]
        self.assertEqual(len(set(ids)), 1)
        self.assertEqual(len(self.store.proposals(self.scope)), 1)

    def test_concurrent_revision_compare_and_swap_has_one_winner(self):
        evidence = [self.evidence()]
        proposals = [self.store.propose(self.scope, f"prompt:lesson-{index}", "Instruction", evidence) for index in range(2)]
        barrier = threading.Barrier(2)

        def apply(identifier):
            with Store(self.path) as store:
                barrier.wait(timeout=5)
                try:
                    return store.apply(self.scope, identifier, expected_revision=0)
                except ValueError:
                    return None

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(apply, identifier) for identifier in proposals]
            revisions = [future.result(timeout=15) for future in futures]
        self.assertCountEqual(revisions, [1, None])
        self.assertEqual(self.store.revision(self.scope), 1)
        self.assertEqual(len(self.store.overlays(self.scope)), 1)
        self.assertCountEqual([row["status"] for row in self.store.proposals(self.scope)], ["applied", "draft"])

    def test_events_are_decoded_limited_and_exactly_scoped(self):
        scopes = (
            self.scope,
            Scope("alice", "project-a", "session-b"),
            Scope("alice", "project-b", "session-a"),
            Scope("bob", "project-a", "session-a"),
        )
        for index, scope in enumerate(scopes):
            self.store.event(scope, "observed", {"index": index, "nested": ["你好", True, None]})
        own = self.store.event(self.scope, "finished", {"ok": True})
        rows = self.store.events(self.scope, limit=1)
        self.assertEqual(rows[0]["id"], own)
        self.assertEqual(rows[0]["payload"], {"ok": True})
        self.assertEqual(set(rows[0]), {"id", "kind", "payload", "created_at"})
        self.assertEqual(len(self.store.events(self.scope)), 2)
        for scope in scopes[1:]:
            self.assertEqual(len(self.store.events(scope)), 1)
        for limit in (0, 1001, True, 1.5, "1"):
            with self.assertRaises(ValueError):
                self.store.events(self.scope, limit)

    def test_event_retention_visibility_and_per_project_cap(self):
        with patch("glide.memory.store.time.time", return_value=1000.0):
            old = self.store.event(self.scope, "old", {})
        with patch("glide.memory.store.time.time", return_value=1000.0 + 31 * 86400):
            self.assertEqual(self.store.events(self.scope), [])
            self.store.event(self.scope, "recent", {})
        with sqlite3.connect(self.path) as db:
            self.assertIsNone(db.execute("SELECT id FROM events WHERE id=?", (old,)).fetchone())
        baseline = 1000.0 + 31 * 86400
        other = Scope("bob", "project-a", "session-a")
        with patch("glide.memory.store.time.time", return_value=baseline):
            self.store.event(other, "independent", {})
        first = None
        for index in range(1001):
            with patch("glide.memory.store.time.time", return_value=baseline + index + 1):
                identifier = self.store.event(self.scope, "trace", {"index": index})
                first = first or identifier
        with patch("glide.memory.store.time.time", return_value=baseline + 1002):
            rows = self.store.events(self.scope, limit=1000)
            self.assertEqual(len(rows), 1000)
            self.assertNotIn(first, {row["id"] for row in rows})
            self.assertEqual(len(self.store.events(other)), 1)

    def test_persisted_boundaries_reject_credentials_and_invalid_json(self):
        evidence = [self.evidence()]
        examples = (
            "password=not-a-real-password",
            "sk-proj-" + "A" * 40,
            "Bearer " + "A" * 30,
            "-----BEGIN PRIVATE KEY-----",
        )
        for index, text in enumerate(examples):
            with self.subTest(pattern=index):
                for save in (
                    lambda text=text: validate_text(text),
                    lambda text=text: self.store.remember(self.scope, "credential", text),
                    lambda text=text: self.store.propose(self.scope, "prompt:credential", text, evidence),
                    lambda text=text: self.store.record_outcome(self.scope, "sensitive", True, text),
                    lambda text=text: self.store.event(self.scope, "sensitive", {"nested": {"text": text}}),
                ):
                    with self.assertRaisesRegex(ValueError, "credentials"):
                        save()
        for payload in ({"number": float("nan")}, {"text": "x" * 16385}, {"object": object()}):
            with self.assertRaises(ValueError):
                self.store.event(self.scope, "invalid", payload)
        self.assertEqual(self.store.memories(self.scope), [])
        self.assertEqual(self.store.proposals(self.scope), [])
        self.assertEqual(self.store.events(self.scope), [])
        self.assertEqual(len(self.store.outcomes(self.scope)), 1)

    def test_connection_configuration_and_private_creation(self):
        for pragma, expected in (
            ("journal_mode", "wal"),
            ("foreign_keys", 1),
            ("busy_timeout", 5000),
            ("secure_delete", 1),
            ("user_version", 1),
        ):
            self.assertEqual(self.store._db.execute(f"PRAGMA {pragma}").fetchone()[0], expected)
        if os.name == "posix":
            self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o600)
            self.store.close()
            self.path.chmod(0o640)
            self.store = Store(self.path)
            self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o600)  # a looser mode is tightened
        with self.assertRaises(FileNotFoundError):
            Store(Path(self.directory.name) / "missing" / "database.sqlite")
        with self.assertRaises(ValueError):
            Store(Path(self.directory.name))
        with self.assertRaises(ValueError):
            Store("file:unsupported.sqlite")

    @unittest.skipUnless(os.name == "posix", "symlink checks require POSIX")
    def test_database_symlink_is_refused(self):
        link = Path(self.directory.name) / "link.sqlite"
        link.symlink_to(self.path)
        with self.assertRaisesRegex(ValueError, "regular file"):
            Store(link)

    def test_inherited_connection_is_refused_after_fork(self):
        """A forked child has another pid. Simulated in-process (no os.fork, D13): the store only compares pids."""
        child_pid = os.getpid() + 1
        with patch("glide.memory.store.os.getpid", return_value=child_pid):
            with self.assertRaisesRegex(RuntimeError, "after a process fork"):
                self.store.memories(self.scope)
            with self.assertRaisesRegex(RuntimeError, "after a process fork"):
                self.store.remember(self.scope, "k", "text")
            with self.assertRaisesRegex(RuntimeError, "after a process fork"):
                self.store.close()
        # back in the owning process the connection is intact: the refusal changed nothing
        self.assertEqual(self.store.memories(self.scope), [])

    @unittest.skipUnless(os.name == "posix", "file modes require POSIX")
    def test_database_files_and_sidecars_are_owner_only(self):
        self.store.remember(self.scope, "k", "text")  # first write creates the -wal and -shm files
        found = [Path(str(self.path) + suffix) for suffix in ("", "-wal", "-shm")]
        self.assertTrue(found[0].exists())
        for path in found:
            if path.exists():
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600, path.name)

    def test_outcome_summaries_refuse_urls_and_credentials(self):
        for summary in ("opened https://example.com/page", "see www.example.com", "ftp://host/file"):
            with self.assertRaisesRegex(ValueError, "URL"):
                self.store.record_outcome(self.scope, "run-url", True, summary)
        with self.assertRaisesRegex(ValueError, "credentials"):
            self.store.record_outcome(self.scope, "run-key", True, "password = hunter2hunter2")
        self.assertEqual(self.store.outcomes(self.scope), [])

    def test_closed_store_and_context_manager_lifecycle(self):
        self.store.close()
        self.store.close()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            self.store.events(self.scope)
        with Store(self.path) as reopened:
            self.assertEqual(reopened.revision(self.scope), 0)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            reopened.memories(self.scope)


if __name__ == "__main__":
    unittest.main()


from memory_guard_loader import no_real_memory_or_mcp  # noqa: E402, F401  (autouse guard)

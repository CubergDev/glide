"""Explicit SQLite persistence; importing this module performs no I/O.

A Store holds user text, so Glide only opens one after the user opted in (see
`glide.memory.settings` and `glide.memory.service`); constructing a Store directly is the low-level
API that the tests and the service use. What it keeps, and nothing else:

- memories: key, text, kind, source, confidence, scope ids, timestamps, optional expiry (user text)
- outcomes: a run id (a hash, never a path or URL) and a short tactic summary; URLs are refused
- proposals and overlays: refinement text derived from outcomes, with evidence ids
- events: a metadata allowlist (ids, names, counts, status); never arguments, results or messages

Events retain at most 30 days and 1,000 rows per user/project; pruning happens
on event writes. Expired memories are hidden immediately and pruned on memory
writes. Forgetting hard-deletes memory text (secure_delete is on); no memory-text audit is generated.
Credential rejection is a conservative heuristic, not a complete secret scanner.
Database files (and their -wal/-shm/-journal sidecars) are kept at mode 0600.
"""

import json
import math
import os
import re
import sqlite3
import stat
import time
from contextlib import contextmanager
from os import PathLike
from threading import RLock
from uuid import uuid4

from .contracts import Scope

_VERSION = 1
_EVENT_DAYS = 30
_EVENT_LIMIT = 1000
_TARGET = re.compile(r"(?:prompt|skill):[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_SECRET = re.compile(
    r"\bsk-(?:proj-|svcacct-|live-|test-)?[A-Za-z0-9_-]{16,}\b"
    r"|\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"
    r"|\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"
    r"|\bxox[baprs]-[A-Za-z0-9-]{10,}\b"
    r"|\bBearer\s+[A-Za-z0-9._~+/=-]{12,}"
    r"|-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"
    r"|[a-z][a-z0-9+.-]*://[^\s/:]+:[^\s/@]+@",
    re.IGNORECASE,
)
_ASSIGNED_SECRET = re.compile(
    r"\b(?:password|passwd|api[_ -]?key|client[_ -]?secret|access[_ -]?token|"
    r"refresh[_ -]?token|private[_ -]?key|secret)[\"']?\s*[:=]\s*[\"']?"
    r"([^\s,;\"'}]+)",
    re.IGNORECASE,
)
_URL = re.compile(r"[a-z][a-z0-9+.-]*://\S|\bwww\.[a-z0-9-]+\.", re.IGNORECASE)
_PLACEHOLDERS = frozenset(
    {"redacted", "<redacted>", "***", "...", "none", "null", "example", "string", "your_api_key", "your_token"}
)


def validate_text(text: str, *, max_length: int = 16384) -> str:
    """Bound persisted text and reject recognizable credentials without echoing it."""
    if not isinstance(text, str) or len(text) > max_length or "\x00" in text:
        raise ValueError("text must be a bounded string without NUL characters")
    if _SECRET.search(text) or any(match.group(1).lower() not in _PLACEHOLDERS for match in _ASSIGNED_SECRET.finditer(text)):
        raise ValueError("text appears to contain credentials; redact them first")
    return text


def validate_summary(text: str, *, max_length: int = 8192) -> str:
    """A run-derived summary: bounded, credential-free and without raw URLs (AGENTS.md recording rule)."""
    validate_text(text, max_length=max_length)
    if _URL.search(text):
        raise ValueError("run summaries must not contain URLs; describe the tactic instead")
    return text


def _identifier(value: str, *, maximum: int = 256) -> str:
    validate_text(value, max_length=maximum)
    if not value.strip() or any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError("identifier must be nonempty and contain no control characters")
    return value


def _scope(scope: Scope) -> tuple[str, str, str]:
    if not isinstance(scope, Scope):
        raise TypeError("scope must be a Scope")
    return tuple(_identifier(value) for value in (scope.user, scope.project, scope.session))


def _target(target: str) -> str:
    if not isinstance(target, str) or not _TARGET.fullmatch(target):
        raise ValueError("target must be prompt:NAME or skill:ID using letters, digits, ._-")
    return target


def _revision(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**63 - 1:
        raise ValueError("expected_revision must be a nonnegative integer")
    return value


def _limit(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 1000:
        raise ValueError("limit must be an integer between 1 and 1000")
    return value


def _prepare_path(path: str | PathLike[str]) -> str:
    """Create local database files privately without changing existing file modes."""
    path = os.fspath(path)
    if not isinstance(path, str) or not path or "\x00" in path or path.startswith("file:"):
        raise ValueError("database path must be a regular local path or :memory:")
    if path == ":memory:":
        return path
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        if not stat.S_ISREG(os.lstat(path).st_mode):
            raise ValueError("database path must reference a regular file") from None
    else:
        os.close(descriptor)
    return path


def _harden(path: str) -> None:
    """Keep the database and the files SQLite derives from it readable by this user only."""
    if path == ":memory:":
        return
    for suffix in ("", "-wal", "-shm", "-journal"):
        try:
            os.chmod(path + suffix, 0o600)
        except FileNotFoundError:
            continue


_SCHEMA = (
    """CREATE TABLE memories (
        id TEXT PRIMARY KEY, user TEXT NOT NULL, project TEXT NOT NULL,
        session TEXT NOT NULL, level TEXT NOT NULL CHECK(level IN ('user','project','session')),
        key TEXT NOT NULL, text TEXT NOT NULL, kind TEXT NOT NULL, source TEXT NOT NULL,
        confidence REAL NOT NULL CHECK(confidence BETWEEN 0 AND 1),
        created_at REAL NOT NULL, updated_at REAL NOT NULL, expires_at REAL,
        UNIQUE(user, project, session, level, key))""",
    "CREATE INDEX memories_scope ON memories(user, project, session)",
    """CREATE TABLE events (
        id TEXT PRIMARY KEY, user TEXT NOT NULL, project TEXT NOT NULL,
        session TEXT NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL,
        created_at REAL NOT NULL)""",
    "CREATE INDEX events_scope_time ON events(user, project, created_at)",
    """CREATE TABLE outcomes (
        id TEXT PRIMARY KEY, user TEXT NOT NULL, project TEXT NOT NULL,
        run_id TEXT NOT NULL, success INTEGER NOT NULL CHECK(success IN (0,1)),
        summary TEXT NOT NULL, created_at REAL NOT NULL, UNIQUE(user, project, run_id))""",
    """CREATE TABLE harness (
        user TEXT NOT NULL, project TEXT NOT NULL, revision INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY(user, project))""",
    """CREATE TABLE overlays (
        user TEXT NOT NULL, project TEXT NOT NULL, target TEXT NOT NULL,
        text TEXT NOT NULL, version INTEGER NOT NULL, PRIMARY KEY(user, project, target),
        FOREIGN KEY(user, project) REFERENCES harness(user, project))""",
    """CREATE TABLE proposals (
        id TEXT PRIMARY KEY, user TEXT NOT NULL, project TEXT NOT NULL,
        target TEXT NOT NULL, text TEXT NOT NULL, evidence_ids TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'draft' CHECK(status IN ('draft','applied','rolled_back')),
        created_at REAL NOT NULL, applied_revision INTEGER, rollback_revision INTEGER,
        prior_text TEXT, prior_version INTEGER, prior_present INTEGER,
        base_target_version INTEGER, base_target_text TEXT)""",
)
_ACCESS = """user=? AND (level='user' OR (project=? AND
            (level='project' OR (level='session' AND session=?))))"""


class Store:
    """One connection serialized by an RLock; separate instances use SQLite locking."""

    def __init__(self, path: str | PathLike[str]):
        self._lock = RLock()
        self._closed = False
        self._pid = os.getpid()
        path = _prepare_path(path)
        self._db = sqlite3.connect(path, timeout=5.0, isolation_level=None, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        try:
            if self._db.execute("PRAGMA user_version").fetchone()[0] > _VERSION:
                raise ValueError("database schema is newer than this extension")
            self._db.execute("PRAGMA busy_timeout=5000")
            self._db.execute("PRAGMA foreign_keys=ON")
            self._db.execute("PRAGMA secure_delete=ON")
            self._db.execute("PRAGMA journal_mode=WAL")
            with self._transaction() as db:
                version = db.execute("PRAGMA user_version").fetchone()[0]
                if version > _VERSION:
                    raise ValueError("database schema is newer than this extension")
                if version == 0:
                    for statement in _SCHEMA:
                        db.execute(statement)
                    db.execute("PRAGMA user_version=1")
            _harden(path)
        except BaseException:
            self.close()
            raise

    def __enter__(self) -> "Store":
        self._check_open()
        with self._lock:
            self._check_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self) -> None:
        if os.getpid() != self._pid:
            raise RuntimeError("open a new Store after a process fork")
        with self._lock:
            if not self._closed:
                self._db.close()
                self._closed = True

    def _check_open(self) -> None:
        if os.getpid() != self._pid:
            raise RuntimeError("open a new Store after a process fork")
        if self._closed:
            raise RuntimeError("store is closed")

    @contextmanager
    def _transaction(self, *, write: bool = True):
        """Serialize writes and read snapshots with the same commit/rollback guard."""
        self._check_open()
        with self._lock:
            self._check_open()
            self._db.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            try:
                yield self._db
                self._db.commit()
            except BaseException:
                if self._db.in_transaction:
                    self._db.rollback()
                raise

    def _rows(self, sql: str, parameters: tuple = ()) -> list[dict]:
        self._check_open()
        with self._lock:
            self._check_open()
            return [dict(row) for row in self._db.execute(sql, parameters).fetchall()]

    def remember(
        self,
        scope: Scope,
        key: str,
        text: str,
        *,
        kind: str = "preference",
        source: str = "user",
        confidence: float = 1.0,
        level: str = "project",
        ttl_seconds: float | None = None,
    ) -> str:
        user, project, session = _scope(scope)
        key, kind, source = (_identifier(value, maximum=128) for value in (key, kind, source))
        validate_text(text, max_length=8192)
        if not text.strip() or level not in ("user", "project", "session"):
            raise ValueError("memory requires nonempty text and a valid level")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not 0 <= confidence <= 1
            or not math.isfinite(confidence)
        ):
            raise ValueError("confidence must be finite and between 0 and 1")
        if ttl_seconds is not None and (
            isinstance(ttl_seconds, bool)
            or not isinstance(ttl_seconds, (int, float))
            or not 0 < ttl_seconds <= 86400 * 3650
            or not math.isfinite(ttl_seconds)
        ):
            raise ValueError("TTL must be finite, positive and at most ten years")
        project = project if level != "user" else ""
        session = session if level == "session" else ""
        now, memory_id = time.time(), uuid4().hex
        expires = None if ttl_seconds is None else now + ttl_seconds
        with self._transaction() as db:
            db.execute("DELETE FROM memories WHERE expires_at <= ?", (now,))
            db.execute(
                """INSERT INTO memories VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(user,project,session,level,key) DO UPDATE SET
                text=excluded.text, kind=excluded.kind, source=excluded.source,
                confidence=excluded.confidence, updated_at=excluded.updated_at,
                expires_at=excluded.expires_at""",
                (
                    memory_id,
                    user,
                    project,
                    session,
                    level,
                    key,
                    text,
                    kind,
                    source,
                    float(confidence),
                    now,
                    now,
                    expires,
                ),
            )
            return db.execute(
                """SELECT id FROM memories WHERE
                user=? AND project=? AND session=? AND level=? AND key=?""",
                (user, project, session, level, key),
            ).fetchone()[0]

    def memories(self, scope: Scope) -> list[dict]:
        rows = self._rows(
            f"""SELECT id,key,text,kind,source,confidence,updated_at
            FROM memories WHERE {_ACCESS} AND (expires_at IS NULL OR expires_at > ?)
            ORDER BY CASE level WHEN 'session' THEN 0 WHEN 'project' THEN 1 ELSE 2 END,
            key, updated_at DESC""",
            (*_scope(scope), time.time()),
        )
        selected = {}
        for row in rows:
            selected.setdefault(row["key"], row)
        return list(selected.values())

    def forget(self, scope: Scope, id: str) -> bool:
        parameters = (_identifier(id), *_scope(scope))
        with self._transaction() as db:
            return db.execute(f"DELETE FROM memories WHERE id=? AND {_ACCESS}", parameters).rowcount == 1

    def event(self, scope: Scope, kind: str, payload: dict) -> str:
        scoped, kind = _scope(scope), _identifier(kind, maximum=128)
        if not isinstance(payload, dict):
            raise ValueError("event payload must be a dictionary")
        try:
            encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True)
        except (TypeError, ValueError, OverflowError, RecursionError) as error:
            raise ValueError("event payload must contain bounded JSON values") from error
        validate_text(encoded)
        event_id, now = uuid4().hex, time.time()
        with self._transaction() as db:
            db.execute("INSERT INTO events VALUES (?,?,?,?,?,?,?)", (event_id, *scoped, kind, encoded, now))
            db.execute("DELETE FROM events WHERE created_at < ?", (now - _EVENT_DAYS * 86400,))
            db.execute(
                """DELETE FROM events WHERE id IN (SELECT id FROM events
                WHERE user=? AND project=? ORDER BY created_at DESC,id DESC
                LIMIT -1 OFFSET ?)""",
                (*scoped[:2], _EVENT_LIMIT),
            )
        return event_id

    def events(self, scope: Scope, limit: int = 50) -> list[dict]:
        """Read retained events for exactly this user, project and session."""
        scoped = _scope(scope)
        limit = _limit(limit)
        rows = self._rows(
            """SELECT id,kind,payload,created_at FROM events
            WHERE user=? AND project=? AND session=? AND created_at >= ?
            ORDER BY created_at DESC,id DESC LIMIT ?""",
            (*scoped, time.time() - _EVENT_DAYS * 86400, limit),
        )
        for row in rows:
            row["payload"] = json.loads(row["payload"])
        return rows

    def record_outcome(self, scope: Scope, run_id: str, success: bool, summary: str) -> str:
        user, project, _ = _scope(scope)
        run_id = _identifier(run_id)
        validate_summary(summary)
        if not isinstance(success, bool):
            raise ValueError("success must be a boolean")
        with self._transaction() as db:
            db.execute(
                "INSERT OR IGNORE INTO outcomes VALUES (?,?,?,?,?,?,?)",
                (uuid4().hex, user, project, run_id, int(success), summary, time.time()),
            )
            return db.execute(
                "SELECT id FROM outcomes WHERE user=? AND project=? AND run_id=?", (user, project, run_id)
            ).fetchone()[0]

    def outcomes(self, scope: Scope, limit: int = 50) -> list[dict]:
        scoped = _scope(scope)
        limit = _limit(limit)
        rows = self._rows(
            """SELECT id,run_id,success,summary FROM outcomes
            WHERE user=? AND project=? ORDER BY created_at DESC,id DESC LIMIT ?""",
            (*scoped[:2], limit),
        )
        for row in rows:
            row["success"] = bool(row["success"])
        return rows

    @staticmethod
    def _evidence(db, scoped: tuple, evidence_ids: list[str]) -> list[str]:
        if not isinstance(evidence_ids, list):
            raise ValueError("evidence IDs must be a list")
        ids = list(evidence_ids)
        if not 1 <= len(ids) <= 64:
            raise ValueError("a proposal requires 1 to 64 outcome evidence IDs")
        ids = [_identifier(value) for value in ids]
        if len(set(ids)) != len(ids):
            raise ValueError("evidence IDs must be distinct")
        placeholders = ",".join("?" for _ in ids)
        count = db.execute(
            f"""SELECT COUNT(*) FROM outcomes WHERE user=? AND project=?
            AND id IN ({placeholders})""",
            (*scoped[:2], *ids),
        ).fetchone()[0]
        if count != len(ids):
            raise ValueError("evidence must reference outcomes from this user and project")
        return ids

    def propose(self, scope: Scope, target: str, text: str, evidence_ids: list[str]) -> str:
        scoped, target = _scope(scope), _target(target)
        validate_text(text, max_length=16384)
        if not text.strip():
            raise ValueError("proposal text must be nonempty")
        proposal_id = uuid4().hex
        with self._transaction() as db:
            evidence_ids = self._evidence(db, scoped, evidence_ids)
            base = db.execute(
                "SELECT text,version FROM overlays WHERE user=? AND project=? AND target=?", (*scoped[:2], target)
            ).fetchone()
            base_version = base["version"] if base else None
            base_text = base["text"] if base else None
            existing = db.execute(
                """SELECT id FROM proposals WHERE user=? AND project=?
                AND target=? AND text=? AND (
                    (status='draft' AND base_target_version IS ? AND base_target_text IS ?)
                    OR (status='applied' AND applied_revision IS ?))
                ORDER BY created_at DESC,id DESC LIMIT 1""",
                (*scoped[:2], target, text, base_version, base_text, base_version),
            ).fetchone()
            if existing:
                return existing["id"]
            db.execute(
                """INSERT INTO proposals
                (id,user,project,target,text,evidence_ids,created_at,base_target_version,base_target_text)
                VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    proposal_id,
                    *scoped[:2],
                    target,
                    text,
                    json.dumps(evidence_ids),
                    time.time(),
                    base_version,
                    base_text,
                ),
            )
        return proposal_id

    def proposals(self, scope: Scope) -> list[dict]:
        rows = self._rows(
            """SELECT id,target,text,evidence_ids,status,created_at,
            applied_revision,rollback_revision,base_target_version,base_target_text
            FROM proposals WHERE user=? AND project=?
            ORDER BY created_at DESC,id DESC""",
            _scope(scope)[:2],
        )
        for row in rows:
            row["evidence_ids"] = json.loads(row["evidence_ids"])
        return rows

    @staticmethod
    def _advance(db, scoped: tuple, expected: int) -> int:
        db.execute("INSERT OR IGNORE INTO harness(user,project) VALUES (?,?)", scoped[:2])
        changed = db.execute(
            """UPDATE harness SET revision=revision+1
            WHERE user=? AND project=? AND revision=?""",
            (*scoped[:2], expected),
        ).rowcount
        if changed != 1:
            raise ValueError("harness revision conflict; refresh before applying")
        return expected + 1

    @staticmethod
    def _proposal(db, scoped: tuple, proposal_id: str, status: str):
        row = db.execute("SELECT * FROM proposals WHERE id=? AND user=? AND project=?", (proposal_id, *scoped[:2])).fetchone()
        if row is None or row["status"] != status:
            raise ValueError(f"proposal must belong to this project and be {status}")
        _target(row["target"])
        return row

    def apply(self, scope: Scope, proposal_id: str, *, expected_revision: int) -> int:
        scoped = _scope(scope)
        proposal_id, expected = _identifier(proposal_id), _revision(expected_revision)
        with self._transaction() as db:
            proposal = self._proposal(db, scoped, proposal_id, "draft")
            self._evidence(db, scoped, json.loads(proposal["evidence_ids"]))
            prior = db.execute(
                "SELECT text,version FROM overlays WHERE user=? AND project=? AND target=?",
                (*scoped[:2], proposal["target"]),
            ).fetchone()
            if (prior["version"] if prior else None) != proposal["base_target_version"] or (
                prior["text"] if prior else None
            ) != proposal["base_target_text"]:
                raise ValueError("target changed since this proposal was drafted")
            revision = self._advance(db, scoped, expected)
            db.execute(
                """INSERT INTO overlays VALUES (?,?,?,?,?) ON CONFLICT(user,project,target)
                DO UPDATE SET text=excluded.text,version=excluded.version""",
                (*scoped[:2], proposal["target"], proposal["text"], revision),
            )
            db.execute(
                """UPDATE proposals SET status='applied',applied_revision=?,
                prior_text=?,prior_version=?,prior_present=? WHERE id=?""",
                (
                    revision,
                    prior["text"] if prior else None,
                    prior["version"] if prior else None,
                    int(prior is not None),
                    proposal_id,
                ),
            )
        return revision

    def rollback(self, scope: Scope, proposal_id: str, *, expected_revision: int) -> int:
        scoped = _scope(scope)
        proposal_id, expected = _identifier(proposal_id), _revision(expected_revision)
        with self._transaction() as db:
            proposal = self._proposal(db, scoped, proposal_id, "applied")
            current = db.execute(
                "SELECT version FROM overlays WHERE user=? AND project=? AND target=?",
                (*scoped[:2], proposal["target"]),
            ).fetchone()
            if current is None or current["version"] != proposal["applied_revision"]:
                raise ValueError("target changed after this proposal; rollback refused")
            revision = self._advance(db, scoped, expected)
            if proposal["prior_present"]:
                db.execute(
                    "UPDATE overlays SET text=?,version=? WHERE user=? AND project=? AND target=?",
                    (proposal["prior_text"], proposal["prior_version"], *scoped[:2], proposal["target"]),
                )
            else:
                db.execute("DELETE FROM overlays WHERE user=? AND project=? AND target=?", (*scoped[:2], proposal["target"]))
            db.execute("UPDATE proposals SET status='rolled_back',rollback_revision=? WHERE id=?", (revision, proposal_id))
        return revision

    def revision(self, scope: Scope) -> int:
        rows = self._rows("SELECT revision FROM harness WHERE user=? AND project=?", _scope(scope)[:2])
        return rows[0]["revision"] if rows else 0

    def overlays(self, scope: Scope) -> dict[str, str]:
        rows = self._rows("SELECT target,text FROM overlays WHERE user=? AND project=? ORDER BY target", _scope(scope)[:2])
        return {row["target"]: row["text"] for row in rows}

    def overlays_with_revision(self, scope: Scope) -> tuple[int, dict[str, str]]:
        """Read the revision and overlays from the same SQLite snapshot."""
        scoped = _scope(scope)[:2]
        with self._transaction(write=False) as db:
            revision = db.execute("SELECT revision FROM harness WHERE user=? AND project=?", scoped).fetchone()
            rows = db.execute("SELECT target,text FROM overlays WHERE user=? AND project=? ORDER BY target", scoped).fetchall()
            return (revision["revision"] if revision else 0, {row["target"]: row["text"] for row in rows})

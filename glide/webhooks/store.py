"""Durable webhook admission and single-owner agent leases, without execution.

Delivery acknowledgement follows an atomic SQLite commit. A lost agent lease is
uncertain rather than retryable: computer actions may already have happened.

D3 recording rule. Unless `record_content=True`, the database never keeps what the task was about once the task
is over: a message that reaches a terminal state (other than `uncertain`, which an operator still has to review)
has its goal and context replaced by a fixed placeholder, worker summaries are replaced by the worker's content-free
`note`, and event text is dropped. `secure_delete` and a WAL truncate make the replaced bytes actually leave the
files. Receipts keep only a source, an event id and a digest of the authenticated body.
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import math
import os
import secrets
import sqlite3
import stat
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

TERMINAL = frozenset({"completed", "blocked", "cancelled", "failed", "uncertain"})
MAX_CALL_BYTES = 65536
MAX_EVENT_BYTES = 8192
MAX_TASK_EVENTS = 100
MAX_SUMMARY_BYTES = 4096
REDACTED_GOAL = "redacted"
RESOLVED_NOTE = "Reconciled by an operator."


class DeliveryConflict(RuntimeError):
    """A delivery identity was reused for different authenticated content."""


class QueueFull(RuntimeError):
    """The bounded durable store cannot accept another delivery."""


class LeaseConflict(RuntimeError):
    """A request does not own the current agent lease."""


def _json(value: dict, maximum: int) -> str:
    if not isinstance(value, dict):
        raise ValueError("A message must be an object.")
    try:
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("A message must contain valid JSON values.") from error
    if len(encoded.encode("utf-8")) > maximum:
        raise ValueError("The message exceeds the storage limit.")
    return encoded


def _redacted_call(encoded: str) -> str:
    """The call without its goal and context: enough to identify the message, nothing about its subject."""
    call = json.loads(encoded)
    kept = {
        key: call[key]
        for key in ("schema_version", "id", "task_id", "agent_id", "operation", "source", "event_id")
        if key in call
    }
    kept.update(goal=REDACTED_GOAL, context={}, allow_actions=False)
    return _json(kept, MAX_CALL_BYTES)


def _secret_digest(token: str) -> str:
    if not isinstance(token, str) or not token or len(token) > 1024:
        raise LeaseConflict("The agent lease is unavailable.")
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _lease_duration(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= 3600:
        raise ValueError("The lease duration must be positive and at most one hour.")
    return float(value)


class QueueStore:
    """Thread-safe store; SQLite transactions also serialize independent processes."""

    def __init__(
        self,
        path: Path,
        max_pending: int = 1000,
        max_events: int = 100000,
        *,
        clock: Callable[[], float] = time.time,
        record_content: bool = False,
    ):
        if isinstance(max_pending, bool) or not isinstance(max_pending, int) or max_pending < 1:
            raise ValueError("The pending-message limit must be positive.")
        if isinstance(max_events, bool) or not isinstance(max_events, int) or max_events < 1:
            raise ValueError("The delivery limit must be positive.")
        self.path = Path(path).absolute()
        self.max_pending = max_pending
        self.max_events = max_events
        self.clock = clock
        self.record_content = bool(record_content)
        self._lock = threading.RLock()
        self._closed = False
        self._prepare_path()
        self._db = sqlite3.connect(self.path, timeout=5, isolation_level=None, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        try:
            self._db.execute("PRAGMA busy_timeout=5000")
            self._db.execute("PRAGMA foreign_keys=ON")
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=FULL")
            self._db.execute("PRAGMA secure_delete=ON")
            self._db.executescript(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    agent_id TEXT NOT NULL,
                    call_json TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    lease_owner TEXT,
                    token_hash TEXT,
                    expires_at REAL,
                    resolved_at REAL,
                    summary TEXT NOT NULL DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS messages_agent_status ON messages(agent_id, status);
                CREATE TABLE IF NOT EXISTS deliveries (
                    source TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    digest TEXT NOT NULL,
                    message_id TEXT REFERENCES messages(id),
                    created_at REAL NOT NULL,
                    PRIMARY KEY(source, event_id)
                );
                CREATE INDEX IF NOT EXISTS deliveries_digest ON deliveries(source, digest);
                CREATE TABLE IF NOT EXISTS task_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message_id TEXT NOT NULL REFERENCES messages(id),
                    event_json TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS task_events_message ON task_events(message_id, id);
                """
            )
            # Existing queues retain their receipts and leases when operator
            # reconciliation support is added. Serialize the small migration.
            with self._transaction() as db:
                columns = {row["name"] for row in db.execute("PRAGMA table_info(messages)")}
                if "resolved_at" not in columns:
                    db.execute("ALTER TABLE messages ADD COLUMN resolved_at REAL")
            self._protect_files()
        except BaseException:
            self._db.close()
            raise

    def _prepare_path(self) -> None:
        parent = self.path.parent
        created_parent = not parent.exists()
        parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if created_parent and os.name == "posix":
            parent.chmod(0o700)
        self._regular_file(self.path, missing=True)
        for suffix in ("-wal", "-shm", "-journal"):
            self._regular_file(Path(str(self.path) + suffix), missing=True)
        if not self.path.exists():
            flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
            try:
                descriptor = os.open(self.path, flags, 0o600)
            except FileExistsError:
                # Another store may create the same database concurrently.
                self._regular_file(self.path)
            else:
                os.close(descriptor)
        if os.name == "posix":
            self.path.chmod(0o600)

    @staticmethod
    def _regular_file(path: Path, *, missing: bool = False) -> None:
        try:
            mode = path.lstat().st_mode
        except FileNotFoundError:
            if missing:
                return
            raise ValueError("The queue database is unavailable.") from None
        if not stat.S_ISREG(mode):
            raise ValueError("The queue database must be a regular file.")

    def _protect_files(self) -> None:
        if os.name != "posix":
            return
        for suffix in ("", "-wal", "-shm"):
            path = Path(str(self.path) + suffix)
            if path.exists():
                self._regular_file(path)
                path.chmod(0o600)

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            if self._closed:
                raise RuntimeError("The queue store is closed.")
            self._db.execute("BEGIN IMMEDIATE")
            try:
                yield self._db
            except BaseException:
                self._db.rollback()
                raise
            else:
                self._db.commit()
                self._protect_files()

    def _checkpoint(self) -> None:
        """Fold the WAL into the database and truncate it, so replaced content does not linger in old frames."""
        with self._lock:
            if self._closed:
                return
            with contextlib.suppress(sqlite3.Error):  # best effort: a busy reader only delays the truncate
                self._db.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchall()
            self._protect_files()

    def _redact(self, db: sqlite3.Connection, message_id: str) -> bool:
        if self.record_content:
            return False
        row = db.execute("SELECT call_json FROM messages WHERE id=?", (message_id,)).fetchone()
        if row is None:
            return False
        db.execute("UPDATE messages SET call_json=? WHERE id=?", (_redacted_call(row["call_json"]), message_id))
        return True

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._db.close()
                self._closed = True

    def enqueue(self, source: str, event_id: str, digest: str, call: dict | None, *, dedupe_body: bool = False) -> dict:
        return self.enqueue_many(
            [{"source": source, "event_id": event_id, "digest": digest, "call": call, "dedupe_body": dedupe_body}]
        )[0]

    def enqueue_many(self, deliveries: list[dict]) -> list[dict]:
        if not isinstance(deliveries, list) or len(deliveries) > 1000:
            raise ValueError("A delivery batch must contain at most one thousand items.")
        if not deliveries:
            return []
        prepared = []
        for delivery in deliveries:
            if not isinstance(delivery, dict) or not {"source", "event_id", "digest", "call"} <= delivery.keys():
                raise ValueError("The delivery batch is invalid.")
            if delivery.keys() - {"source", "event_id", "digest", "call", "dedupe_body"}:
                raise ValueError("The delivery batch is invalid.")
            source, event_id, digest, call = (delivery[key] for key in ("source", "event_id", "digest", "call"))
            encoded = self._prepare_delivery(source, event_id, digest, call)
            dedupe_body = delivery.get("dedupe_body", False)
            if not isinstance(dedupe_body, bool):
                raise ValueError("The delivery batch is invalid.")
            prepared.append((source, event_id, digest, encoded, dedupe_body))
        with self._transaction() as db:
            return [self._enqueue(db, *delivery) for delivery in prepared]

    @staticmethod
    def _prepare_delivery(source: str, event_id: str, digest: str, call: dict | None) -> str | None:
        if any(not isinstance(value, str) or not value or len(value) > 4096 for value in (source, event_id, digest)):
            raise ValueError("Delivery metadata is invalid.")
        encoded = None if call is None else _json(call, MAX_CALL_BYTES)
        if call is not None and any(not isinstance(call.get(key), str) or not call[key] for key in ("id", "agent_id")):
            raise ValueError("The agent message identity is invalid.")
        return encoded

    def _enqueue(
        self, db: sqlite3.Connection, source: str, event_id: str, digest: str, encoded: str | None, dedupe_body: bool
    ) -> dict:
        existing = db.execute(
            "SELECT digest, message_id FROM deliveries WHERE source=? AND event_id=?", (source, event_id)
        ).fetchone()
        if existing:
            if not hmac.compare_digest(existing["digest"].encode("utf-8"), digest.encode("utf-8")):
                raise DeliveryConflict("The delivery identity conflicts with an existing receipt.")
            return {"status": "duplicate", "message_id": existing["message_id"]}
        body_duplicate = (
            db.execute("SELECT message_id FROM deliveries WHERE source=? AND digest=? LIMIT 1", (source, digest)).fetchone()
            if dedupe_body
            else None
        )
        if body_duplicate:
            # An unsigned delivery-ID alias is not a new authenticated event.
            # Acknowledge it even at capacity without retaining arbitrary
            # aliases, so a captured valid payload cannot exhaust receipts.
            return {"status": "duplicate", "message_id": body_duplicate["message_id"]}
        if db.execute("SELECT COUNT(*) FROM deliveries").fetchone()[0] >= self.max_events:
            raise QueueFull("The webhook queue capacity has been reached.")
        now = self.clock()
        if encoded is None:
            message_id = None
            status = "ignored"
        else:
            count = db.execute("SELECT COUNT(*) FROM messages WHERE status IN ('pending','leased','uncertain')").fetchone()[0]
            if count >= self.max_pending:
                raise QueueFull("The webhook queue capacity has been reached.")
            call = json.loads(encoded)
            message_id = call["id"]
            try:
                db.execute(
                    "INSERT INTO messages(id, agent_id, call_json, created_at, updated_at) VALUES(?,?,?,?,?)",
                    (message_id, call["agent_id"], encoded, now, now),
                )
            except sqlite3.IntegrityError:
                raise DeliveryConflict("The agent message identity conflicts with an existing message.") from None
            status = "accepted"
        db.execute(
            "INSERT INTO deliveries(source,event_id,digest,message_id,created_at) VALUES(?,?,?,?,?)",
            (source, event_id, digest, message_id, now),
        )
        return {"status": status, "message_id": message_id}

    @staticmethod
    def _expire(db: sqlite3.Connection, agent_id: str, now: float) -> None:
        db.execute(
            "UPDATE messages SET status='uncertain', updated_at=?, summary='Agent lease expired; review before further execution.' "
            "WHERE agent_id=? AND status='leased' AND expires_at<=?",
            (now, agent_id, now),
        )

    def claim(self, agent_id: str, subject: str, lease_seconds: float = 60) -> dict | None:
        duration = _lease_duration(lease_seconds)
        if not isinstance(subject, str) or not subject:
            raise LeaseConflict("The agent lease is unavailable.")
        with self._transaction() as db:
            now = self.clock()
            self._expire(db, agent_id, now)
            if db.execute(
                "SELECT 1 FROM messages WHERE agent_id=? AND status IN ('leased','uncertain') LIMIT 1", (agent_id,)
            ).fetchone():
                return None
            row = db.execute(
                "SELECT * FROM messages WHERE agent_id=? AND status='pending' ORDER BY created_at,rowid LIMIT 1", (agent_id,)
            ).fetchone()
            if row is None:
                return None
            token = secrets.token_urlsafe(32)
            expires_at = now + duration
            db.execute(
                "UPDATE messages SET status='leased',lease_owner=?,token_hash=?,expires_at=?,updated_at=? WHERE id=?",
                (subject, _secret_digest(token), expires_at, now, row["id"]),
            )
            return {
                "message_id": row["id"],
                "call": json.loads(row["call_json"]),
                "lease_token": token,
                "leased_at": now,
                "expires_at": expires_at,
            }

    def _owned(
        self, db: sqlite3.Connection, agent_id: str, subject: str, message_id: str, token: str, now: float
    ) -> sqlite3.Row | None:
        self._expire(db, agent_id, now)
        row = db.execute("SELECT * FROM messages WHERE id=? AND agent_id=?", (message_id, agent_id)).fetchone()
        try:
            digest = _secret_digest(token)
        except LeaseConflict:
            return None
        if row is None or row["lease_owner"] != subject or not row["token_hash"]:
            return None
        return row if hmac.compare_digest(row["token_hash"], digest) else None

    def heartbeat(
        self,
        agent_id: str,
        subject: str,
        message_id: str,
        token: str,
        lease_seconds: float = 60,
        *,
        include_server_time: bool = False,
    ) -> float | dict:
        duration = _lease_duration(lease_seconds)
        expires_at = None
        with self._transaction() as db:
            now = self.clock()
            row = self._owned(db, agent_id, subject, message_id, token, now)
            if row is not None and row["status"] == "leased":
                expires_at = now + duration
                db.execute("UPDATE messages SET expires_at=?,updated_at=? WHERE id=?", (expires_at, now, message_id))
        if expires_at is None:
            raise LeaseConflict("The agent lease is unavailable.")
        return {"expires_at": expires_at, "server_time": now} if include_server_time else expires_at

    def finish(
        self,
        agent_id: str,
        subject: str,
        message_id: str,
        token: str,
        outcome: str,
        *,
        summary: str = "",
        note: str = "",
    ) -> dict:
        if outcome not in TERMINAL or not isinstance(summary, str) or not isinstance(note, str):
            raise ValueError("The task outcome is invalid.")
        # `summary` may carry task content; `note` is a content-free status phrase from the worker.
        kept = summary if self.record_content else note
        summary = kept.encode("utf-8")[:MAX_SUMMARY_BYTES].decode("utf-8", errors="ignore")
        result = None
        redacted = False
        with self._transaction() as db:
            now = self.clock()
            row = self._owned(db, agent_id, subject, message_id, token, now)
            if row is not None and row["status"] == "leased":
                db.execute("UPDATE messages SET status=?,summary=?,updated_at=? WHERE id=?", (outcome, summary, now, message_id))
                if outcome != "uncertain":
                    redacted = self._redact(db, message_id)
                result = {"message_id": message_id, "status": outcome}
            elif (
                row is not None
                and row["status"] == outcome
                and row["status"] in TERMINAL
                and row["updated_at"] < row["expires_at"]
            ):
                # Acknowledgement retries keep the first result, even after its lease deadline.
                result = {"message_id": message_id, "status": outcome}
        if redacted:
            self._checkpoint()
        if result is None:
            raise LeaseConflict("The agent lease is unavailable.")
        return result

    def publish(self, agent_id: str, subject: str, message_id: str, token: str, event: dict) -> dict:
        if not self.record_content and "text" in event:
            event = {**event, "text": ""}  # progress text can quote the task; keep the kind, outcome and timing
        encoded = _json(event, MAX_EVENT_BYTES)
        result = None
        with self._transaction() as db:
            now = self.clock()
            row = self._owned(db, agent_id, subject, message_id, token, now)
            task_matches = row is not None and (
                event.get("task_id") is None or event["task_id"] == json.loads(row["call_json"]).get("task_id")
            )
            if row is not None and row["status"] == "leased" and task_matches:
                cursor = db.execute(
                    "INSERT INTO task_events(message_id,event_json,created_at) VALUES(?,?,?)", (message_id, encoded, now)
                )
                db.execute(
                    "DELETE FROM task_events WHERE message_id=? AND id NOT IN "
                    "(SELECT id FROM task_events WHERE message_id=? ORDER BY id DESC LIMIT ?)",
                    (message_id, message_id, MAX_TASK_EVENTS),
                )
                result = {"status": "published", "event_id": cursor.lastrowid}
        if result is None:
            raise LeaseConflict("The agent lease is unavailable.")
        return result

    def resolve(self, agent_id: str, message_id: str, outcome: str, *, summary: str = "") -> dict:
        """Explicit operator reconciliation; this never requeues or reruns a task.

        The API must authorize this separately from ordinary worker leases.
        Clearing the old lease prevents its owner from changing a reconciled
        result or sending new events after the operator has reviewed it.
        """
        if outcome not in {"completed", "failed", "cancelled"} or not isinstance(summary, str):
            raise ValueError("The reconciliation outcome is invalid.")
        summary = (
            (summary if self.record_content else RESOLVED_NOTE)
            .encode("utf-8")[:MAX_SUMMARY_BYTES]
            .decode("utf-8", errors="ignore")
        )
        result = None
        redacted = False
        with self._transaction() as db:
            now = self.clock()
            self._expire(db, agent_id, now)
            row = db.execute("SELECT * FROM messages WHERE id=? AND agent_id=?", (message_id, agent_id)).fetchone()
            if row is not None and row["status"] == "uncertain":
                db.execute(
                    "UPDATE messages SET status=?,summary=?,updated_at=?,resolved_at=?,lease_owner=NULL,token_hash=NULL WHERE id=?",
                    (outcome, summary, now, now, message_id),
                )
                redacted = self._redact(db, message_id)
                row = db.execute("SELECT * FROM messages WHERE id=?", (message_id,)).fetchone()
                result = self._metadata(db, row)
            elif row is not None and row["resolved_at"] is not None and row["status"] == outcome:
                result = self._metadata(db, row)
        if redacted:
            self._checkpoint()
        if result is None:
            raise LeaseConflict("The task cannot be reconciled in its current state.")
        return result

    @staticmethod
    def _metadata(db: sqlite3.Connection, row: sqlite3.Row) -> dict:
        events = db.execute("SELECT event_json FROM task_events WHERE message_id=? ORDER BY id", (row["id"],)).fetchall()
        return {
            "message_id": row["id"],
            "status": row["status"],
            "call": json.loads(row["call_json"]),
            "summary": row["summary"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "expires_at": row["expires_at"],
            "resolved_at": row["resolved_at"],
            "events": [json.loads(event["event_json"]) for event in events],
        }

    def get(self, agent_id: str, message_id: str) -> dict | None:
        with self._transaction() as db:
            self._expire(db, agent_id, self.clock())
            row = db.execute("SELECT * FROM messages WHERE id=? AND agent_id=?", (message_id, agent_id)).fetchone()
            return self._metadata(db, row) if row is not None else None

    def messages(self, agent_id: str, *, limit: int = 50, status: str | None = None) -> list[dict]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("The message listing limit must be between one and one hundred.")
        if status is not None and status not in TERMINAL | {"pending", "leased"}:
            raise ValueError("The task status is invalid.")
        with self._transaction() as db:
            self._expire(db, agent_id, self.clock())
            if status is None:
                rows = db.execute(
                    "SELECT * FROM messages WHERE agent_id=? ORDER BY created_at DESC,rowid DESC LIMIT ?", (agent_id, limit)
                ).fetchall()
            else:
                rows = db.execute(
                    "SELECT * FROM messages WHERE agent_id=? AND status=? ORDER BY created_at DESC,rowid DESC LIMIT ?",
                    (agent_id, status, limit),
                ).fetchall()
            return [self._metadata(db, row) for row in rows]

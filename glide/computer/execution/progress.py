"""Atomic effect ledger and the bounded waits. The ledger holds identities and counts, never task content."""

import os
import sqlite3
import time
import uuid

from ..control import checkpoint
from ..diagnostics import event
from .contracts import InvalidAction, effect, primitive_effect


class Ledger:
    """What each milestone needs and which observed effects have been counted for it, per task, in one SQLite file.

    The file is 0600 (created before SQLite opens it, so its WAL files follow). A milestone's contract is fixed once
    registered: a replan may add milestones but cannot change the meaning of one already counted. The planner's
    milestone ids can echo the request, so the file holds an opaque key per milestone (`m1`, `m2`, ...) and the
    planner's id stays in memory (D3). A pending operation blocks every task that reuses the journal: the CLI mints a
    new task id per invocation, so the id cannot identify a restarted run.
    """

    def __init__(self, path, task_id):
        self.task_id = task_id
        self.keys: dict[str, str] = {}  # planner id -> the key stored in the file, for this run only
        os.close(os.open(path, os.O_CREAT | os.O_WRONLY, 0o600))
        self.db = sqlite3.connect(path)
        self.db.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS milestones(task TEXT, id TEXT, contract TEXT, requested INTEGER,
                PRIMARY KEY(task,id));
            CREATE TABLE IF NOT EXISTS effects(task TEXT, milestone TEXT, effect TEXT,
                PRIMARY KEY(task,milestone,effect));
            CREATE TABLE IF NOT EXISTS operations(id TEXT PRIMARY KEY, task TEXT, milestone TEXT,
                action_hash TEXT, status TEXT, elapsed REAL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS revisions(task TEXT, version INTEGER, PRIMARY KEY(task,version));
        """)

    def key(self, milestone):
        """The opaque key of a milestone id, minted on first sight."""
        return self.keys.setdefault(milestone, f"m{len(self.keys) + 1}")

    def register(self, steps, revision):
        with self.db:
            for s in steps:
                key = self.key(s.id)
                old = self.db.execute("SELECT contract FROM milestones WHERE task=? AND id=?", (self.task_id, key)).fetchone()
                if old and old[0] != s.contract:
                    raise InvalidAction("Replanning changed an existing completion contract")
                self.db.execute("INSERT OR IGNORE INTO milestones VALUES(?,?,?,?)", (self.task_id, key, s.contract, s.quantity))
            self.db.execute("INSERT OR IGNORE INTO revisions VALUES(?,?)", (self.task_id, revision))

    def count(self, milestone):
        row = self.db.execute(
            "SELECT count(*) FROM effects WHERE task=? AND milestone=?", (self.task_id, self.key(milestone))
        ).fetchone()
        return row[0]

    def begin(self, step, action):
        op = str(uuid.uuid4())
        with self.db:
            self.db.execute(
                "INSERT INTO operations(id,task,milestone,action_hash,status) VALUES(?,?,?,?,?)",
                (op, self.task_id, self.key(step.id), action.identity, "pending"),
            )
        return op

    def finish(self, operation, step, effect_id, elapsed, operation_effect=""):
        with self.db:
            if effect_id:
                self.db.execute("INSERT OR IGNORE INTO effects VALUES(?,?,?)", (self.task_id, self.key(step.id), effect_id))
            self.db.execute(
                "UPDATE operations SET status=?, elapsed=? WHERE id=?",
                ("verified" if effect_id or operation_effect else "no_effect", elapsed, operation),
            )

    def observe_effect(self, step, effect_id):
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO effects VALUES(?,?,?)", (self.task_id, self.key(step.id), effect_id))

    def unresolved(self):
        row = self.db.execute("SELECT 1 FROM operations WHERE status='pending' LIMIT 1").fetchone()
        return bool(row)

    def summary(self, *, opaque=False):
        """Counts per milestone, under the planner's ids (the planner needs them) or, for what is stored, the opaque keys."""
        names = {key: name for name, key in self.keys.items()}
        rows = self.db.execute("SELECT id,requested FROM milestones WHERE task=? ORDER BY rowid", (self.task_id,)).fetchall()
        out = []
        for key, requested in rows:
            verified = self.db.execute(
                "SELECT count(*) FROM effects WHERE task=? AND milestone=?", (self.task_id, key)
            ).fetchone()[0]
            out.append(
                {
                    "id": key if opaque else names.get(key, key),
                    "requested": requested,
                    "verified": verified,
                    "remaining": max(0, requested - verified),
                }
            )
        return out

    def close(self):
        self.db.close()


def observe(backend, step=None, action=None):
    """Use an adapter's scoped snapshot when available; fakes and other adapters keep full reads."""
    scoped = getattr(backend, "inspect_for", None)
    return scoped(step, action) if scoped and step is not None else backend.inspect()


def wait_effect(backend, step, action, before, receipt, *, timeout, poll=0.05):
    """Read after a dispatched action until its effect is observed or `timeout` seconds pass: `(after, verified)`.

    An intermediate operation (typing into the field a later keypress submits, say) ends the wait early unless the
    milestone can only be proved by its final effect. There is no fixed settle delay: the wait is the poll.
    """
    deadline = time.monotonic() + timeout
    needs_final_effect = step.effect == "media_playing" or (
        step.effect == "query_submitted" and action.kind == "key" and action.parameter_source.startswith("query:")
    )
    while True:
        checkpoint()
        after = observe(backend, step, action)
        checkpoint()
        verified = effect(step, action, before, after, receipt)
        if (
            verified
            or (not needs_final_effect and primitive_effect(action, before, after, receipt))
            or time.monotonic() >= deadline
        ):
            return after, verified
        time.sleep(min(poll, max(0, deadline - time.monotonic())))


def wait_ready(backend, observed, timeout, *, step=None, action=None, condition=None):
    """Poll the required observed state, not just document readiness.

    A document that is not ready by the deadline is an `InvalidAction` (nothing was dispatched). A ready document that
    never offers what `condition` asks for is returned as it is at the deadline, so selection reports the missing
    capability itself.
    """
    deadline = time.monotonic() + timeout
    while not observed.ready or (condition is not None and not condition(observed)):
        checkpoint()
        if time.monotonic() >= deadline:
            if not observed.ready:
                event(
                    "readiness_timed_out",
                    timeout_s=timeout,
                    ready=observed.ready,
                    url=observed.url,
                    active_tab=observed.active_tab,
                )
                raise InvalidAction("The target did not become ready before the bounded deadline")
            break
        time.sleep(min(0.05, max(0, deadline - time.monotonic())))
        checkpoint()
        observed = observe(backend, step, action)
    checkpoint()
    return observed

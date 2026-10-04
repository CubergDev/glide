"""Durable webhook receipts and agent leases, without servers or real agents."""

from __future__ import annotations

import json
import os
import sqlite3
import stat
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from glide.webhooks.store import (
    MAX_EVENT_BYTES,
    MAX_SUMMARY_BYTES,
    MAX_TASK_EVENTS,
    DeliveryConflict,
    LeaseConflict,
    QueueFull,
    QueueStore,
)


@pytest.fixture
def stored(tmp_path):
    clock = SimpleNamespace(now=100.0)
    # Content recording on: these tests cover leases, receipts and ordering, not the D3 redaction.
    store = QueueStore(tmp_path / "queue" / "webhooks.sqlite", clock=lambda: clock.now, record_content=True)
    yield store, clock
    store.close()


@pytest.fixture
def redacting(tmp_path):
    """The default: nothing about a finished task's subject stays in the database."""
    clock = SimpleNamespace(now=100.0)
    store = QueueStore(tmp_path / "queue" / "webhooks.sqlite", clock=lambda: clock.now)
    yield store, clock
    store.close()


def call(identity="m1", *, agent="laptop"):
    return {
        "id": identity,
        "task_id": "task-" + identity,
        "agent_id": agent,
        "operation": "execute",
        "goal": "Summarize the pull request before making a change.",
        "source": "github",
        "event_id": identity,
        "allow_actions": False,
        "context": {"number": 12},
    }


def enqueue(store, identity="m1", *, agent="laptop", source="github", digest=None, **kwargs):
    return store.enqueue(source, identity, digest or "sha256-" + identity, call(identity, agent=agent), **kwargs)


def claim(store):
    return store.claim("laptop", "operator-one", lease_seconds=60)


def args(lease, *, agent="laptop", subject="operator-one", token=None):
    return agent, subject, lease["message_id"], token or lease["lease_token"]


def test_atomic_receipt_survives_restart_and_duplicate_never_changes_call(tmp_path):
    path = tmp_path / "webhooks.sqlite"
    store = QueueStore(path)
    original = call()
    assert store.enqueue("github", "delivery", "digest", original) == {"status": "accepted", "message_id": "m1"}
    original["goal"] = "Mutated after submission"
    store.close()
    store = QueueStore(path)
    try:
        assert store.enqueue("github", "delivery", "digest", call("other")) == {"status": "duplicate", "message_id": "m1"}
        assert store.get("laptop", "m1")["call"]["goal"] != original["goal"]
        with pytest.raises(DeliveryConflict):
            store.enqueue("github", "delivery", "different", call("other"))
        assert len(store.messages("laptop")) == 1
    finally:
        store.close()


def test_github_unsigned_delivery_id_cannot_replay_identical_signed_body(stored):
    store, _ = stored
    enqueue(store, digest="same-body", dedupe_body=True)
    assert enqueue(store, "m2", digest="same-body", dedupe_body=True) == {"status": "duplicate", "message_id": "m1"}
    assert store.get("laptop", "m2") is None
    with pytest.raises(DeliveryConflict):
        enqueue(store, "m1", digest="changed-body", dedupe_body=True)
    # An unsigned alias is not reserved against a subsequent distinct event.
    assert enqueue(store, "m2", digest="changed-body", dedupe_body=True)["status"] == "accepted"
    assert len(store.messages("laptop")) == 2


def test_equal_content_can_be_separate_events_when_id_is_signed(stored):
    store, _ = stored
    enqueue(store, digest="same")
    assert enqueue(store, "m2", digest="same")["status"] == "accepted"
    assert enqueue(store, "m3", source="other", digest="same", dedupe_body=True)["status"] == "accepted"
    assert len(store.messages("laptop")) == 3


def test_ignored_deliveries_have_receipts_and_never_create_agent_work(stored):
    store, _ = stored
    assert store.enqueue("github", "ignored", "digest", None) == {"status": "ignored", "message_id": None}
    assert store.enqueue("github", "ignored", "digest", call()) == {"status": "duplicate", "message_id": None}
    assert store.enqueue("github", "alias", "digest", call(), dedupe_body=True) == {"status": "duplicate", "message_id": None}
    assert claim(store) is None and store.messages("laptop") == []


def test_full_queue_rolls_back_receipt_and_duplicate_is_still_acknowledged(tmp_path):
    store = QueueStore(tmp_path / "queue.sqlite", max_pending=1)
    try:
        enqueue(store)
        with pytest.raises(QueueFull):
            enqueue(store, "m2")
        assert enqueue(store)["status"] == "duplicate"
        lease = claim(store)
        store.finish(*args(lease), "completed")
        assert enqueue(store, "m2")["status"] == "accepted"
    finally:
        store.close()


def test_receipt_capacity_includes_ignored_events_without_counting_unsigned_aliases(tmp_path):
    store = QueueStore(tmp_path / "queue.sqlite", max_events=2)
    try:
        store.enqueue("github", "one", "body", None)
        store.enqueue("github", "alias", "body", None, dedupe_body=True)
        assert store.enqueue("github", "third", "other", None)["status"] == "ignored"
        with pytest.raises(QueueFull):
            store.enqueue("github", "fourth", "new", None)
        assert store.enqueue("github", "one", "body", None)["status"] == "duplicate"
    finally:
        store.close()


def test_captured_body_aliases_do_not_consume_capacity_and_are_acknowledged_when_full(tmp_path):
    path = tmp_path / "queue.sqlite"
    store = QueueStore(path, max_events=1)
    try:
        enqueue(store, digest="captured-body", dedupe_body=True)
        for index in range(100):
            assert enqueue(store, f"unsigned-alias-{index}", digest="captured-body", dedupe_body=True) == {
                "status": "duplicate",
                "message_id": "m1",
            }
        with sqlite3.connect(path) as db:
            assert db.execute("SELECT COUNT(*) FROM deliveries").fetchone()[0] == 1
        assert len(store.messages("laptop")) == 1
        with pytest.raises(DeliveryConflict):
            enqueue(store, "m1", digest="different-body", dedupe_body=True)
        with pytest.raises(QueueFull):
            enqueue(store, "new-event", digest="different-body", dedupe_body=True)
    finally:
        store.close()


def test_multiple_connections_ack_duplicate_body_aliases_without_persisting_them(tmp_path):
    path = tmp_path / "queue.sqlite"
    stores = [QueueStore(path, max_events=1), QueueStore(path, max_events=1)]
    try:
        enqueue(stores[0], digest="captured-body", dedupe_body=True)
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(
                pool.map(
                    lambda item: enqueue(item[1], f"alias-{item[0]}", digest="captured-body", dedupe_body=True),
                    enumerate(stores),
                )
            )
        assert results == [{"status": "duplicate", "message_id": "m1"}] * 2
        with sqlite3.connect(path) as db:
            assert db.execute("SELECT COUNT(*) FROM deliveries").fetchone()[0] == 1
    finally:
        for store in stores:
            store.close()


def deliveries(*identities):
    return [{"source": "outlook", "event_id": identity, "digest": identity, "call": call(identity)} for identity in identities]


def test_delivery_batch_is_atomic_when_later_item_exhausts_capacity(tmp_path):
    store = QueueStore(tmp_path / "queue.sqlite", max_pending=1)
    try:
        with pytest.raises(QueueFull):
            store.enqueue_many(deliveries("m1", "m2"))
        assert store.messages("laptop") == []
        assert store.enqueue_many(deliveries("m1"))[0]["status"] == "accepted"
    finally:
        store.close()


def test_batch_delivery_conflict_rolls_back_earlier_new_receipts(stored):
    store, _ = stored
    batch = deliveries("m1", "m1")
    batch[1]["digest"] = "conflicting-content"
    with pytest.raises(DeliveryConflict):
        store.enqueue_many(batch)
    assert store.messages("laptop") == []
    assert store.enqueue_many(deliveries("m1"))[0]["status"] == "accepted"


def test_overlapping_notification_batches_ack_existing_items_once(stored):
    store, _ = stored
    first = store.enqueue_many(deliveries("m1", "m2"))
    assert [item["status"] for item in first] == ["accepted", "accepted"]
    second = store.enqueue_many(deliveries("m2", "m3", "m3"))
    assert [item["status"] for item in second] == ["duplicate", "accepted", "duplicate"]
    assert len(store.messages("laptop")) == 3


def test_batch_validates_all_messages_before_any_receipt(stored):
    store, _ = stored
    batch = deliveries("m1", "m2")
    batch[1]["call"]["context"] = {"invalid": float("nan")}
    with pytest.raises(ValueError):
        store.enqueue_many(batch)
    assert store.messages("laptop") == []


def test_empty_notification_batch_does_not_create_receipts(stored):
    store, _ = stored
    assert store.enqueue_many([]) == []
    assert store.messages("laptop") == []


def test_conflicting_message_identity_has_no_receipt_side_effect(stored):
    store, _ = stored
    enqueue(store)
    with pytest.raises(DeliveryConflict):
        store.enqueue("github", "new-delivery", "new-body", call())
    assert store.enqueue("github", "new-delivery", "new-body", call("m2"))["status"] == "accepted"


def test_claim_is_fifo_and_only_one_lease_is_active_per_agent(stored):
    store, _ = stored
    enqueue(store)
    enqueue(store, "m2")
    enqueue(store, "m3", agent="other")
    first = claim(store)
    assert first["call"]["id"] == "m1" and first["expires_at"] == 160
    assert claim(store) is None
    assert store.claim("laptop", "operator-two") is None
    other = store.claim("other", "operator-one")
    assert other["call"]["id"] == "m3" and other["lease_token"] != first["lease_token"]
    store.finish(*args(first), "completed")
    assert claim(store)["call"]["id"] == "m2"


def test_restart_preserves_exclusive_lease_and_token_is_never_persisted_or_returned_in_status(tmp_path):
    path = tmp_path / "queue.sqlite"
    store = QueueStore(path)
    enqueue(store)
    lease = claim(store)
    store.close()
    store = QueueStore(path)
    try:
        assert claim(store) is None
        assert lease["lease_token"] not in json.dumps(store.get("laptop", "m1"))
        with sqlite3.connect(path) as db:
            row = db.execute("SELECT token_hash,lease_owner,call_json,summary FROM messages").fetchone()
            assert row[0] != lease["lease_token"] and len(row[0]) == 64
            assert lease["lease_token"] not in repr(row)
        assert store.heartbeat(*args(lease))["expires_at"] > 0
    finally:
        store.close()


@pytest.mark.parametrize("bad", [{"agent": "other"}, {"subject": "operator-two"}, {"token": "stolen-or-invalid"}])
def test_lease_owner_agent_and_token_are_all_required(stored, bad):
    store, _ = stored
    enqueue(store)
    lease = claim(store)
    for action in (
        lambda: store.heartbeat(*args(lease, **bad)),
        lambda: store.finish(*args(lease, **bad), "completed"),
        lambda: store.publish(*args(lease, **bad), {"kind": "accepted"}),
    ):
        with pytest.raises(LeaseConflict):
            action()
    assert store.get("laptop", "m1")["status"] == "leased"
    assert store.get("other", "m1") is None


def test_heartbeat_extends_only_the_current_owned_lease(stored):
    store, clock = stored
    enqueue(store)
    lease = claim(store)
    clock.now = 150
    assert store.heartbeat(*args(lease), lease_seconds=30)["expires_at"] == 180
    clock.now = 160
    assert store.get("laptop", "m1")["status"] == "leased"
    clock.now = 180
    assert store.get("laptop", "m1")["status"] == "uncertain"


@pytest.mark.parametrize("operation", ["claim", "heartbeat", "finish", "publish"])
def test_expiry_is_durable_and_blocks_further_work_without_replaying(stored, operation):
    store, clock = stored
    enqueue(store)
    enqueue(store, "m2")
    lease = claim(store)
    clock.now = lease["expires_at"]
    if operation == "claim":
        assert claim(store) is None
    else:
        with pytest.raises(LeaseConflict):
            if operation == "heartbeat":
                store.heartbeat(*args(lease))
            elif operation == "finish":
                store.finish(*args(lease), "completed")
            else:
                store.publish(*args(lease), {"kind": "completed"})
    assert store.get("laptop", "m1")["status"] == "uncertain"
    assert claim(store) is None
    assert store.get("laptop", "m2")["status"] == "pending"
    with pytest.raises(LeaseConflict):
        store.finish(*args(lease), "uncertain")


def test_finish_ack_retry_keeps_the_first_outcome_summary_and_original_owner(stored):
    store, clock = stored
    enqueue(store)
    lease = claim(store)
    expected = {"message_id": "m1", "status": "completed"}
    assert store.finish(*args(lease), "completed", summary="Verified result") == expected
    clock.now = 500
    assert store.finish(*args(lease), "completed", summary="Later altered result") == expected
    assert store.get("laptop", "m1")["summary"] == "Verified result"
    with pytest.raises(LeaseConflict):
        store.finish(*args(lease), "failed")
    with pytest.raises(LeaseConflict):
        store.finish(*args(lease, subject="other"), "completed")
    with pytest.raises(LeaseConflict):
        store.heartbeat(*args(lease))


def test_explicit_uncertain_result_blocks_subsequent_agent_jobs(stored):
    store, _ = stored
    enqueue(store)
    enqueue(store, "m2")
    lease = claim(store)
    store.finish(*args(lease), "uncertain", summary="Cannot verify whether a write committed.")
    assert claim(store) is None


def test_operator_resolution_unblocks_next_job_without_replaying_expired_execution(stored):
    store, clock = stored
    enqueue(store)
    enqueue(store, "m2")
    lease = claim(store)
    clock.now = lease["expires_at"]
    assert claim(store) is None
    resolved = store.resolve("laptop", "m1", "completed", summary="Checked the application; the operation completed.")
    assert resolved["status"] == "completed" and resolved["resolved_at"] == clock.now
    assert resolved["summary"] == "Checked the application; the operation completed."
    next_job = claim(store)
    assert next_job["message_id"] == "m2"
    assert next_job["lease_token"] != lease["lease_token"]
    with pytest.raises(LeaseConflict):
        store.finish(*args(lease), "completed")
    with pytest.raises(LeaseConflict):
        store.heartbeat(*args(lease))
    with pytest.raises(LeaseConflict):
        store.publish(*args(lease), {"kind": "completed"})


@pytest.mark.parametrize("outcome", ["completed", "failed", "cancelled"])
def test_resolution_retry_retains_original_review_and_invalidates_unexpired_old_token(stored, outcome):
    store, _ = stored
    enqueue(store)
    lease = claim(store)
    store.finish(*args(lease), "uncertain")
    result = store.resolve("laptop", "m1", outcome, summary="Original review")
    assert store.resolve("laptop", "m1", outcome, summary="Changed review") == result
    assert result["summary"] == "Original review"
    with pytest.raises(LeaseConflict):
        store.finish(*args(lease), outcome)
    with pytest.raises(LeaseConflict):
        store.resolve("laptop", "m1", "cancelled" if outcome != "cancelled" else "failed")


@pytest.mark.parametrize("state", ["pending", "leased", "completed", "blocked", "cancelled", "failed"])
def test_operator_resolution_rejects_normal_task_states(stored, state):
    store, _ = stored
    enqueue(store)
    if state != "pending":
        lease = claim(store)
        if state != "leased":
            store.finish(*args(lease), state)
    with pytest.raises(LeaseConflict):
        store.resolve("laptop", "m1", "completed")
    assert store.get("laptop", "m1")["status"] == state
    with pytest.raises(LeaseConflict):
        store.resolve("other", "m1", "completed")
    with pytest.raises(LeaseConflict):
        store.resolve("laptop", "missing", "completed")


@pytest.mark.parametrize("outcome", ["pending", "leased", "blocked", "uncertain", "retry"])
def test_operator_resolution_never_requeues_or_selects_an_unreviewed_outcome(stored, outcome):
    store, _ = stored
    enqueue(store)
    lease = claim(store)
    store.finish(*args(lease), "uncertain")
    with pytest.raises(ValueError):
        store.resolve("laptop", "m1", outcome)
    assert store.get("laptop", "m1")["status"] == "uncertain" and claim(store) is None


def test_existing_queue_migrates_without_losing_messages_or_lease_ownership(tmp_path):
    path = tmp_path / "queue.sqlite"
    store = QueueStore(path)
    enqueue(store)
    lease = claim(store)
    store.close()
    with sqlite3.connect(path) as db:
        db.execute("ALTER TABLE messages DROP COLUMN resolved_at")
    store = QueueStore(path)
    try:
        assert store.get("laptop", "m1")["resolved_at"] is None
        assert claim(store) is None
        store.finish(*args(lease), "uncertain")
        assert store.resolve("laptop", "m1", "cancelled")["status"] == "cancelled"
    finally:
        store.close()


def test_task_events_are_bounded_and_cannot_attach_to_another_task(stored):
    store, _ = stored
    enqueue(store)
    lease = claim(store)
    for index in range(MAX_TASK_EVENTS + 5):
        store.publish(*args(lease), {"task_id": "task-m1", "kind": "model_completed", "elapsed_s": index})
    events = store.get("laptop", "m1")["events"]
    assert len(events) == MAX_TASK_EVENTS and events[0]["elapsed_s"] == 5
    with pytest.raises(LeaseConflict):
        store.publish(*args(lease), {"task_id": "different-task", "kind": "completed"})
    with pytest.raises(ValueError):
        store.publish(*args(lease), {"text": "x" * MAX_EVENT_BYTES})
    with pytest.raises(ValueError):
        store.publish(*args(lease), {"elapsed_s": float("nan")})
    assert store.get("laptop", "m1")["events"] == events


def test_terminal_job_does_not_accept_late_status_events(stored):
    store, _ = stored
    enqueue(store)
    lease = claim(store)
    store.finish(*args(lease), "blocked")
    with pytest.raises(LeaseConflict):
        store.publish(*args(lease), {"kind": "completed"})


def test_summary_limits_utf8_bytes_without_cutting_a_character(stored):
    store, _ = stored
    enqueue(store)
    lease = claim(store)
    store.finish(*args(lease), "completed", summary="停止" * 5000)
    summary = store.get("laptop", "m1")["summary"]
    assert len(summary.encode("utf-8")) <= MAX_SUMMARY_BYTES
    assert summary.encode("utf-8").decode("utf-8") == summary


def test_multiple_connections_atomically_deduplicate_and_claim_one_message(tmp_path):
    path = tmp_path / "queue.sqlite"
    stores = [QueueStore(path), QueueStore(path)]
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda store: enqueue(store), stores))
            assert sorted(result["status"] for result in results) == ["accepted", "duplicate"]
            claimed = list(pool.map(claim, stores))
        assert sum(lease is not None for lease in claimed) == 1
        assert len(stores[0].messages("laptop")) == 1
    finally:
        for store in stores:
            store.close()


def test_sql_metacharacters_remain_data(stored):
    store, _ = stored
    value = "'; DROP TABLE messages; --"
    assert store.enqueue(value, value, value, call(value, agent=value))["status"] == "accepted"
    assert store.claim(value, value)["call"]["agent_id"] == value
    assert store.messages("laptop") == []


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan"), True, "60"])
def test_lease_duration_is_bounded_and_finite(stored, value):
    store, _ = stored
    enqueue(store)
    with pytest.raises(ValueError):
        store.claim("laptop", "operator-one", lease_seconds=value)
    assert store.get("laptop", "m1")["status"] == "pending"


def test_database_rejects_directory_or_symlink(tmp_path):
    with pytest.raises(ValueError):
        QueueStore(tmp_path)
    if os.name == "posix":
        target = tmp_path / "target.sqlite"
        target.touch()
        link = tmp_path / "link.sqlite"
        link.symlink_to(target)
        with pytest.raises(ValueError):
            QueueStore(link)


@pytest.mark.skipif(os.name != "posix", reason="POSIX file permissions")
def test_new_database_directory_and_journal_files_are_private(tmp_path):
    parent = tmp_path / "private"
    store = QueueStore(parent / "queue.sqlite")
    try:
        enqueue(store)
        assert stat.S_IMODE(parent.stat().st_mode) == 0o700
        for path in parent.iterdir():
            assert stat.S_IMODE(path.stat().st_mode) == 0o600
    finally:
        store.close()


def test_close_is_idempotent_and_stops_store_use(stored):
    store, _ = stored
    store.close()
    store.close()
    with pytest.raises(RuntimeError):
        enqueue(store)


# -- D3: the queue keeps no excerpts, goals, summaries or event text once a task is over --------------------------

MARKER = "UNIQUE-MARKER-4f9c1d7a-do-not-store"


def database_bytes(store) -> bytes:
    """Everything on disk for this queue: the database and its write-ahead log."""
    data = b""
    for suffix in ("", "-wal", "-shm"):
        path = type(store.path)(str(store.path) + suffix)
        if path.exists():
            data += path.read_bytes()
    return data


def marked_call(identity="m1"):
    item = call(identity)
    item["goal"] = f"Do the thing about {MARKER}"
    item["context"] = {"request_context": {"topic": MARKER}}
    return item


def test_a_finished_task_leaves_no_goal_or_context_in_the_database_files(redacting):
    store, _ = redacting
    store.enqueue("standard", "e1", "digest", marked_call())
    lease = claim(store)
    assert MARKER in lease["call"]["goal"]  # the worker gets the real call while the task is live
    assert MARKER.encode() in database_bytes(store)
    store.finish(*args(lease), "completed", summary="A report quoting " + MARKER, note="Done.")
    stored_call = store.get("laptop", "m1")
    assert stored_call["status"] == "completed" and stored_call["summary"] == "Done."
    assert stored_call["call"]["goal"] == "redacted" and stored_call["call"]["context"] == {}
    assert stored_call["call"]["operation"] == "execute" and stored_call["call"]["source"] == "github"
    assert MARKER.encode() not in database_bytes(store)
    store.close()
    assert MARKER.encode() not in database_bytes(store)


def test_event_text_is_dropped_but_kind_and_outcome_are_kept(redacting):
    store, _ = redacting
    store.enqueue("standard", "e1", "digest", marked_call())
    lease = claim(store)
    store.publish(*args(lease), {"kind": "progress", "text": "step about " + MARKER, "outcome": "ok", "elapsed_s": 1.5})
    [event] = store.get("laptop", "m1")["events"]
    assert event == {"kind": "progress", "text": "", "outcome": "ok", "elapsed_s": 1.5}
    store.finish(*args(lease), "completed")
    store.close()
    assert MARKER.encode() not in database_bytes(store)


def test_an_uncertain_task_keeps_its_call_until_the_operator_resolves_it(redacting):
    store, _ = redacting
    store.enqueue("standard", "e1", "digest", marked_call())
    lease = claim(store)
    store.finish(*args(lease), "uncertain", note="Interrupted.")
    assert store.get("laptop", "m1")["call"]["goal"].endswith(MARKER)  # an operator needs to know what it was
    resolved = store.resolve("laptop", "m1", "failed", summary="notes mentioning " + MARKER)
    assert resolved["call"]["goal"] == "redacted" and resolved["summary"] == "Reconciled by an operator."
    store.close()
    assert MARKER.encode() not in database_bytes(store)


def test_recording_opt_in_keeps_everything_for_review(stored):
    store, _ = stored
    store.enqueue("standard", "e1", "digest", marked_call())
    lease = claim(store)
    store.publish(*args(lease), {"kind": "progress", "text": "step about " + MARKER})
    store.finish(*args(lease), "completed", summary="report " + MARKER, note="ignored when recording")
    item = store.get("laptop", "m1")
    assert item["call"]["goal"].endswith(MARKER) and item["summary"] == "report " + MARKER
    assert item["events"][0]["text"].endswith(MARKER)


def test_the_note_is_what_the_default_store_keeps_of_a_blocked_run(redacting):
    store, _ = redacting
    enqueue(store)
    lease = claim(store)
    store.finish(*args(lease), "blocked", summary="secret-looking " + MARKER, note="Desktop run not approved.")
    assert store.get("laptop", "m1")["summary"] == "Desktop run not approved."
    # a retried acknowledgement of the same outcome is still accepted
    assert store.finish(*args(lease), "blocked", note="Desktop run not approved.")["status"] == "blocked"


def test_secure_delete_is_on(redacting):
    store, _ = redacting
    assert store._db.execute("PRAGMA secure_delete").fetchone()[0] == 1


# -- characterisation: argument validation and the listing API ----------------------------------------------------------


@pytest.mark.parametrize(
    "limits", [{"max_pending": 0}, {"max_pending": True}, {"max_pending": "5"}, {"max_events": 0}, {"max_events": 1.5}]
)
def test_the_capacity_limits_must_be_positive_integers(tmp_path, limits):
    with pytest.raises(ValueError):
        QueueStore(tmp_path / "q.sqlite", **limits)


@pytest.mark.parametrize(
    "item",
    [
        "text",
        {"source": "s", "event_id": "e", "digest": "d"},  # no call key
        {"source": "s", "event_id": "e", "digest": "d", "call": None, "extra": 1},
        {"source": "s", "event_id": "e", "digest": "d", "call": None, "dedupe_body": "yes"},
        {"source": "", "event_id": "e", "digest": "d", "call": None},
        {"source": "s", "event_id": "e" * 4097, "digest": "d", "call": None},
        {"source": "s", "event_id": 7, "digest": "d", "call": None},
        {"source": "s", "event_id": "e", "digest": "d", "call": {"agent_id": "laptop"}},  # no message id
        {"source": "s", "event_id": "e", "digest": "d", "call": {"id": "m", "agent_id": ""}},
        {"source": "s", "event_id": "e", "digest": "d", "call": {"id": "m", "agent_id": "a", "n": float("nan")}},
        {"source": "s", "event_id": "e", "digest": "d", "call": {"id": "m", "agent_id": "a", "text": "x" * 70000}},
    ],
)
def test_a_malformed_delivery_is_refused_before_anything_is_stored(stored, item):
    store, _ = stored
    with pytest.raises(ValueError):
        store.enqueue_many([item])
    assert store.messages("laptop") == [] and store.enqueue_many([]) == []


def test_a_batch_must_be_a_list_of_at_most_one_thousand(stored):
    store, _ = stored
    with pytest.raises(ValueError):
        store.enqueue_many(tuple())
    with pytest.raises(ValueError):
        store.enqueue_many([{}] * 1001)


def test_listing_is_newest_first_filtered_and_bounded(stored):
    store, clock = stored
    for index, identity in enumerate(("m1", "m2", "m3")):
        clock.now = 100.0 + index
        enqueue(store, identity)
    assert [m["message_id"] for m in store.messages("laptop")] == ["m3", "m2", "m1"]
    assert [m["message_id"] for m in store.messages("laptop", limit=2)] == ["m3", "m2"]
    claim(store)
    assert [m["message_id"] for m in store.messages("laptop", status="leased")] == ["m1"]
    assert [m["message_id"] for m in store.messages("laptop", status="pending")] == ["m3", "m2"]
    assert store.messages("other") == []
    for bad in ({"limit": 0}, {"limit": 101}, {"limit": True}, {"limit": "5"}, {"status": "ignored"}, {"status": "DROP TABLE"}):
        with pytest.raises(ValueError):
            store.messages("laptop", **bad)


@pytest.mark.parametrize("subject", ["", None, 7])
def test_a_claim_needs_a_named_owner(stored, subject):
    store, _ = stored
    enqueue(store)
    with pytest.raises(LeaseConflict):
        store.claim("laptop", subject)
    assert store.get("laptop", "m1")["status"] == "pending"


@pytest.mark.parametrize("token", ["", None, 7, "t" * 1025])
def test_an_unusable_token_is_a_lease_conflict_everywhere(stored, token):
    store, _ = stored
    enqueue(store)
    lease = claim(store)
    for action in (
        lambda: store.heartbeat("laptop", "operator-one", "m1", token),
        lambda: store.finish("laptop", "operator-one", "m1", token, "completed"),
        lambda: store.publish("laptop", "operator-one", "m1", token, {"kind": "progress"}),
    ):
        with pytest.raises(LeaseConflict):
            action()
    assert store.get("laptop", "m1")["status"] == "leased" and lease["lease_token"]


def test_finish_resolve_and_publish_refuse_unusable_values(stored):
    store, _ = stored
    enqueue(store)
    lease = claim(store)
    for bad in (
        {"outcome": "pending"},
        {"outcome": "execute"},
        {"outcome": "completed", "summary": 7},
        {"outcome": "completed", "note": None},
    ):
        outcome = bad.pop("outcome")
        with pytest.raises(ValueError):
            store.finish(*args(lease), outcome, **bad)
    with pytest.raises(ValueError):
        store.publish(*args(lease), "not an event")
    with pytest.raises(ValueError):
        store.publish(*args(lease), {"kind": "progress", "text": "x" * (MAX_EVENT_BYTES + 1)})
    for outcome in ("uncertain", "pending", "ignored"):
        with pytest.raises(ValueError):
            store.resolve("laptop", "m1", outcome)
    with pytest.raises(ValueError):
        store.resolve("laptop", "m1", "failed", summary=7)
    assert store.get("laptop", "m1")["status"] == "leased"


@pytest.mark.parametrize("fixture,kept", [("stored", "Reviewed by the operator."), ("redacting", "Reconciled by an operator.")])
def test_a_resolution_keeps_its_summary_only_under_recording(request, fixture, kept):
    store, _ = request.getfixturevalue(fixture)
    enqueue(store)
    lease = claim(store)
    store.finish(*args(lease), "uncertain", summary="s", note="n")
    assert store.resolve("laptop", "m1", "failed", summary="Reviewed by the operator.")["summary"] == kept


def test_a_full_receipt_table_drops_old_receipts_of_finished_work_instead_of_refusing_forever(tmp_path):
    """PR7-4175614924: nothing ever deleted receipts, so after max_events every callback got 503."""
    clock = SimpleNamespace(now=1000.0)
    store = QueueStore(tmp_path / "q.sqlite", max_events=3, clock=lambda: clock.now, receipt_retention_s=100)
    try:
        enqueue(store, "done")
        store.finish(*args(claim(store)), "completed")
        enqueue(store, "blocked-on-operator")
        lease = claim(store)
        store.finish(*args(lease), "uncertain")  # an operator still has to look at this one
        store.enqueue("github", "ignored", "digest-ignored", None)  # a receipt with no message at all
        with pytest.raises(QueueFull):  # table full, nothing old enough yet
            enqueue(store, "new-1")
        clock.now += 101
        assert enqueue(store, "new-1")["status"] == "accepted"
        assert enqueue(store, "new-2")["status"] == "accepted"  # the ignored receipt went too
        with pytest.raises(QueueFull):  # the uncertain message's receipt is never dropped to make room
            enqueue(store, "new-3")
        assert enqueue(store, "blocked-on-operator")["status"] == "duplicate"
        with sqlite3.connect(tmp_path / "q.sqlite") as db:
            kept = {row[0] for row in db.execute("SELECT event_id FROM deliveries")}
        assert kept == {"blocked-on-operator", "new-1", "new-2"}  # the aged finished and ignored receipts are gone
    finally:
        store.close()


def test_a_blocked_redaction_checkpoint_is_retried_by_the_next_commit(tmp_path):
    """PR7-4175264726: a reader can make wal_checkpoint(TRUNCATE) return busy; the old text must still go."""
    path = tmp_path / "q.sqlite"
    marker = call()["goal"].encode()
    store = QueueStore(path)  # records no content: finishing redacts the goal
    reader = sqlite3.connect(path, isolation_level=None)
    try:
        store._db.execute("PRAGMA busy_timeout=0")  # do not wait for the reader
        enqueue(store)
        lease = claim(store)
        reader.execute("BEGIN")
        reader.execute("SELECT count(*) FROM messages").fetchall()  # holds a read snapshot of the WAL
        assert store.finish(*args(lease), "completed") == {"message_id": "m1", "status": "completed"}
        wal = path.with_name(path.name + "-wal")
        assert marker in wal.read_bytes()  # the truncate was blocked: the scenario of the finding
        store._db.execute("PRAGMA busy_timeout=5000")  # as in production: a retry must not wait this long for the reader
        started = time.monotonic()
        store.messages("laptop")  # a retry while the reader is still open does not make the operation wait for it
        assert time.monotonic() - started < 2 and marker in wal.read_bytes()
        reader.close()
        store.messages("laptop")  # any later commit pays the owed truncate
        assert marker not in (wal.read_bytes() if wal.exists() else b"")
    finally:
        reader.close()
        store.close()


# -- D3 exception, bounded: an uncertain row's content is deleted at the retention limit or at reconciliation --------

DAY = 86400


def uncertain_row(store, clock, identity="m1"):
    store.enqueue("standard", identity, "digest-" + identity, marked_call(identity))
    lease = claim(store)
    store.finish(*args(lease), "uncertain", note="Interrupted.")
    return lease


def test_the_default_retention_is_fourteen_days(tmp_path):
    store = QueueStore(tmp_path / "q.sqlite3")
    assert store.uncertain_retention_s == 14 * DAY
    store.close()


@pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf"), True, "7"])
def test_the_retention_must_be_a_positive_number(tmp_path, bad):
    with pytest.raises(ValueError):
        QueueStore(tmp_path / "q.sqlite3", uncertain_retention_s=bad)


def test_the_purge_waits_for_the_limit_and_fires_exactly_at_it(redacting):
    store, clock = redacting
    uncertain_row(store, clock)
    became_uncertain = clock.now
    clock.now = became_uncertain + 14 * DAY - 1
    assert store.purge_uncertain() == 0
    assert store.get("laptop", "m1")["call"]["goal"].endswith(MARKER)  # one second before the limit: still reviewable
    clock.now = became_uncertain + 14 * DAY  # the boundary counts as expired
    assert store.purge_uncertain() == 1
    row = store.get("laptop", "m1")
    assert row["call"]["goal"] == "redacted" and row["call"]["context"] == {} and row["call"]["id"] == "m1"
    assert row["status"] == "uncertain" and row["summary"] == "Expired, not reconciled." and row["expired_at"] == clock.now
    assert claim(store) is None  # it still blocks its agent until an operator decides
    assert store.purge_uncertain() == 0  # each row once


def test_the_purge_uses_the_injected_clock_and_the_configured_limit(tmp_path):
    clock = SimpleNamespace(now=1000.0)
    store = QueueStore(tmp_path / "q.sqlite3", clock=lambda: clock.now, uncertain_retention_s=3 * DAY)
    uncertain_row(store, clock)
    clock.now += 3 * DAY - 1
    assert store.purge_uncertain() == 0
    clock.now += 1
    assert store.purge_uncertain() == 1
    store.close()


def test_the_expired_content_is_gone_from_the_database_and_wal_bytes(redacting):
    store, clock = redacting
    uncertain_row(store, clock)
    assert MARKER.encode() in database_bytes(store)  # kept until the limit
    clock.now += 14 * DAY
    assert store.purge_uncertain() == 1
    store.close()
    assert MARKER.encode() not in database_bytes(store)


def test_the_purge_never_touches_pending_leased_or_finished_rows(redacting):
    store, clock = redacting
    store.enqueue("standard", "pending", "d1", marked_call("pending") | {"agent_id": "a-pending"})
    store.enqueue("standard", "leased", "d2", marked_call("leased") | {"agent_id": "a-leased"})
    store.claim("a-leased", "worker", lease_seconds=3600)
    store.enqueue("standard", "done", "d3", marked_call("done") | {"agent_id": "a-done"})
    lease = store.claim("a-done", "worker")
    store.finish("a-done", "worker", "done", lease["lease_token"], "blocked")
    clock.now += 80 * DAY  # far past the limit; the leased row's one-hour lease has also lapsed, but nobody touched it
    with sqlite3.connect(store.path) as raw:
        before = raw.execute("SELECT id, status, call_json FROM messages WHERE id IN ('pending','leased')").fetchall()
    assert store.purge_uncertain() == 0
    with sqlite3.connect(store.path) as raw:
        after = raw.execute("SELECT id, status, call_json FROM messages WHERE id IN ('pending','leased')").fetchall()
    assert before == after and all(MARKER in call_json for _, _, call_json in after)
    assert {status for _, status, _ in after} == {"pending", "leased"}


def test_recording_on_keeps_an_uncertain_rows_content_forever(stored):
    store, clock = stored
    uncertain_row(store, clock)
    clock.now += 80 * DAY
    assert store.purge_uncertain() == 0
    assert store.get("laptop", "m1")["call"]["goal"].endswith(MARKER)


@pytest.mark.parametrize(("verdict", "status"), [("done", "completed"), ("not-done", "cancelled"), ("unknown", "failed")])
def test_a_reconcile_writes_the_verdict_and_deletes_the_content_at_once(redacting, verdict, status):
    store, clock = redacting
    uncertain_row(store, clock)
    agent = store.agent_of("m1")
    assert agent == "laptop"
    result = store.reconcile(agent, "m1", verdict)
    assert result["status"] == status and result["resolved_at"] == clock.now
    assert result["summary"] == f"Reconciled by an operator: {verdict}."
    assert result["call"]["goal"] == "redacted" and result["call"]["context"] == {}
    store.close()
    assert MARKER.encode() not in database_bytes(store)


def test_a_reconcile_after_expiry_still_works_and_unblocks_the_agent(redacting):
    store, clock = redacting
    uncertain_row(store, clock)
    clock.now += 20 * DAY
    store.purge_uncertain()
    store.reconcile("laptop", "m1", "not-done")
    assert store.get("laptop", "m1")["status"] == "cancelled"
    store.enqueue("standard", "m2", "digest-m2", marked_call("m2"))
    assert claim(store) is not None


def test_reconcile_refuses_other_states_and_unknown_verdicts(redacting):
    store, _ = redacting
    store.enqueue("standard", "m1", "d", marked_call())
    with pytest.raises(LeaseConflict):
        store.reconcile("laptop", "m1", "done")  # pending
    with pytest.raises(ValueError):
        store.reconcile("laptop", "m1", "maybe")
    assert store.agent_of("nope") is None
    assert store.get("laptop", "m1")["status"] == "pending" and store.get("laptop", "m1")["call"]["goal"].endswith(MARKER)


def test_reconcile_with_recording_on_keeps_the_content(stored):
    store, clock = stored
    uncertain_row(store, clock)
    assert store.reconcile("laptop", "m1", "done")["call"]["goal"].endswith(MARKER)

"""ingest_run reads only the outcome label of run.json and never stores run content."""

import json
import sqlite3

import pytest

from glide.memory import Catalog, Harness, Scope, Store, ingest_run


@pytest.fixture
def harness(tmp_path):
    (tmp_path / "catalog").mkdir()
    store = Store(tmp_path / "state.sqlite")
    yield Harness(store, Catalog(tmp_path / "catalog"))
    store.close()


SCOPE = Scope("user", "project", "session")


def write_run(directory, **fields):
    directory.mkdir()
    (directory / "run.json").write_text(json.dumps({"outcome": "done", **fields}))
    return directory / "run.json"


def test_never_uses_model_success_and_deduplicates(tmp_path, harness):
    payload = {"goal_achieved": True, "answer": "secret content should not be stored"}
    one, two = write_run(tmp_path / "one", **payload), write_run(tmp_path / "two", **payload)
    first = ingest_run(harness, SCOPE, one)
    assert harness.store.outcomes(SCOPE) == []
    second = ingest_run(harness, SCOPE, two, verified_success=True, strategy_summary="Verified tactic")
    assert first == second and first.startswith("run:")
    ingest_run(harness, SCOPE, one, verified_success=True, strategy_summary="Verified tactic")
    assert len(harness.store.outcomes(SCOPE)) == 1
    stored = json.dumps([harness.store.events(SCOPE), harness.store.outcomes(SCOPE)])
    assert "secret content" not in stored and str(tmp_path) not in stored


def test_rejects_wrong_names_oversize_and_malformed_files(tmp_path, harness):
    (tmp_path / "other.json").write_text("{}")
    with pytest.raises(ValueError, match=r"run\.json"):
        ingest_run(harness, SCOPE, tmp_path / "other.json")
    with pytest.raises(ValueError, match=r"run\.json"):
        ingest_run(harness, SCOPE, tmp_path / "missing" / "run.json")
    big = tmp_path / "big"
    big.mkdir()
    (big / "run.json").write_text('{"outcome": "' + "x" * 1048577 + '"}')
    with pytest.raises(ValueError, match="1 MiB"):
        ingest_run(harness, SCOPE, big / "run.json")
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "run.json").write_text("not json")
    with pytest.raises(ValueError, match="valid JSON"):
        ingest_run(harness, SCOPE, bad / "run.json")
    (bad / "run.json").write_text("[]")
    with pytest.raises(ValueError, match="outcome"):
        ingest_run(harness, SCOPE, bad / "run.json")


def test_verified_flag_must_be_a_boolean_and_summaries_carry_no_urls(tmp_path, harness):
    run = write_run(tmp_path / "r")
    with pytest.raises(ValueError, match="boolean"):
        ingest_run(harness, SCOPE, run, verified_success="yes")
    with pytest.raises(ValueError, match="URL"):
        ingest_run(harness, SCOPE, run, verified_success=True, strategy_summary="went to https://example.com/inbox")
    assert harness.store.outcomes(SCOPE) == []


def test_store_error_types_are_value_errors_not_sqlite_errors(tmp_path, harness):
    run = write_run(tmp_path / "r")
    with pytest.raises(ValueError):
        ingest_run(harness, SCOPE, run, verified_success=True, strategy_summary="x" * 9000)
    assert not issubclass(ValueError, sqlite3.Error)

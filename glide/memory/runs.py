"""Learn from a finished screen-driving run without copying its content.

Only the outcome label of a completed `run.json` is read, and the run is identified by a hash of that
file, never by its path. Screenshots, transcripts, utterances, typed text and URLs are not read. A
writer's `achieved` flag is advisory and never becomes success evidence by itself: the caller passes
`verified_success` after an independent, fresh observation of the effect (AGENTS.md).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .contracts import Scope
from .harness import Harness

MAX_RUN_BYTES = 1048576


def ingest_run(
    harness: Harness,
    scope: Scope,
    path: Path,
    *,
    verified_success: bool | None = None,
    strategy_summary: str = "",
) -> str:
    """Record that a run was seen, or its host-verified outcome. Returns the run id (`run:` + sha256)."""
    if path.name != "run.json" or not path.is_file():
        raise ValueError("expected a regular run.json file")
    if path.stat().st_size > MAX_RUN_BYTES:
        raise ValueError("run summary exceeds 1 MiB")
    data = path.read_bytes()
    if len(data) > MAX_RUN_BYTES:
        raise ValueError("run summary exceeds 1 MiB")
    try:
        summary = json.loads(data)
    except ValueError:
        raise ValueError("run.json is not valid JSON") from None
    if not isinstance(summary, dict) or not isinstance(summary.get("outcome"), str):
        raise ValueError("expected a completed run.json with an outcome")
    run_id = "run:" + hashlib.sha256(data).hexdigest()
    if verified_success is None:
        harness.store.event(scope, "run_observed", {"run_id": run_id, "verified": False})
        return run_id
    if not isinstance(verified_success, bool):
        raise ValueError("verified_success must be a host-verified boolean")
    harness.record_outcome(scope, run_id, verified_success, strategy_summary)
    return run_id

"""One bounded, detached JSON round trip, shared by the harness (host callbacks) and the MCP bridge (wire data)."""

import json
from typing import Any


def bounded_json(value: Any, limit: int, error: type[ValueError], what: str) -> Any:
    """A private copy of `value` after a strict JSON round trip (no NaN), or `error(...)` if it is not JSON or too big."""
    try:
        encoded = json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError, OverflowError, RecursionError) as cause:
        raise error(f"{what} must contain bounded JSON values") from cause
    if len(encoded.encode()) > limit:
        raise error(f"{what} exceeds its byte budget")
    return json.loads(encoded)

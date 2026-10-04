"""Log a failure without what it carried: the type and where it was raised, never the exception's message.

`log.exception` prints the message, and a handler's exception can carry the typed request, a reply or a screen string.
"""

from __future__ import annotations

import logging
import sys
import traceback
from pathlib import Path

FRAMES = 3


def failed(log: logging.Logger, what: str, *args: object, error: BaseException | None = None) -> None:
    """`what % args`, then the exception's type and its innermost frames (file name, line, function)."""
    error = error or sys.exc_info()[1]
    where = ""
    if error is not None:
        frames = traceback.extract_tb(error.__traceback__)[-FRAMES:]
        where = ", ".join(f"{Path(f.filename).name}:{f.lineno} {f.name}" for f in frames)
    log.error("%s (%s at %s)", what % args if args else what, type(error).__name__ if error else "unknown", where or "?")

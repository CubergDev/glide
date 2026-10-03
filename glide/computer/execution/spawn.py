"""The one place an execution backend starts an external process.

Only the Playwright CLI bridge needs one (the user's own `playwright-cli`, driving a session the user opened).
Everything goes through `start`, called as `spawn.start(...)` so a test replaces it in one place.
`tests/guards_exec-backends.py` refuses it unless a test patches it, and `tests/test_exec_backends_guard.py`
fails if any other module of the execution package, or `browser_settings`, can start a process.

Nothing here launches a browser. Obscura and CDP are attach-only: the user starts them.
"""

from __future__ import annotations

import subprocess

TimeoutExpired = subprocess.TimeoutExpired  # so callers need not import subprocess themselves


def start(args: list[str], **options) -> subprocess.Popen:
    """Start `args` without a shell. `options` are Popen's (stdin, stdout, stderr, env)."""
    return subprocess.Popen(args, shell=False, **options)

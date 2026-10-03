"""Guard for the execution backends: the process seam refuses unless a test patches it.

`glide.computer.execution.spawn.start` is the only call in the execution package that starts a process (the
Playwright CLI bridge). `subprocess.Popen` is already refused by conftest; this refuses the seam itself, by name,
so a test that forgets to replace it fails with a message that says which seam to patch, and so a later change
that stops the seam from reaching Popen cannot unguard it.

The other machine-reaching names of these backends are covered elsewhere: `dom._get_json` (bound at import) by
`guards_core-run.py`, and the platform adapter calls `NativeBackend` makes by conftest and `guards_core-run.py`.
Nothing else in these modules reaches the machine without one of those calls.

A test that exercises the real body of the seam keeps a reference at import and patches `subprocess.Popen`.
"""

from __future__ import annotations

import pytest

from glide.computer.execution import spawn


@pytest.fixture(autouse=True)
def no_real_process_from_execution_backends(monkeypatch):
    def refused(*args, **kwargs):
        raise RuntimeError("a test reached the real machine through execution.spawn.start; patch the seam in the test")

    monkeypatch.setattr(spawn, "start", refused)

"""The seams where the platform adapters meet the rest of the machine: the shared escape hatch, the
AppleScript argv boundary, the osascript deadline and cancel, the dispatch gate, and loopback-only CDP.

Everything here runs offline against fakes. `tests/conftest.py` refuses the real functions in every test
(`macos.osascript`, `windows.open_url`, `cdp._get_json`, `websocket.create_connection`, `subprocess.Popen`),
so the real ones are captured at import time, before any guard runs, and each test fakes only the layer
below the one it exercises.
"""

import pytest

from glide.computer import macos, platform_adapter, windows
from glide.computer.models import Abort
from glide.computer.platform_adapter import using

ADAPTERS = pytest.mark.parametrize("adapter", [macos, windows], ids=["macos", "windows"])

# ------------------------------------------------------------------ the shared escape hatch (finding 12)


def test_the_escape_hatch_is_defined_once():
    assert macos.sleep_watching is windows.sleep_watching is platform_adapter.sleep_watching
    assert macos.abort_hint is windows.abort_hint is platform_adapter.abort_hint


@ADAPTERS
def test_check_abort_reads_the_adapters_pointer_when_it_is_called(monkeypatch, adapter):
    monkeypatch.setattr(adapter, "mouse_location", lambda: (0.0, 0.0))
    with pytest.raises(Abort, match="top-left corner"):
        adapter.check_abort()
    monkeypatch.setattr(adapter, "mouse_location", lambda: (900.0, 900.0))
    adapter.check_abort()


@ADAPTERS
def test_sleep_watching_asks_the_check_in_use_before_it_sleeps(monkeypatch, adapter):
    """`abort_on` replaces `check_abort` on the adapter; a wait must reach that replacement."""
    monkeypatch.setattr(platform_adapter.time, "sleep", lambda _: pytest.fail("slept past a stop"))

    def stopped():
        raise Abort("stopped")

    with using(adapter):
        monkeypatch.setattr(adapter, "check_abort", stopped)
        with pytest.raises(Abort, match="stopped"):
            adapter.sleep_watching(30)

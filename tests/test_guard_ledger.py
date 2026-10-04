"""The refusal ledger: a refusal that production code swallows still fails the test that fired it, at teardown.

The guard raises RuntimeError, and several production handlers catch `Exception` (`perception.ax_nodes`, `macos._ax_attr`,
`macos.ax_press`), so a test that forgot a mock could pass on whatever the fallback returned. `RefusalLedger` records every
refusal (`conftest.refusal`) and fails the test unless it declared `@pytest.mark.expect_refusals`.

The end-to-end checks run a nested pytest in this process with a fresh ledger as its plugin, so the failure this file asserts
is the real teardown error and not a model of it.
"""

from __future__ import annotations

import socket
import textwrap
from types import SimpleNamespace

import conftest
import pytest
from conftest import KNOWN_UNMOCKED, RefusalLedger, refusal

pytest_plugins = ["pytester"]

pytestmark = pytest.mark.expect_refusals


def item(*, marker: tuple | None = None, nodeid: str = "tests/test_x.py::test_y") -> SimpleNamespace:
    found = None if marker is None else SimpleNamespace(args=marker)
    return SimpleNamespace(nodeid=nodeid, get_closest_marker=lambda name: found if name == "expect_refusals" else None)


def ledger_with(*fired: str) -> RefusalLedger:
    ledger = RefusalLedger()
    ledger.fired = list(fired)
    return ledger


def test_a_refusal_records_itself_in_the_running_tests_ledger():
    mine = conftest._LEDGERS[-1]
    before = len(mine.fired)
    error = refusal("os.fork")
    assert isinstance(error, RuntimeError)
    assert "real machine" in str(error)
    assert mine.fired[before:] == ["os.fork"]


def test_every_guard_refusal_records_itself_through_the_same_ledger():
    """The conftest refusals, a table refusal, and the per-owner guards files all land in the one ledger."""
    mine = conftest._LEDGERS[-1]
    before = len(mine.fired)
    with pytest.raises(RuntimeError):
        socket.getaddrinfo("example.com", 80)  # a conftest closure
    with pytest.raises(RuntimeError):
        conftest._Refused("a table row")()
    from guards_voice import _refusal  # the guards files are loaded under two names; both must reach this ledger

    with pytest.raises(RuntimeError, match="fake"):
        _refusal("sounddevice.rec")()
    assert [what for what in mine.fired[before:]] == ["a lookup of 'example.com'", "a table row", "sounddevice.rec"]


def test_a_refusal_outside_every_test_is_kept_for_the_session_report(monkeypatch):
    outside: list[str] = []
    monkeypatch.setattr(conftest, "_LEDGERS", [])
    monkeypatch.setattr(conftest, "_OUTSIDE_A_TEST", outside)
    refusal("os.fork")
    assert outside == ["os.fork"]


def test_nothing_fired_outside_a_test_during_collection_or_session_setup():
    assert conftest._OUTSIDE_A_TEST == []


def test_without_a_marker_any_refusal_is_unexpected():
    assert ledger_with().unexpected(item()) is None
    message = ledger_with("os.fork").unexpected(item())
    assert message is not None
    assert "os.fork" in message
    assert "swallowed" in message


def test_the_marker_expects_every_refusal_or_only_the_named_ones():
    assert ledger_with("os.fork").unexpected(item(marker=())) is None
    assert ledger_with("os.fork").unexpected(item(marker=("os.fork",))) is None
    assert ledger_with("os.fork").unexpected(item(marker=("fork",))) is None
    message = ledger_with("os.fork", "subprocess.Popen").unexpected(item(marker=("os.fork",)))
    assert message is not None
    assert "subprocess.Popen" in message
    assert "os.fork" not in message


def test_a_known_unmocked_test_is_excused_by_its_node_id_and_no_other(monkeypatch):
    monkeypatch.setitem(KNOWN_UNMOCKED, "tests/test_x.py::test_y", "the test lacks a Quartz mock")
    assert ledger_with("Quartz.CGDisplayBounds").unexpected(item()) is None
    assert ledger_with("Quartz.CGDisplayBounds").unexpected(item(nodeid="tests/test_x.py::test_other")) is not None


def test_every_known_unmocked_entry_says_why():
    for nodeid, reason in KNOWN_UNMOCKED.items():
        assert "::" in nodeid
        assert len(reason) > 20, f"{nodeid} is excused without a reason"


# ------------------------------------------------------------------ the real teardown error, in a nested run

NESTED = textwrap.dedent(
    """
    import pytest
    from conftest import refusal

    def swallowed():
        try:
            raise refusal("the accessibility tree")
        except Exception:
            return []  # what perception.ax_nodes does

    def test_passes_on_a_fallback_after_a_swallowed_refusal():
        assert swallowed() == []

    @pytest.mark.expect_refusals
    def test_a_guard_test_expects_it():
        with pytest.raises(RuntimeError):
            raise refusal("os.fork")

    @pytest.mark.expect_refusals("os.fork")
    def test_expects_one_kind_but_another_fires():
        swallowed()

    def test_quiet():
        assert True
    """
)


def run_nested(pytester):
    pytester.makepyfile(test_nested=NESTED)
    return pytester.runpytest_inprocess("-p", "no:cacheprovider", plugins=[RefusalLedger()])


def test_a_swallowed_refusal_fails_the_test_at_teardown(pytester):
    result = run_nested(pytester)
    result.assert_outcomes(passed=4, errors=2)  # the call passes; its teardown reports the ledger
    result.stdout.fnmatch_lines(
        [
            "*ERROR at teardown of test_passes_on_a_fallback_after_a_swallowed_refusal*",
            "*a refusal fired during this test that it did not expect (the accessibility tree)*",
            "*ERROR at teardown of test_expects_one_kind_but_another_fires*",
        ]
    )


def test_a_test_that_expects_the_refusal_and_a_quiet_test_are_not_failed(pytester):
    result = run_nested(pytester)
    output = result.stdout.str()
    assert "ERROR at teardown of test_a_guard_test_expects_it" not in output
    assert "ERROR at teardown of test_quiet" not in output

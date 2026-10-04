"""Mutation-testing follow-up for `glide/providers/chain.py` (see docs/MUTATION_REPORT.md).

Each test was written against a mutant of the chain that the existing suite let survive, and fails on that mutant. They
pin the parts of failover that safety depends on: a call is never raced unless the caller says it is safe to run twice,
a cancel is a cancel (never a provider fault, never failed over, never a switch), every switch is a visible
`SwitchEvent`, and a provider's health is kept the same way for streams, calls and races.

Everything is a fake. A thread that has to wait waits on an Event; `WAIT` is only the longest it will wait before
deciding the thing is not coming. The two tests that need a timer to expire say so.
"""

from __future__ import annotations

import threading

import pytest

from glide.computer.control import RunControl, controlled
from glide.providers.chain import Chain, ChainPolicy, Slot
from glide.providers.errors import CANCELLED, AllProvidersFailed, ProviderError, cancelled

WAIT = 3.0


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def err(kind: str = "server", **kw) -> ProviderError:
    return ProviderError(f"{kind} failure", kind=kind, **kw)


class Fake:
    """A provider that answers from a script: a value to return or an exception to raise, in turn."""

    def __init__(self, *script):
        self.script = list(script)
        self.calls = 0

    def __call__(self):
        self.calls += 1
        step = self.script[min(self.calls - 1, len(self.script) - 1)]
        if isinstance(step, BaseException):
            raise step
        return step


def chain(*fakes, policy=None, clock=None, on_event=None):
    slots = [Slot("abcdefgh"[i], f) for i, f in enumerate(fakes)]
    return Chain("llm", slots, policy, clock=clock or Clock(), on_event=on_event)


def run(c, **kw):
    return c.call(lambda slot: slot.client(), **kw)


# -- a call is only raced when the caller says it is safe to run twice ---------------------------------------------


def test_a_call_is_not_raced_unless_the_caller_asks_for_it(monkeypatch):
    """`hedge` defaults to off: a call that was not marked safe to run twice must never start a second provider."""

    def boom(*args, **kwargs):
        raise AssertionError("raced a call that was not marked safe to run twice")

    monkeypatch.setattr(Chain, "_hedged", boom)
    second = Fake("B")
    c = chain(Fake("A"), second, policy=ChainPolicy(hedge_after_s=0.01))
    assert run(c) == "A"
    assert second.calls == 0


@pytest.mark.parametrize("kind", ["bad_request", CANCELLED])
def test_a_hedged_call_raises_a_non_failover_error_at_once_and_does_not_try_the_next_provider(kind):
    second = Fake("B")
    first = Fake(cancelled("a", "stopped by the user") if kind == CANCELLED else err(kind))
    c = chain(first, second, policy=ChainPolicy(hedge_after_s=5))
    with pytest.raises(ProviderError) as caught:
        run(c, hedge=True)
    assert caught.value.kind == kind
    assert second.calls == 0
    assert not c.events  # and it is not a switch


def test_a_hedged_failover_is_a_visible_switch():
    seen = []
    c = chain(Fake(err("server")), Fake("B"), policy=ChainPolicy(hedge_after_s=5), on_event=seen.append)
    assert run(c, hedge=True) == "B"
    assert [(e.from_slot, e.to_slot, e.kind) for e in c.events] == [("a", "b", "server")]
    assert seen == list(c.events)


def test_a_cancel_of_a_hedged_call_reports_the_callers_reason_and_starts_nothing_more():
    started, release = threading.Event(), threading.Event()
    second = Fake("B")

    def first():
        started.set()
        release.wait(WAIT)
        return "late"

    parent = RunControl("t")
    c = chain(first, second, policy=ChainPolicy(hedge_after_s=5))
    threading.Thread(target=lambda: (started.wait(WAIT), parent.cancel("stopped by the user")), daemon=True).start()
    try:
        with controlled(parent), pytest.raises(ProviderError) as caught:
            run(c, hedge=True)
    finally:
        release.set()
    assert caught.value.kind == CANCELLED and "stopped by the user" in str(caught.value)
    assert second.calls == 0


def test_a_cancel_that_lands_between_two_hedged_launches_is_a_cancel_with_its_reason():
    """The switch listener runs on the calling thread just before the second racer starts: cancel there."""
    release = threading.Event()
    second = Fake("B")
    parent = RunControl("t")

    def listener(event):
        parent.cancel("stopped by the user")

    c = chain(lambda: release.wait(WAIT), second, policy=ChainPolicy(hedge_after_s=0.02), on_event=listener)
    try:
        with controlled(parent), pytest.raises(ProviderError) as caught:
            run(c, hedge=True)
    finally:
        release.set()
    assert caught.value.kind == CANCELLED and "stopped by the user" in str(caught.value)
    assert second.calls == 0


def test_once_every_provider_is_racing_the_call_waits_for_an_answer_instead_of_timing_out():
    """Needs one real timer: the second racer takes 0.15 s, many times the 0.01 s hedge delay."""
    release = threading.Event()

    def slow_second():
        threading.Event().wait(0.15)
        return "B"

    c = chain(lambda: release.wait(WAIT), slow_second, policy=ChainPolicy(hedge_after_s=0.01))
    try:
        assert run(c, hedge=True) == "B"
    finally:
        release.set()
    assert [(e.from_slot, e.to_slot, e.kind) for e in c.events] == [("a", "b", "slow")]


# -- a cancel is not a provider's fault -----------------------------------------------------------------------------


def test_whatever_a_provider_raises_once_the_control_is_cancelled_is_the_cancel_and_not_a_fault():
    c = chain(Fake("unused"))
    slot = c._slots[0]
    control = RunControl("t")
    control.cancel("stopped by the user")

    def closed_connection(_slot):
        raise err("transport")  # what a connection closed on purpose reads as

    with pytest.raises(ProviderError) as caught:
        c._attempt(closed_connection, slot, control, hop=False)
    assert caught.value.kind == CANCELLED and "stopped by the user" in str(caught.value)
    assert c.status()[0]["failures"] == 0 and not c.events


def test_a_cancel_the_provider_raised_itself_is_kept_as_it_is():
    c = chain(Fake("unused"))
    control = RunControl("t")
    control.cancel("stopped by the user")

    def own(_slot):
        raise cancelled("a", "the provider's own reason")

    with pytest.raises(ProviderError) as caught:
        c._attempt(own, c._slots[0], control, hop=False)
    assert "the provider's own reason" in str(caught.value)


def test_a_fault_raised_while_the_control_is_still_running_is_a_fault():
    c = chain(Fake("unused"))

    def fails(_slot):
        raise err("server")

    with pytest.raises(ProviderError) as caught:
        c._attempt(fails, c._slots[0], RunControl("t"), hop=False)
    assert caught.value.kind == "server"
    assert c.status()[0]["failures"] == 1


def test_a_generator_that_a_cancel_abandoned_is_closed_when_its_blocked_step_returns():
    """The cancel releases the caller at once; the step that was blocked finishes later, on its own thread, and
    closes the generator there. The test holds a reference so that only an explicit close can end it."""
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()
    kept = []

    def items():
        try:
            entered.set()
            release.wait(WAIT)
            yield "late"
        finally:
            closed.set()

    def fn(_slot):
        generator = items()
        kept.append(generator)
        return generator

    control = RunControl("t")
    c = Chain("llm", [Slot("a", object())], clock=Clock())
    with controlled(control):
        stream = c.stream(fn)
    threading.Thread(target=lambda: (entered.wait(WAIT), control.cancel("stopped by the user")), daemon=True).start()
    with pytest.raises(ProviderError) as caught:
        next(stream)
    assert caught.value.kind == CANCELLED
    assert not closed.is_set()  # still blocked inside its step
    release.set()
    assert closed.wait(WAIT), "the abandoned generator was never closed"
    assert c.status()[0]["failures"] == 0 and not c.events


# -- streams keep a provider's health as calls do -------------------------------------------------------------------


def test_a_stream_records_its_latency_to_the_first_item_and_one_call():
    clock = Clock()

    def slow_first():
        clock.advance(2.0)
        yield "x"
        clock.advance(10.0)  # later items are not the provider's latency
        yield "y"

    c = Chain("llm", [Slot("a", slow_first)], clock=clock)
    assert list(c.stream(lambda s: s.client())) == ["x", "y"]
    row = c.status()[0]
    assert (row["calls"], row["avg_latency_s"], row["failures"]) == (1, 2.0, 0)


def test_an_empty_stream_is_one_timed_success():
    clock = Clock()

    def empty():
        clock.advance(1.5)
        return
        yield  # pragma: no cover

    c = Chain("llm", [Slot("a", empty)], clock=clock)
    assert list(c.stream(lambda s: s.client())) == []
    row = c.status()[0]
    assert (row["calls"], row["avg_latency_s"]) == (1, 1.5)


def test_a_stream_that_fails_before_its_first_item_counts_against_that_provider():
    def dead():
        raise err("server")
        yield  # pragma: no cover

    def good():
        yield "x"

    c = Chain("llm", [Slot("a", dead), Slot("b", good)], ChainPolicy(fail_threshold=1, cooldown_s=30), clock=Clock())
    assert list(c.stream(lambda s: s.client())) == ["x"]
    row = c.status()[0]
    assert (row["failures"], row["resting_s"]) == (1, 30.0)
    assert row["last_error"] == "server: server failure"
    assert [(e.from_slot, e.to_slot, e.kind) for e in c.events] == [("a", "b", "server")]


def test_a_stream_that_dies_midway_counts_against_its_provider_and_names_it():
    def breaks():
        yield "x"
        raise err("transport", status=502)

    c = Chain("llm", [Slot("a", breaks)], ChainPolicy(fail_threshold=1, cooldown_s=30), clock=Clock())
    with pytest.raises(ProviderError) as caught:
        list(c.stream(lambda s: s.client()))
    assert (caught.value.kind, caught.value.provider, caught.value.status) == ("stream", "a", 502)
    row = c.status()[0]
    assert (row["failures"], row["resting_s"]) == (1, 30.0)


def test_a_stream_error_keeps_the_provider_an_adapter_named():
    def breaks():
        yield "x"
        raise err("transport", provider="adapter-name")

    c = Chain("llm", [Slot("a", breaks)], clock=Clock())
    with pytest.raises(ProviderError) as caught:
        list(c.stream(lambda s: s.client()))
    assert caught.value.provider == "adapter-name"


def test_a_consumer_that_stops_early_gives_the_connection_back():
    closed = threading.Event()
    kept = []

    def items():
        try:
            yield "x"
            yield "y"
        finally:
            closed.set()

    def fn(_slot):
        generator = items()
        kept.append(generator)  # a reference, so that only an explicit close can end it
        return generator

    c = Chain("llm", [Slot("a", object())], clock=Clock())
    stream = c.stream(fn)
    assert next(stream) == "x"
    stream.close()
    assert closed.is_set()


# -- health and order ------------------------------------------------------------------------------------------------


def test_a_success_between_two_failures_keeps_the_provider_in_service():
    """The threshold counts consecutive failures: a success starts the count again."""
    clock = Clock()
    c = chain(Fake(err(), "A", err()), Fake("B"), policy=ChainPolicy(fail_threshold=2, cooldown_s=30), clock=clock)
    assert run(c) == "B"  # a fails
    assert run(c) == "A"  # a answers
    assert run(c) == "B"  # a fails again, but not twice in a row
    row = c.status()[0]
    assert (row["failures"], row["resting_s"], row["calls"]) == (1, 0.0, 1)


def test_status_says_what_it_knows_about_each_provider():
    clock = Clock()
    c = chain(Fake(err("server")), Fake("B"), clock=clock)
    assert [(r["name"], r["pinned"], r["resting_s"], r["failures"], r["calls"], r["last_error"]) for r in c.status()] == [
        ("a", False, 0.0, 0, 0, None),
        ("b", False, 0.0, 0, 0, None),
    ]
    c.pin("b")
    assert [r["pinned"] for r in c.status()] == [False, True]
    assert run(c) == "B"
    rows = c.status()
    assert (rows[1]["calls"], rows[1]["last_error"]) == (1, None)
    c = chain(Fake(err("server")), Fake("B"), clock=clock)
    run(c)  # a failing provider's last error is kept for display
    assert c.status()[0]["last_error"] == "server: server failure"


def test_the_remaining_rest_is_shown_to_a_tenth_of_a_second():
    c = chain(Fake(err("rate_limit", retry_after=5.26)), Fake("B"), clock=Clock())
    run(c)
    assert c.status()[0]["resting_s"] == 5.3


def test_a_slot_is_ready_again_at_the_moment_its_rest_ends_and_listed_once():
    clock = Clock()
    a = Fake(err(), err(), "A")
    c = chain(a, Fake("B"), policy=ChainPolicy(fail_threshold=2, cooldown_s=30), clock=clock)
    run(c), run(c)  # a rests until 1030
    clock.advance(30)
    assert run(c) == "A" and a.calls == 3


def test_a_provider_whose_rest_ends_now_is_tried_once_when_everything_fails():
    clock = Clock()
    a, b = Fake(err()), Fake(err())
    c = chain(a, b, policy=ChainPolicy(fail_threshold=1, cooldown_s=30), clock=clock)
    with pytest.raises(AllProvidersFailed):
        run(c)  # both rest until 1030
    clock.advance(30)
    with pytest.raises(AllProvidersFailed) as caught:
        run(c)
    assert (a.calls, b.calls) == (2, 2)
    assert [name for name, _ in caught.value.errors] == ["a", "b"]


def test_latency_order_times_an_unmeasured_provider_before_trusting_a_measured_fast_one():
    clock = Clock()

    def timed(seconds, value):
        def call():
            clock.advance(seconds)
            return value

        return call

    c = chain(timed(0.5, "a"), timed(0.1, "b"), policy=ChainPolicy(order="latency"), clock=clock)
    assert run(c) == "a"  # a is timed: 0.5 s
    assert run(c) == "b"  # b has no measurement yet, so it goes first even though a is quick


def test_a_pinned_provider_is_tried_first_even_when_every_provider_is_resting():
    clock = Clock()
    a, b = Fake(err("auth"), "A"), Fake(err("auth"), "B")
    c = chain(a, b, clock=clock)
    with pytest.raises(AllProvidersFailed):
        run(c)  # both rest for the auth cooldown, until the same moment
    c.pin("b")
    assert run(c) == "B"
    assert a.calls == 1

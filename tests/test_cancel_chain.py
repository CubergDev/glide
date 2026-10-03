"""A cancel reaches a provider call through the chain: the caller is released, the connection is closed, and a
cancel is never a provider's fault (D2).

Everything here is a fake that blocks on an Event. Nothing sleeps to give a thread time: a test waits for the
thing it needs (an Event, a join) and `WAIT` is only the longest it will wait before deciding it is not coming.
"""

from __future__ import annotations

import threading

import pytest

from glide.computer.control import RunControl, controlled, current_control, spawn
from glide.computer.models import Abort
from glide.providers import interrupt
from glide.providers.chain import Chain, ChainPolicy, Slot
from glide.providers.errors import CANCELLED, FAILOVER_KINDS, ProviderError, cancelled

WAIT = 3.0


class Clock:
    now = 1000.0

    def __call__(self) -> float:
        return self.now


class Connection:
    """What a provider holds open while it waits: `close()` is what a cancel is supposed to call."""

    def __init__(self, *, wakes: bool = True) -> None:
        self.closed = threading.Event()
        self.opened = threading.Event()
        self.release = threading.Event()
        self.wakes = wakes  # whether closing it ends the wait, as shutting a real socket down does

    def close(self) -> None:
        self.closed.set()
        if self.wakes:
            self.release.set()

    def wait(self) -> None:
        self.opened.set()
        self.release.wait(WAIT)


class Blocked:
    """A provider call that holds a connection and waits, and records how many times it was started."""

    def __init__(self, *, wakes: bool = True, then: Exception | str = "late answer") -> None:
        self.connection = Connection(wakes=wakes)
        self.calls = 0
        self.then = then
        self.control_seen: RunControl | None = None

    def __call__(self):
        self.calls += 1
        self.control_seen = current_control()
        with interrupt.closing(self.connection.close, "blocked"):
            self.connection.wait()
        if isinstance(self.then, Exception):
            raise self.then
        return self.then


def in_thread(call):
    done, box = threading.Event(), {}

    def target():
        try:
            box["value"] = call()
        except BaseException as exc:
            box["error"] = exc
        finally:
            done.set()

    threading.Thread(target=target, daemon=True).start()
    return done, box


def chain_of(*clients, policy=None, events=None):
    slots = [Slot("abcdef"[i], client) for i, client in enumerate(clients)]
    return Chain("llm", slots, policy, clock=Clock(), on_event=events.append if events is not None else None)


def call_under(control, chain, **kw):
    def run():
        with controlled(control):
            return chain.call(lambda slot: slot.client(), **kw)

    return in_thread(run)


def test_the_kind_is_not_a_failover_kind_and_the_message_carries_nothing_but_a_reason():
    assert CANCELLED not in FAILOVER_KINDS
    error = cancelled("openrouter:m", "stopped by the user")
    assert error.kind == CANCELLED and error.provider == "openrouter:m"
    assert str(error) == "openrouter:m call cancelled (stopped by the user)"


def test_a_cancel_closes_the_connection_and_releases_the_caller_without_a_retry_or_a_switch():
    a, b, events = Blocked(), Blocked(), []
    chain = chain_of(a, b, events=events)
    control = RunControl()
    done, box = call_under(control, chain)
    assert a.connection.opened.wait(WAIT)
    control.cancel("stopped by the user")
    assert done.wait(WAIT), "the caller is still blocked after the cancel"
    assert a.connection.closed.is_set()
    assert box["error"].kind == CANCELLED and "stopped by the user" in str(box["error"])
    assert (a.calls, b.calls) == (1, 0)  # not retried, not failed over
    assert list(chain.events) == [] and events == []  # and not a SwitchEvent
    assert chain.status()[0]["failures"] == 0 and chain.status()[0]["last_error"] is None


def test_the_caller_is_released_even_when_the_provider_cannot_be_woken():
    """A request still waiting for its first byte has nothing to close: the thread is left to end on its own."""
    a = Blocked(wakes=False)
    control = RunControl()
    done, box = call_under(control, chain_of(a))
    assert a.connection.opened.wait(WAIT)
    control.cancel()
    assert done.wait(WAIT)
    assert box["error"].kind == CANCELLED
    assert not a.connection.release.is_set()  # still blocked in its own thread; nobody is waiting for it
    a.connection.release.set()


def test_an_answer_that_arrives_after_the_cancel_is_dropped():
    a = Blocked(then="late answer")
    control = RunControl()
    done, box = call_under(control, chain_of(a))
    assert a.connection.opened.wait(WAIT)
    control.cancel()
    a.connection.release.set()
    assert done.wait(WAIT)
    assert "value" not in box and box["error"].kind == CANCELLED


def test_what_a_closed_connection_raises_is_the_cancel_and_not_a_transport_fault_to_fail_over_from():
    a = Blocked(then=ProviderError("a could not be reached", kind="transport", provider="a"))
    b, events = Blocked(), []
    chain = chain_of(a, b, events=events)
    control = RunControl()
    done, box = call_under(control, chain)
    assert a.connection.opened.wait(WAIT)
    control.cancel()
    assert done.wait(WAIT)
    assert box["error"].kind == CANCELLED
    assert b.calls == 0 and events == [] and chain.status()[0]["failures"] == 0


def test_a_cancel_before_the_call_calls_nobody():
    a = Blocked()
    control = RunControl()
    control.cancel("stopped by the user")
    done, box = call_under(control, chain_of(a))
    assert done.wait(WAIT)
    assert box["error"].kind == CANCELLED and a.calls == 0


def test_the_provider_runs_under_the_control_of_the_caller():
    seen = []

    def client():
        seen.append(current_control())
        return "ok"

    control = RunControl()
    with controlled(control):
        assert chain_of(client).call(lambda slot: slot.client()) == "ok"
    assert seen == [control]


def test_without_a_control_nothing_changes_and_no_thread_is_started():
    caller = threading.get_ident()
    ran_on = []
    chain = chain_of(lambda: ran_on.append(threading.get_ident()) or "ok")
    assert chain.call(lambda slot: slot.client()) == "ok" and ran_on == [caller]


def test_a_cancel_while_streaming_is_raised_as_cancelled_and_never_as_a_stream_failure():
    gate, connection, closed_stream = threading.Event(), Connection(), threading.Event()

    def stream():
        try:
            with interrupt.closing(connection.close, "a"):
                yield "one"
                gate.set()
                connection.wait()
                yield "never"
        finally:
            closed_stream.set()

    chain = chain_of(stream)
    control = RunControl()
    seen = []

    def consume():
        with controlled(control):
            for item in chain.stream(lambda slot: slot.client()):
                seen.append(item)

    done, box = in_thread(consume)
    assert gate.wait(WAIT) and connection.opened.wait(WAIT)
    control.cancel("stopped by the user")
    assert done.wait(WAIT)
    assert seen == ["one"] and box["error"].kind == CANCELLED
    assert connection.closed.is_set() and closed_stream.wait(WAIT)
    assert list(chain.events) == [] and chain.status()[0]["failures"] == 0


def test_a_stream_the_consumer_abandons_is_closed_and_the_control_is_the_one_current_at_the_call():
    closed, control = threading.Event(), RunControl()
    seen = []

    def stream():
        seen.append(current_control())
        try:
            yield from ("x", "y", "z")
        finally:
            closed.set()

    chain = chain_of(stream)
    with controlled(control):
        items = chain.stream(lambda slot: slot.client())  # called here, first item asked for outside the block
    assert next(items) == "x" and seen == [control]
    items.close()
    assert closed.is_set()


def test_a_cancel_before_the_first_item_ends_the_stream_without_a_failover():
    a, b = Blocked(), Blocked()
    events = []
    chain = chain_of(a, b, events=events)
    control = RunControl()

    def consume():
        with controlled(control):
            return list(chain.stream(lambda slot: iter([slot.client()])))

    done, box = in_thread(consume)
    assert a.connection.opened.wait(WAIT)
    control.cancel()
    assert done.wait(WAIT)
    assert box["error"].kind == CANCELLED and b.calls == 0 and events == []


# -- hedging ----------------------------------------------------------------------------------------


def hedged(*clients, events=None):
    return chain_of(*clients, policy=ChainPolicy(hedge_after_s=0.01), events=events)


def test_the_hedged_loser_is_cancelled_the_same_way_and_is_not_a_failure():
    slow, events = Blocked(wakes=True), []
    chain = hedged(slow, lambda: "fast", events=events)
    assert chain.call(lambda slot: slot.client(), hedge=True) == "fast"
    assert slow.connection.closed.wait(WAIT), "the slower request was left running"
    assert [(e.from_slot, e.to_slot, e.kind) for e in chain.events] == [("a", "b", "slow")]  # the race, and nothing else
    assert chain.status()[0]["failures"] == 0 and chain.status()[0]["last_error"] is None


def test_the_loser_sees_its_own_control_and_the_winner_is_not_cancelled():
    slow = Blocked()
    seen = []

    def fast():
        seen.append(current_control())
        return "fast"

    chain = hedged(slow, fast)
    parent = RunControl()
    with controlled(parent):
        assert chain.call(lambda slot: slot.client(), hedge=True) == "fast"
    assert slow.connection.closed.wait(WAIT)
    assert slow.control_seen.cancelled.is_set() and not seen[0].cancelled.is_set()
    assert slow.control_seen is not parent and not parent.cancelled.is_set()  # the user's control is untouched


def test_a_cancel_reaches_every_racer_and_launches_nobody_else():
    a, b, c = Blocked(), Blocked(), Blocked()
    events = []
    chain = hedged(a, b, c, events=events)
    control = RunControl()
    done, box = call_under(control, chain, hedge=True)
    assert a.connection.opened.wait(WAIT) and b.connection.opened.wait(WAIT)
    control.cancel("stopped by the user")
    assert done.wait(WAIT)
    assert box["error"].kind == CANCELLED
    assert a.connection.closed.wait(WAIT) and b.connection.closed.wait(WAIT)
    assert not any(e.kind != "slow" for e in chain.events) and c.calls <= 1
    assert all(status["failures"] == 0 for status in chain.status())


def test_a_hedged_call_made_after_the_cancel_starts_nothing():
    a, b = Blocked(), Blocked()
    control = RunControl()
    control.cancel()
    done, box = call_under(control, hedged(a, b), hedge=True)
    assert done.wait(WAIT) and box["error"].kind == CANCELLED and (a.calls, b.calls) == (0, 0)


# -- the control's own helpers ------------------------------------------------------------------------


def test_interruptible_returns_the_value_and_reraises_what_the_call_raised():
    control = RunControl()
    assert control.interruptible(lambda: 7) == 7
    with pytest.raises(KeyError):
        control.interruptible(lambda: {}["missing"])


def test_interruptible_raises_abort_with_the_reason_the_moment_the_control_is_cancelled():
    control, started, release = RunControl(), threading.Event(), threading.Event()
    done, box = in_thread(lambda: control.interruptible(lambda: (started.set(), release.wait(WAIT))))
    assert started.wait(WAIT)
    control.cancel("stopped by the user")
    assert done.wait(WAIT)
    assert isinstance(box["error"], Abort) and str(box["error"]) == "stopped by the user"
    release.set()


def test_spawn_carries_the_context_of_the_caller_and_never_blocks_the_end_of_the_process():
    control = RunControl()
    with controlled(control):
        future = spawn(lambda: (current_control(), threading.current_thread().daemon))
    assert future.result(WAIT) == (control, True)

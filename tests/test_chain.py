"""Provider failover: order, cooldown, pinning, streaming, and hedging. No network, no real clock."""

from __future__ import annotations

import threading

import pytest

from glide.providers.chain import Chain, ChainPolicy, Slot
from glide.providers.errors import AllProvidersFailed, ProviderError, from_status


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
    names = "abcdefgh"
    slots = [Slot(names[i], f) for i, f in enumerate(fakes)]
    return Chain("llm", slots, policy, clock=clock or Clock(), on_event=on_event)


def run(c, **kw):
    return c.call(lambda slot: slot.client(), **kw)


def test_the_first_provider_answers():
    a, b = Fake("A"), Fake("B")
    assert run(chain(a, b)) == "A"
    assert (a.calls, b.calls) == (1, 0)


def test_a_failure_moves_to_the_next_provider_and_says_so():
    seen = []
    c = chain(Fake(err("rate_limit")), Fake("B"), on_event=seen.append)
    assert run(c) == "B"
    assert [(e.from_slot, e.to_slot, e.kind) for e in c.events] == [("a", "b", "rate_limit")]
    assert seen == list(c.events)


def test_our_own_bad_request_is_raised_and_not_tried_elsewhere():
    b = Fake("B")
    c = chain(Fake(err("bad_request")), b)
    with pytest.raises(ProviderError) as caught:
        run(c)
    assert caught.value.kind == "bad_request"
    assert b.calls == 0
    assert c.status()[0]["failures"] == 0  # and it says nothing about the provider's health


def test_every_provider_failing_reports_each_error_in_order():
    c = chain(Fake(err("timeout")), Fake(err("auth")))
    with pytest.raises(AllProvidersFailed) as caught:
        run(c)
    assert [(name, e.kind) for name, e in caught.value.errors] == [("a", "timeout"), ("b", "auth")]
    assert caught.value.kind == "exhausted"


def test_a_provider_that_keeps_failing_rests_then_is_probed_again():
    clock = Clock()
    a, b = Fake(err(), err(), "A"), Fake("B")
    c = chain(a, b, policy=ChainPolicy(fail_threshold=2, cooldown_s=30), clock=clock)
    assert run(c) == "B"  # a fails once
    assert run(c) == "B"  # a fails twice and starts resting
    assert a.calls == 2
    assert run(c) == "B" and a.calls == 2  # skipped while it rests
    clock.advance(31)
    assert run(c) == "A" and a.calls == 3  # probed after the rest, and healthy again
    assert run(c) == "A"


def test_one_failure_after_a_rest_sends_the_provider_straight_back_to_rest():
    clock = Clock()
    a = Fake(err(), err(), err(), "A")
    c = chain(a, Fake("B"), policy=ChainPolicy(fail_threshold=2, cooldown_s=30), clock=clock)
    run(c), run(c)
    clock.advance(31)
    run(c)  # the probe fails
    assert run(c) == "B" and a.calls == 3  # and it rests again at once


def test_a_refused_key_rests_for_the_auth_cooldown_and_a_rate_limit_for_its_retry_after():
    clock = Clock()
    c = chain(Fake(err("auth")), Fake(err("rate_limit", retry_after=5.0)), Fake("C"), clock=clock)
    assert run(c) == "C"
    rows = {r["name"]: r for r in c.status()}
    assert rows["a"]["resting_s"] == 900.0
    assert rows["b"]["resting_s"] == 5.0


def test_when_everything_rests_the_one_that_recovers_first_is_still_tried():
    clock = Clock()
    a, b = Fake(err("rate_limit", retry_after=60), "A"), Fake(err("rate_limit", retry_after=10), "B")
    c = chain(a, b, clock=clock)
    with pytest.raises(AllProvidersFailed):
        run(c)
    assert run(c) == "B"  # b rests for less time, so it goes first even though both rest


def test_a_pin_moves_a_provider_first_and_keeps_the_rest_as_fallbacks():
    a, b = Fake("A"), Fake(err(), "B")
    c = chain(a, b)
    assert c.pin("b") == "b"
    assert run(c) == "A"  # b failed, so the chain fell back to a
    assert c.events[-1].from_slot == "b" and c.events[-1].to_slot == "a"
    c.unpin()
    assert c.pinned is None


def test_a_strict_pin_never_falls_back():
    a = Fake("A")
    c = chain(a, Fake(err("server")))
    c.pin("b", strict=True)
    with pytest.raises(AllProvidersFailed):
        run(c)
    assert a.calls == 0


def test_a_pin_may_be_a_unique_prefix_and_must_name_exactly_one():
    c = Chain("llm", [Slot("openai:gpt", Fake()), Slot("openrouter:deepseek", Fake()), Slot("openrouter:other", Fake())])
    assert c.pin("openai") == "openai:gpt"
    with pytest.raises(ValueError):
        c.pin("openrouter")  # two match
    with pytest.raises(ValueError):
        c.pin("nope")


def test_latency_order_puts_the_fastest_healthy_provider_first_after_each_is_timed():
    clock = Clock()

    def timed(seconds, value):
        def call():
            clock.advance(seconds)
            return value

        return call

    c = chain(timed(3.0, "slow"), timed(0.5, "fast"), policy=ChainPolicy(order="latency"), clock=clock)
    assert run(c) == "slow"  # nothing is measured yet: listed order, and a is timed
    assert run(c) == "fast"  # b has no measurement, so it is timed next
    assert run(c) == "fast"  # now b is the faster of the two
    assert [r["avg_latency_s"] for r in c.status()] == [3.0, 0.5]


def test_stream_fails_over_before_the_first_item_and_not_after():
    def good():
        yield from ("x", "y")

    def dead():
        raise err("transport")
        yield  # pragma: no cover

    def breaks_midway():
        yield "x"
        raise err("transport")

    c = Chain("llm", [Slot("a", dead), Slot("b", good)], clock=Clock())
    assert list(c.stream(lambda s: s.client())) == ["x", "y"]
    assert c.events[-1].from_slot == "a"

    c = Chain("llm", [Slot("a", breaks_midway), Slot("b", good)], clock=Clock())
    seen = []
    with pytest.raises(ProviderError) as caught:
        for item in c.stream(lambda s: s.client()):
            seen.append(item)
    assert seen == ["x"] and caught.value.kind == "stream"


def test_an_empty_stream_is_a_success_not_a_failure():
    c = Chain("llm", [Slot("a", lambda: iter(()))], clock=Clock())
    assert list(c.stream(lambda s: s.client())) == []
    assert c.status()[0]["failures"] == 0


def test_a_slow_provider_is_raced_against_the_next_and_the_first_answer_wins():
    release = threading.Event()

    def slow():
        release.wait(5)
        return "slow"

    c = Chain("llm", [Slot("a", slow), Slot("b", lambda: "fast")], ChainPolicy(hedge_after_s=0.05))
    try:
        assert c.call(lambda s: s.client(), hedge=True) == "fast"
    finally:
        release.set()
    assert [(e.from_slot, e.to_slot, e.kind) for e in c.events] == [("a", "b", "slow")]


def test_hedging_still_fails_over_when_the_first_provider_fails_fast():
    c = Chain("llm", [Slot("a", Fake(err("server"))), Slot("b", lambda: "B")], ChainPolicy(hedge_after_s=5))
    assert c.call(lambda s: s.client(), hedge=True) == "B"


def test_hedging_reports_every_failure_when_all_fail():
    c = Chain("llm", [Slot("a", Fake(err("server"))), Slot("b", Fake(err("timeout")))], ChainPolicy(hedge_after_s=5))
    with pytest.raises(AllProvidersFailed) as caught:
        c.call(lambda s: s.client(), hedge=True)
    assert {n for n, _ in caught.value.errors} == {"a", "b"}


def test_hedge_is_ignored_unless_the_policy_asks_for_it():
    a, b = Fake("A"), Fake("B")
    assert run(chain(a, b), hedge=True) == "A"
    assert b.calls == 0


def test_a_broken_listener_does_not_break_the_call():
    def listener(event):
        raise RuntimeError("boom")

    assert run(chain(Fake(err()), Fake("B"), on_event=listener)) == "B"


def test_a_bug_in_the_call_propagates_and_leaves_health_alone():
    c = chain(Fake(KeyError("bug")), Fake("B"))
    with pytest.raises(KeyError):
        run(c)
    assert c.status()[0]["failures"] == 0


def test_status_codes_map_to_error_kinds():
    kinds = {s: from_status(s, "body", provider="p").kind for s in (400, 401, 403, 404, 408, 429, 500, 503, 504)}
    assert kinds == {
        400: "bad_request",
        401: "auth",
        403: "auth",
        404: "unsupported",
        408: "timeout",
        429: "rate_limit",
        500: "server",
        503: "server",
        504: "timeout",
    }


def test_an_error_message_keeps_the_status_and_a_short_snippet_only():
    message = str(from_status(500, "x" * 1000, provider="p"))
    assert "500" in message and len(message) < 400


def test_duplicate_or_missing_providers_are_refused():
    with pytest.raises(ValueError):
        Chain("llm", [])
    with pytest.raises(ValueError):
        Chain("llm", [Slot("a", Fake()), Slot("a", Fake())])
    with pytest.raises(ValueError):
        ChainPolicy(order="random")

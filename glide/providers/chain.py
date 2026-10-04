"""An ordered chain of interchangeable clients for one role, with automatic failover.

One `Chain` holds the providers for one job (the fast LLM, speech to text, ...). A call goes to the
first healthy provider, moves on when it fails, and stops sending to one that keeps failing until a
cooldown passes. The user can pin a provider, and the chain can race a slow one against the next.

Every switch is reported as a `SwitchEvent`, kept on `chain.events` and passed to `on_event`, so a
fallback is never silent.

The chain is transport-agnostic: it reacts only to `ProviderError` (see errors.py). Any other
exception is a bug and propagates untouched, with no effect on a provider's health.

Interruption. A call made while a `RunControl` is current (glide/computer/control.py) is cancellable: the
provider is called through `control.interruptible`, so the caller gets a `ProviderError` of kind "cancelled" the
moment the control is cancelled, however the provider is blocked, and whatever the provider raises after that
(a closed connection reads as a transport error) is that same cancel. A cancel is not a provider fault: it is not
retried, is never failed over to the next slot, leaves the slot's health alone and is never a `SwitchEvent`.
A hedged call gives each racer its own control, linked to the caller's: the user's cancel reaches both, and the
loser is cancelled alone when the winner answers, so its connection is closed instead of being left to finish.
"""

from __future__ import annotations

import contextlib
import queue
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import Future
from dataclasses import dataclass
from functools import partial

from glide.computer.control import RunControl, current_control, linked, spawn
from glide.computer.models import Abort

from .errors import CANCELLED, FAILOVER_KINDS, AllProvidersFailed, ProviderError, cancelled

ORDERS = ("priority", "latency")
HEDGE_LOST = "another provider answered first"
_END = object()


def _begin[T, R](fn: Callable[[Slot[T]], Iterator[R]], slot: Slot[T]) -> Iterator[R]:
    return iter(fn(slot))


def _advance[R](items: Iterator[R], control: RunControl | None) -> R | object:
    """The next item, or `_END`. When the control was cancelled while it was being made, the generator is closed here,
    on the thread that ran it: the caller has gone and cannot close a generator that was still executing."""
    item = next(items, _END)
    if control is not None and control.cancelled.is_set():
        close = getattr(items, "close", None)
        if close is not None:
            with contextlib.suppress(Exception):
                close()
    return item


@dataclass(frozen=True)
class Slot[T]:
    """One provider in a chain. `name` is unique within it, such as 'openrouter:deepseek/deepseek-v4.1-flash'."""

    name: str
    client: T


@dataclass(frozen=True)
class ChainPolicy:
    """How a chain behaves. Every field has a default that works without configuration."""

    order: str = "priority"  # "priority": as listed. "latency": healthy slots, fastest average first
    fail_threshold: int = 2  # consecutive failures before a slot rests
    cooldown_s: float = 30.0  # how long a failing slot rests
    auth_cooldown_s: float = 900.0  # a refused key will not fix itself soon
    hedge_after_s: float | None = None  # race the next slot when the first has not answered by then
    failover_kinds: frozenset[str] = FAILOVER_KINDS
    latency_alpha: float = 0.3  # weight of the newest sample in the average

    def __post_init__(self) -> None:
        if self.order not in ORDERS:
            raise ValueError(f"order must be one of {', '.join(ORDERS)}, not {self.order!r}")


@dataclass(frozen=True)
class SwitchEvent:
    """A provider was skipped, failed, or raced. `to_slot` is None when nothing was left to try."""

    role: str
    from_slot: str
    to_slot: str | None
    kind: str  # the error kind, or "slow" for a hedge
    reason: str


@dataclass
class _Health:
    failures: int = 0
    rest_until: float = 0.0
    latency: float | None = None
    calls: int = 0
    last_error: str | None = None


class Chain[T]:
    def __init__(
        self,
        role: str,
        slots: list[Slot[T]],
        policy: ChainPolicy | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        on_event: Callable[[SwitchEvent], None] | None = None,
    ):
        if not slots:
            raise ValueError(f"the {role} chain has no providers")
        names = [s.name for s in slots]
        if len(set(names)) != len(names):
            raise ValueError(f"the {role} chain lists a provider twice: {names}")
        self.role = role
        self.policy = policy or ChainPolicy()
        self.events: deque[SwitchEvent] = deque(maxlen=200)
        self._slots = list(slots)
        self._health = {s.name: _Health() for s in slots}
        self._pinned: str | None = None
        self._strict = False
        self._announced: dict[str, float] = {}  # slot name to the rest that has been announced
        self._clock = clock
        self._on_event = on_event
        self._lock = threading.RLock()

    # -- control --------------------------------------------------------------------------------

    @property
    def names(self) -> list[str]:
        return [s.name for s in self._slots]

    @property
    def pinned(self) -> str | None:
        return self._pinned

    def pin(self, name: str, *, strict: bool = False) -> str:
        """Prefer one provider. `name` is a slot's full name or a prefix that names exactly one.

        Not strict: the pinned slot goes first and the rest remain as fallbacks. Strict: only the
        pinned slot is ever used, and its failure is the caller's error.
        """
        matches = [n for n in self.names if n == name] or [n for n in self.names if n.startswith(name)]
        if len(matches) != 1:
            raise ValueError(f"{name!r} names {len(matches)} of the {self.role} providers {self.names}")
        with self._lock:
            self._pinned, self._strict = matches[0], strict
        return matches[0]

    def unpin(self) -> None:
        with self._lock:
            self._pinned, self._strict = None, False

    def status(self) -> list[dict]:
        """One row per slot, in listed order, for a status display or `glide doctor`."""
        now = self._clock()
        with self._lock:
            return [
                {
                    "name": s.name,
                    "pinned": s.name == self._pinned,
                    "resting_s": round(max(0.0, self._health[s.name].rest_until - now), 1),
                    "failures": self._health[s.name].failures,
                    "calls": self._health[s.name].calls,
                    "avg_latency_s": self._health[s.name].latency,
                    "last_error": self._health[s.name].last_error,
                }
                for s in self._slots
            ]

    # -- calling --------------------------------------------------------------------------------

    def call[R](self, fn: Callable[[Slot[T]], R], *, hedge: bool = False) -> R:
        """Run `fn(slot)` on the first provider that answers.

        `hedge` is for calls that are safe to run twice (a chat completion, a transcription), and
        only does anything when the policy sets `hedge_after_s`. Never hedge an action on the world.
        """
        order = self._order()
        self._announce_resting(order)
        errors: list[tuple[str, ProviderError]] = []
        if hedge and self.policy.hedge_after_s is not None and len(order) > 1:
            return self._hedged(fn, order, errors)
        control = current_control()
        for i, slot in enumerate(order):
            try:
                return self._attempt(fn, slot, control)
            except ProviderError as e:
                errors.append((slot.name, e))
                if e.kind not in self.policy.failover_kinds:
                    raise
                self._emit(slot.name, order[i + 1].name if i + 1 < len(order) else None, e)
        raise AllProvidersFailed(self.role, errors)

    def stream[R](self, fn: Callable[[Slot[T]], Iterator[R]]) -> Iterator[R]:
        """Yield from `fn(slot)` on the first provider that produces an item.

        Failover happens only until the first item arrives. After that the caller has already used
        what came before, so a failure is raised to it as a ProviderError of kind "stream". A cancel is
        raised as it is, as "cancelled", at any point.

        The control is the one current when this is called, not when the first item is asked for.
        """
        return self._stream(fn, current_control())

    def _stream[R](self, fn: Callable[[Slot[T]], Iterator[R]], control: RunControl | None) -> Iterator[R]:
        order = self._order()
        self._announce_resting(order)
        errors: list[tuple[str, ProviderError]] = []
        for i, slot in enumerate(order):
            started = self._clock()
            items: Iterator[R] | None = None
            try:
                items = self._step(slot, control, partial(_begin, fn, slot))
                first = self._step(slot, control, partial(_advance, items, control))
                if first is _END:
                    self._succeeded(slot.name, self._clock() - started)
                    return
            except ProviderError as e:
                self._failed(slot.name, e)
                errors.append((slot.name, e))
                if e.kind not in self.policy.failover_kinds:
                    raise
                self._emit(slot.name, order[i + 1].name if i + 1 < len(order) else None, e)
                continue
            self._succeeded(slot.name, self._clock() - started)  # latency to the first item
            try:
                yield first
                while (item := self._step(slot, control, partial(_advance, items, control))) is not _END:
                    yield item
            except ProviderError as e:
                if e.kind == CANCELLED:
                    raise
                self._failed(slot.name, e)
                raise ProviderError(str(e), kind="stream", provider=e.provider or slot.name, status=e.status) from e
            finally:
                close = getattr(
                    items, "close", None
                )  # the consumer stopped early, or a cancel ended it: give the connection back
                if close is not None:
                    with contextlib.suppress(Exception):  # ValueError while an abandoned step is still running: it closes itself
                        close()
            return
        raise AllProvidersFailed(self.role, errors)

    # -- internals ------------------------------------------------------------------------------

    def _order(self) -> list[Slot[T]]:
        """Slots in the order to try them: ready ones first, resting ones last as a final resort."""
        now = self._clock()
        with self._lock:
            slots = list(self._slots)
            if self._pinned and self._strict:
                return [s for s in slots if s.name == self._pinned]
            ready = [s for s in slots if self._health[s.name].rest_until <= now]
            resting = sorted(
                (s for s in slots if self._health[s.name].rest_until > now), key=lambda s: self._health[s.name].rest_until
            )
            if self.policy.order == "latency":
                # A slot with no measurement yet goes first, so each is timed once.
                ready.sort(key=lambda s: -1.0 if self._health[s.name].latency is None else self._health[s.name].latency)
            if self._pinned:
                ready.sort(key=lambda s: s.name != self._pinned)  # stable: only the pinned slot moves up
                resting.sort(key=lambda s: s.name != self._pinned)
            return ready + resting

    def _announce_resting(self, order: list[Slot[T]]) -> None:
        """Say, once for each rest, that a slot resting after failures is being passed over for the one that takes this call.

        `_order` moves a resting slot to the end without a word, and its failures were announced only when they
        happened, so later calls served by another slot would otherwise show nothing. When every slot left is
        resting, the first one is tried as a last resort: that is a try, not a skip.
        """
        now = self._clock()
        with self._lock:
            if not order or self._health[order[0].name].rest_until > now:
                return
            usable = {s.name for s in order}
            news = [
                (s.name, h.rest_until, h.last_error)
                for s in self._slots
                if s.name in usable
                and (h := self._health[s.name]).rest_until > now
                and self._announced.get(s.name) != h.rest_until
            ]
            self._announced.update({name: until for name, until, _ in news})
        for name, _, last in news:
            self._emit_raw(name, order[0].name, "resting", f"{name} is resting after {last or 'earlier failures'}")

    def _attempt[R](self, fn: Callable[[Slot[T]], R], slot: Slot[T], control: RunControl | None, *, hop: bool = True) -> R:
        started = self._clock()
        try:
            result = self._step(slot, control, lambda: fn(slot), hop=hop)
        except ProviderError as e:
            self._failed(slot.name, e)
            raise
        self._succeeded(slot.name, self._clock() - started)
        return result

    @staticmethod
    def _step[R](slot: Slot[T], control: RunControl | None, call: Callable[[], R], *, hop: bool = True) -> R:
        """One blocking step of a provider's work. With a control it ends the moment the control is cancelled,
        and whatever the provider raised once that has happened is the cancel, not a fault of its own.

        `hop=False` is for a caller that is already on a thread of its own to abandon (a hedged racer)."""
        if control is None:
            return call()
        try:
            return control.interruptible(call) if hop else call()
        except Abort:
            raise cancelled(slot.name, control.reason) from None
        except ProviderError as e:
            if control.cancelled.is_set() and e.kind != CANCELLED:
                raise cancelled(slot.name, control.reason) from None
            raise

    def _hedged[R](self, fn: Callable[[Slot[T]], R], order: list[Slot[T]], errors: list[tuple[str, ProviderError]]) -> R:
        parent = current_control()
        finished: queue.SimpleQueue[Future | None] = queue.SimpleQueue()  # an attempt that ended, or None: cancelled
        racers: dict[Future, RunControl] = {}
        slots: dict[Future, Slot[T]] = {}
        launched = 0
        try:
            with contextlib.ExitStack() as links:
                if parent is not None:
                    try:
                        links.enter_context(parent.closing_on_cancel(lambda: finished.put(None)))
                    except Abort:
                        raise cancelled(order[0].name, parent.reason) from None

                def launch() -> Future:
                    nonlocal launched
                    slot = order[launched]
                    launched += 1
                    try:
                        control = links.enter_context(linked(parent))
                    except Abort:
                        raise cancelled(slot.name, parent.reason if parent else "") from None
                    future = spawn(lambda: self._attempt(fn, slot, control, hop=False), control)
                    racers[future], slots[future] = control, slot
                    future.add_done_callback(finished.put)
                    return future

                pending = {launch()}
                while pending:
                    racing = launched < len(order)
                    try:
                        first = finished.get(timeout=self.policy.hedge_after_s if racing else None)
                    except queue.Empty:
                        waiting_on = ", ".join(slots[f].name for f in pending)
                        self._emit_raw(waiting_on, order[launched].name, "slow", f"no answer in {self.policy.hedge_after_s}s")
                        pending.add(launch())
                        continue
                    batch = [first]
                    while True:  # whatever else has ended by now is judged together: a success beats a failure
                        try:
                            batch.append(finished.get_nowait())
                        except queue.Empty:
                            break
                    cancelled_now = parent is not None and parent.cancelled.is_set()
                    if None in batch or cancelled_now:  # an answer that arrived with the cancel is dropped
                        raise cancelled(order[0].name, parent.reason if parent else "")
                    pending.difference_update(batch)
                    for future in sorted(batch, key=lambda f: f.exception() is not None):
                        slot = slots[future]
                        try:
                            return future.result()
                        except ProviderError as e:
                            errors.append((slot.name, e))
                            if e.kind not in self.policy.failover_kinds:
                                raise
                            carrying_on = [slots[f].name for f in pending]  # a racer in flight takes over before a new one does
                            self._emit(
                                slot.name,
                                carrying_on[0] if carrying_on else order[launched].name if launched < len(order) else None,
                                e,
                            )
                    if not pending and launched < len(order):
                        pending.add(launch())
                raise AllProvidersFailed(self.role, errors)
        finally:
            for future, control in racers.items():  # a racer that is still going has lost: close its connection
                if not future.done():
                    control.cancel(HEDGE_LOST)

    def _succeeded(self, name: str, latency: float) -> None:
        with self._lock:
            h = self._health[name]
            h.failures, h.rest_until, h.last_error = 0, 0.0, None
            h.calls += 1
            alpha = self.policy.latency_alpha
            h.latency = latency if h.latency is None else alpha * latency + (1 - alpha) * h.latency

    def _failed(self, name: str, error: ProviderError) -> None:
        if error.kind not in self.policy.failover_kinds:
            return  # our own bad request says nothing about the provider
        now = self._clock()
        with self._lock:
            h = self._health[name]
            h.failures += 1
            h.last_error = f"{error.kind}: {error}"
            if error.kind == "auth":
                h.rest_until = now + self.policy.auth_cooldown_s
            elif error.kind == "rate_limit":
                h.rest_until = now + (error.retry_after or self.policy.cooldown_s)
            elif h.failures >= self.policy.fail_threshold:
                # The count is kept, so the first call after the rest is a probe: one more failure
                # and the slot rests again, one success and it is healthy.
                h.rest_until = now + self.policy.cooldown_s

    def _emit(self, from_slot: str, to_slot: str | None, error: ProviderError) -> None:
        self._emit_raw(from_slot, to_slot, error.kind, str(error))

    def _emit_raw(self, from_slot: str, to_slot: str | None, kind: str, reason: str) -> None:
        event = SwitchEvent(self.role, from_slot, to_slot, kind, reason)
        self.events.append(event)
        if self._on_event is not None:
            with contextlib.suppress(Exception):  # a broken listener must not break the call it is listening to
                self._on_event(event)

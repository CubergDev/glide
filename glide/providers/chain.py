"""An ordered chain of interchangeable clients for one role, with automatic failover.

One `Chain` holds the providers for one job (the fast LLM, speech to text, ...). A call goes to the
first healthy provider, moves on when it fails, and stops sending to one that keeps failing until a
cooldown passes. The user can pin a provider, and the chain can race a slow one against the next.

Every switch is reported as a `SwitchEvent`, kept on `chain.events` and passed to `on_event`, so a
fallback is never silent.

The chain is transport-agnostic: it reacts only to `ProviderError` (see errors.py). Any other
exception is a bug and propagates untouched, with no effect on a provider's health.
"""

from __future__ import annotations

import contextlib
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass

from .errors import FAILOVER_KINDS, AllProvidersFailed, ProviderError

ORDERS = ("priority", "latency")


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
        errors: list[tuple[str, ProviderError]] = []
        if hedge and self.policy.hedge_after_s is not None and len(order) > 1:
            return self._hedged(fn, order, errors)
        for i, slot in enumerate(order):
            try:
                return self._attempt(fn, slot)
            except ProviderError as e:
                errors.append((slot.name, e))
                if e.kind not in self.policy.failover_kinds:
                    raise
                self._emit(slot.name, order[i + 1].name if i + 1 < len(order) else None, e)
        raise AllProvidersFailed(self.role, errors)

    def stream[R](self, fn: Callable[[Slot[T]], Iterator[R]]) -> Iterator[R]:
        """Yield from `fn(slot)` on the first provider that produces an item.

        Failover happens only until the first item arrives. After that the caller has already used
        what came before, so a failure is raised to it as a ProviderError of kind "stream".
        """
        order = self._order()
        errors: list[tuple[str, ProviderError]] = []
        for i, slot in enumerate(order):
            started = self._clock()
            try:
                items = iter(fn(slot))
                first = next(items)
            except StopIteration:
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
            yield first
            try:
                yield from items
            except ProviderError as e:
                self._failed(slot.name, e)
                raise ProviderError(str(e), kind="stream", provider=e.provider or slot.name, status=e.status) from e
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

    def _attempt[R](self, fn: Callable[[Slot[T]], R], slot: Slot[T]) -> R:
        started = self._clock()
        try:
            result = fn(slot)
        except ProviderError as e:
            self._failed(slot.name, e)
            raise
        self._succeeded(slot.name, self._clock() - started)
        return result

    def _hedged[R](self, fn: Callable[[Slot[T]], R], order: list[Slot[T]], errors: list[tuple[str, ProviderError]]) -> R:
        pool = ThreadPoolExecutor(max_workers=len(order), thread_name_prefix=f"chain-{self.role}")
        futures: dict[Future, Slot[T]] = {}
        launched = 0

        def launch() -> Future:
            nonlocal launched
            slot = order[launched]
            launched += 1
            future = pool.submit(self._attempt, fn, slot)
            futures[future] = slot
            return future

        pending = {launch()}
        try:
            while pending:
                racing = launched < len(order)
                done, pending = wait(pending, timeout=self.policy.hedge_after_s if racing else None, return_when=FIRST_COMPLETED)
                if not done:
                    waiting_on = ", ".join(futures[f].name for f in pending)
                    self._emit_raw(waiting_on, order[launched].name, "slow", f"no answer in {self.policy.hedge_after_s}s")
                    pending.add(launch())
                    continue
                for future in sorted(done, key=lambda f: f.exception() is not None):  # a success beats a failure
                    slot = futures[future]
                    try:
                        return future.result()
                    except ProviderError as e:
                        errors.append((slot.name, e))
                        if e.kind not in self.policy.failover_kinds:
                            raise
                        self._emit(slot.name, order[launched].name if launched < len(order) else None, e)
                if not pending and launched < len(order):
                    pending.add(launch())
            raise AllProvidersFailed(self.role, errors)
        finally:
            # The slower request cannot be cancelled once sent; its answer is simply dropped.
            pool.shutdown(wait=False, cancel_futures=True)

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

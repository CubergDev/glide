"""RunControl: the stop gate every input action passes (D2)."""

import threading

import pytest

from glide.computer.control import RunControl, checkpoint, controlled, current_control, dispatch
from glide.computer.models import Abort


def test_no_control_means_no_gate():
    assert current_control() is None
    checkpoint()
    assert dispatch(lambda a, b=0: a + b, 1, b=2) == 3


def test_a_cancelled_control_raises_at_checkpoint_with_the_first_reason():
    control = RunControl("t")
    control.cancel("stopped by the user")
    control.cancel("something later")
    with pytest.raises(Abort, match="stopped by the user"):
        control.check()
    with controlled(control), pytest.raises(Abort, match="stopped by the user"):
        checkpoint()


def test_default_reason_names_the_cancel():
    control = RunControl()
    control.cancel()
    with pytest.raises(Abort, match="task cancelled"):
        control.check(wait=False)


def test_dispatch_after_cancel_never_runs_the_action():
    """Cancel during a correction (a re-plan): the gate refuses, so nothing reaches the machine."""
    control = RunControl("t")
    sent = []
    with controlled(control):
        control.cancel()
        with pytest.raises(Abort):
            dispatch(sent.append, "click")
    assert sent == [] and not control.in_flight


def test_dispatch_marks_the_write_in_flight_until_a_fresh_observation_clears_it():
    control = RunControl("t")
    with controlled(control):
        assert not control.in_flight
        dispatch(lambda: None)
        assert control.in_flight
    control.in_flight = False  # the runner does this after reading the screen again


def test_a_stop_during_the_action_leaves_the_write_marked_unknown():
    control = RunControl("t")

    def action():
        control.cancel()
        checkpoint()

    with controlled(control), pytest.raises(Abort):
        dispatch(action)
    assert control.in_flight


def test_cancel_closes_registered_connections_and_forgets_them_afterwards():
    control = RunControl("t")
    closed = []
    with control.closing_on_cancel(lambda: closed.append("socket")):
        control.cancel()
    assert closed == ["socket"]
    later = RunControl("t2")
    with later.closing_on_cancel(lambda: closed.append("never")):
        pass
    later.cancel()
    assert closed == ["socket"]


def test_a_failing_close_callback_does_not_stop_the_cancel():
    control = RunControl("t")
    seen = []

    def bad():
        raise OSError("already closed")

    with control.closing_on_cancel(bad), control.closing_on_cancel(lambda: seen.append("ok")):
        control.cancel()
    assert seen == ["ok"] and control.cancelled.is_set()


def test_pause_blocks_until_resume_and_cancel_releases_a_paused_wait():
    control = RunControl("t")
    control.pause()
    released = threading.Event()
    errors = []

    def wait():
        try:
            control.check()
        except Abort as error:
            errors.append(str(error))
        released.set()

    thread = threading.Thread(target=wait, daemon=True)
    thread.start()
    assert not released.wait(0.2)
    control.cancel("stop")
    assert released.wait(2) and errors == ["stop"]


def test_the_control_is_per_context_not_process_wide():
    control = RunControl("t")
    seen = []
    with controlled(control):
        thread = threading.Thread(target=lambda: seen.append(current_control()))
        thread.start()
        thread.join()
    assert seen == [None] and current_control() is None


def test_events_reach_the_emitter_with_the_task_id():
    events = []
    control = RunControl("task-7", events.append)
    control.event("progress", "working", outcome="")
    assert [(e.task_id, e.kind, e.text) for e in events] == [("task-7", "progress", "working")]


# -- two cancellers, and a close callback against the block it belongs to (finding 5) ---------------------


class GatedEvent(threading.Event):
    """An Event whose first `set()` holds there until `proceed` is set (or WAIT passes): the moment between a canceller
    having chosen a reason and the control being marked cancelled."""

    def __init__(self) -> None:
        super().__init__()
        self.entered, self.proceed, self._first = threading.Event(), threading.Event(), True

    def set(self) -> None:
        if self._first:
            self._first = False
            self.entered.set()
            self.proceed.wait(0.3)
        super().set()


def test_two_concurrent_cancellers_agree_on_the_first_reason():
    """F5: the reason was written under the lock but the flag set outside it, so a second canceller in between
    overwrote the reason the first had chosen."""
    control = RunControl("t")
    control.cancelled = GatedEvent()
    first = threading.Thread(target=control.cancel, args=("first",))
    first.start()
    assert control.cancelled.entered.wait(3)
    second = threading.Thread(target=lambda: (control.cancel("second"), control.cancelled.proceed.set()))
    second.start()
    first.join(3)
    second.join(3)
    assert control.reason == "first"
    with pytest.raises(Abort, match="first"):
        control.check(wait=False)


def test_a_close_callback_never_runs_after_its_block_has_exited():
    """F5: `cancel` took a snapshot of the callbacks and ran them with nothing held, so the callback of a block that
    had exited in the meantime (a connection by then back in a pool) still ran."""
    control = RunControl("t")
    ran, running, release = [], [], threading.Event()
    first_running = threading.Event()

    def make(name):
        def close():
            ran.append(name)
            if not first_running.is_set():  # whichever the cancel reaches first holds there
                running.append(name)
                first_running.set()
                release.wait(3)

        return close

    a, b = control.closing_on_cancel(make("a")), control.closing_on_cancel(make("b"))
    blocks = {"a": a, "b": b}
    a.__enter__()
    b.__enter__()
    canceller = threading.Thread(target=control.cancel, args=("stop",))
    canceller.start()
    assert first_running.wait(3)
    (held,) = running
    other = "b" if held == "a" else "a"
    blocks[other].__exit__(None, None, None)  # its block ends while the cancel is still going through the callbacks
    release.set()
    canceller.join(3)
    blocks[held].__exit__(None, None, None)
    assert ran == [held]


def test_leaving_a_block_waits_for_its_own_callback_that_is_running():
    control = RunControl("t")
    running, release, left = threading.Event(), threading.Event(), threading.Event()
    block = control.closing_on_cancel(lambda: (running.set(), release.wait(3)))
    block.__enter__()
    canceller = threading.Thread(target=control.cancel, args=("stop",))
    canceller.start()
    assert running.wait(3)
    leaver = threading.Thread(target=lambda: (block.__exit__(None, None, None), left.set()))
    leaver.start()
    assert not left.wait(0.2)  # the callback is still using what the block owned
    release.set()
    assert left.wait(3)
    canceller.join(3)
    leaver.join(3)

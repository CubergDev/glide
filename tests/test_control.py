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

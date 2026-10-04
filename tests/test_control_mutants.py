"""Mutation-testing follow-up for `glide/computer/control.py` (see docs/MUTATION_REPORT.md).

Each test here was written against a mutant of RunControl that the existing suite let survive. They pin behaviour
that the stop path depends on: a paused control waits for `resume`, a closing registration does not wait on a pause,
and a child control follows its parent's cancel and nobody else's. No sleeps: waits are replaced by a scripted
`ready` event, and the one real thread is joined with a bound that only a broken implementation can reach.
"""

import threading

import pytest

from glide.computer.control import RunControl, checkpoint, controlled, linked
from glide.computer.models import Abort

BOUND_S = 5.0  # a join that only a hang can exhaust


class ScriptedReady:
    """Stands in for `RunControl.ready`: `wait` reports "still paused" a set number of times, then "resumed"."""

    def __init__(self, paused_polls: int) -> None:
        self.paused_polls = paused_polls
        self.waits = 0

    def wait(self, timeout: float | None = None) -> bool:
        self.waits += 1
        return self.waits > self.paused_polls

    def set(self) -> None:
        self.paused_polls = 0

    def clear(self) -> None:
        pass


def test_resume_lets_a_paused_control_through():
    control = RunControl("t")
    control.pause()
    assert not control.ready.is_set()
    control.resume()
    assert control.ready.is_set()

    done = threading.Event()
    threading.Thread(target=lambda: (control.check(), done.set()), daemon=True).start()
    assert done.wait(BOUND_S), "check() stayed blocked after resume()"


def test_a_checkpoint_waits_for_resume_by_default_and_only_polls_when_told_not_to():
    control = RunControl("t")
    control.ready = ScriptedReady(paused_polls=3)
    checkpoint(control)
    assert control.ready.waits == 4  # polled three times while paused, passed on the fourth

    control.ready = ScriptedReady(paused_polls=3)
    checkpoint(control, wait=False)
    assert control.ready.waits == 0  # a caller that cannot wait is never held by a pause

    control.ready = ScriptedReady(paused_polls=2)
    with controlled(control):
        checkpoint()  # the context's control is held by a pause as well
    assert control.ready.waits == 3


def test_registering_a_close_does_not_wait_on_a_pause():
    """A paused control must still accept a connection to close, or a stop during a pause could not reach it."""
    control = RunControl("t")
    control.pause()
    closed = []
    entered = threading.Event()

    def register():
        with control.closing_on_cancel(lambda: closed.append("socket")):
            entered.set()
            control.cancel("stop")

    threading.Thread(target=register, daemon=True).start()
    assert entered.wait(BOUND_S), "closing_on_cancel blocked on a paused control"
    assert closed == ["socket"]


def test_an_answer_is_returned_from_a_paused_control_not_held_until_resume():
    control = RunControl("t")
    control.pause()
    result = []
    finished = threading.Event()

    def call():
        result.append(control.interruptible(lambda: "answer"))
        finished.set()

    threading.Thread(target=call, daemon=True).start()
    assert finished.wait(BOUND_S), "interruptible() blocked on a paused control that was never cancelled"
    assert result == ["answer"]


def test_a_linked_child_follows_its_parents_cancel_with_the_parents_reason():
    parent = RunControl("task-1")
    with linked(parent) as child:
        assert child is not parent and child.task_id == "task-1"
        assert not child.cancelled.is_set()
        parent.cancel("stopped by the user")
        assert child.cancelled.is_set()
        with pytest.raises(Abort, match="stopped by the user"):
            child.check(wait=False)


def test_a_linked_child_cancelled_alone_leaves_its_parent_running():
    parent = RunControl("task-1")
    with linked(parent) as child:
        child.cancel("another provider answered first")
    assert not parent.cancelled.is_set()


def test_a_child_is_not_cancelled_by_a_parent_cancel_after_its_block_has_ended():
    parent = RunControl("task-1")
    with linked(parent) as child:
        pass
    parent.cancel("late")
    assert not child.cancelled.is_set()


def test_linking_to_a_parent_that_is_already_cancelled_raises_at_once():
    parent = RunControl("task-1")
    parent.cancel("stopped by the user")
    with pytest.raises(Abort, match="stopped by the user"), linked(parent):
        pytest.fail("the block must not run under a cancelled parent")


def test_without_a_parent_a_child_is_a_plain_control():
    with linked(None) as child:
        assert child.task_id == "" and not child.cancelled.is_set()

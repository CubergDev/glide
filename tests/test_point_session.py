"""PointSession: follow-ups, interruption, late answers, a shutdown that returns at once, and stale pins.

The model call runs on a worker thread, so these tests block a fake `generate` on events they control and read time
from a hand-wound clock: nothing sleeps to prove a point, and "returned at once" is a statement about the clock.
"""

from __future__ import annotations

import json
import logging
import threading
from types import SimpleNamespace

import pytest
from point_fakes import Clock, SyntheticDesktop, reply_writer
from test_assistant_fakes import WAIT, wait_until

from glide.assistant.point_ask import HARD_TTL_S, capture_point
from glide.assistant.point_session import EXPIRED, PROVIDER_FAILED, UNEXPECTED, PointSession
from glide.computer.generation import GenerationResult
from glide.computer.platform_adapter import using
from glide.computer.writer import WriterError

SECRET = "selected secret and private provider body"


class Rig:
    """A session over a synthetic desktop, with what it emitted and spoke kept in order."""

    def __init__(self, writer=None, *, clock=None, **capture):
        self.clock = clock or Clock()
        self.fake = SyntheticDesktop()
        self.events: list[dict] = []
        self.spoken: list[str] = []
        self.cancelled_speech = 0
        with using(self.fake):
            self.selection = capture_point(clock=self.clock, **capture)
        self.session = PointSession(
            self.selection,
            writer if writer is not None else reply_writer(),
            lambda kind, **data: self.events.append({"kind": kind, **data}),
            speak=self.spoken.append,
            cancel_speech=self._cancel_speech,
            clock=self.clock,
        )

    def _cancel_speech(self):
        self.cancelled_speech += 1

    def kinds(self):
        return [event["kind"] for event in self.events]

    def count(self, kind):
        return self.kinds().count(kind)

    def wait_for(self, kind, n=1):
        assert wait_until(lambda: self.count(kind) >= n), f"no {kind} event: {self.kinds()}"

    def ask_and_wait(self, text, n):
        assert self.session.ask(text)
        self.wait_for("answer", n)


class Blocking:
    """A writer whose `generate` says it was entered, then blocks until released. `cancel` is what it was given."""

    def __init__(self, clock=None, take=0.0, reply='{"answer":"LATE RESULT","uncertain":false}'):
        self.entered, self.release, self.controls, self.requests = threading.Event(), threading.Event(), [], []
        self.clock, self.take, self.reply = clock, take, reply

    def generate(self, request, cancel=None):
        self.requests.append(request)
        self.controls.append(cancel)
        self.entered.set()
        assert self.release.wait(WAIT), "the test never released the provider"
        if self.clock is not None:
            self.clock.advance(self.take)  # the provider took this long to answer
        return GenerationResult(self.reply, "fixture", stop_reason="stop")


def test_follow_ups_share_only_four_exchanges_and_never_recapture_or_act():
    packets = []

    def generate(request, cancel=None):
        packet = json.loads(request.text)
        packets.append(packet)
        assert set(request.schema["properties"]) == {"answer", "uncertain"} and request.role == "recovery"
        return GenerationResult(
            json.dumps({"answer": "Answer " + packet["question"], "uncertain": False}), "actual-model", stop_reason="stop"
        )

    rig = Rig(SimpleNamespace(generate=generate))
    with using(rig.fake):
        for index in range(7):
            rig.fake.point = (1000, 700)  # the pointer wanders; the pin does not
            rig.ask_and_wait(f"Question {index}", index + 1)
    assert packets[0]["previous_exchanges"] == []
    assert [item["question"] for item in packets[-1]["previous_exchanges"]] == [f"Question {i}" for i in range(2, 6)]
    assert all(packet["observed"]["point"] == [200, 150] for packet in packets)
    assert len(rig.session.history) == 4 and rig.spoken == [f"Answer Question {i}" for i in range(7)]
    answers = [e for e in rig.events if e["kind"] == "answer"]
    assert len(answers) == 7 and all(e["model"] == "actual-model" and e["uncertain"] is False for e in answers)
    rig.session.close()
    assert not rig.session.history and not rig.session.available
    assert rig.fake.calls == [("pointer",), ("target", (200.0, 150.0))]  # one capture, for the whole conversation


def test_a_question_is_trimmed_and_an_empty_or_oversized_one_starts_nothing():
    rig = Rig(SimpleNamespace(generate=lambda *a, **k: pytest.fail("no model call")))
    assert not rig.session.ask("   ") and not rig.session.ask(None)
    assert not rig.session.ask("q" * 2049)
    assert rig.events == [{"kind": "error", "text": "Question is too long. Ask a shorter question.", "closed": False}]
    rig.session.close()


# -- interruption and late answers (c) ------------------------------------------------------------------


@pytest.mark.parametrize("interrupt", ["new_question", "stop", "repoint", "close"])
def test_interruption_cancels_the_run_and_discards_late_text_speech_and_history(interrupt):
    blocking = Blocking()
    rig = Rig(blocking)
    assert rig.session.ask("Explain")
    assert blocking.entered.wait(WAIT)
    new_selection = None
    if interrupt == "new_question":
        rig.session.ask("Translate instead")  # supersedes the first before it came back
    elif interrupt == "stop":
        rig.session.stop()
    elif interrupt == "repoint":
        with using(rig.fake):
            new_selection = capture_point(clock=rig.clock)
        assert rig.session.repoint(new_selection)
    else:
        rig.session.close()
    assert blocking.controls[0].cancelled.is_set()  # the RunControl handed to the provider was cancelled
    blocking.release.set()  # now the abandoned provider call returns, late
    if interrupt == "new_question":
        assert wait_until(lambda: len(blocking.requests) == 2)  # the newer question is the one that runs
        assert wait_until(lambda: rig.count("answer") == 1)
        assert [e["text"] for e in rig.events if e["kind"] == "answer"] == ["LATE RESULT"]  # the newer answer, once
        assert len(rig.session.history) == 1 and rig.session.history[0]["question"] == "Translate instead"
    else:
        wait_until(lambda: False, timeout=0.3)  # give a wrongly kept worker the chance to show itself
        assert "answer" not in rig.kinds() and "error" not in rig.kinds()
        assert rig.session.history == [] and rig.spoken == []
    assert rig.cancelled_speech >= 1
    rig.session.close()


def test_close_returns_at_once_while_the_provider_is_still_holding_the_connection():
    clock = Clock()
    blocking = Blocking(clock, take=30.0)  # a provider that would take its whole 30 s deadline
    rig = Rig(blocking, clock=clock)
    rig.session.ask("Explain")
    assert blocking.entered.wait(WAIT)
    before = clock()
    rig.session.close()
    assert clock() - before == 0.0  # the provider has not answered yet: close did not wait for it
    assert not blocking.release.is_set() and not rig.session.available
    blocking.release.set()  # the provider finally answers, 30 s late
    assert wait_until(lambda: clock() - before == 30.0)
    wait_until(lambda: False, timeout=0.3)
    assert rig.kinds() == ["thinking"] and rig.spoken == [] and rig.session.history == []


def test_a_provider_that_registers_its_connection_is_closed_by_the_cancel_and_the_worker_ends_promptly():
    """The cancellable path: the writer is handed the RunControl, and a provider opens its connection under
    `closing_on_cancel`. Cancelling the session closes it, so the call does not have to be waited out."""
    closed, entered, ended = threading.Event(), threading.Event(), threading.Event()

    def generate(request, cancel=None):
        try:
            with cancel.closing_on_cancel(closed.set):
                entered.set()
                assert closed.wait(WAIT), "the connection was never closed"
            cancel.check(wait=False)  # what ChainWriter does once the call comes back
        finally:
            ended.set()

    rig = Rig(SimpleNamespace(generate=generate))
    rig.session.ask("Explain")
    assert entered.wait(WAIT)
    rig.session.close()
    assert closed.wait(WAIT) and ended.wait(WAIT)
    assert rig.kinds() == ["thinking"]


def test_a_stop_keeps_the_pin_and_the_next_question_is_answered():
    blocking = Blocking()
    rig = Rig(blocking)
    rig.session.ask("Explain")
    assert blocking.entered.wait(WAIT)
    rig.session.stop()
    assert rig.session.available and rig.kinds()[-1] == "stopped"
    blocking.release.set()
    wait_until(lambda: False, timeout=0.2)
    assert rig.session.history == []
    rig.session.close()


def test_a_newer_question_cancels_the_speech_of_the_older_answer():
    rig = Rig()
    rig.ask_and_wait("one", 1)
    before = rig.cancelled_speech
    rig.ask_and_wait("two", 2)
    assert rig.cancelled_speech > before and rig.spoken == ["This is error 0007. Check the connection."] * 2
    rig.session.close()


# -- stale pins (d) -------------------------------------------------------------------------------------


def test_the_model_and_the_answer_card_are_told_how_old_the_pin_is():
    clock = Clock()
    packets = []

    def generate(request, cancel=None):
        packets.append(json.loads(request.text))
        clock.advance(2.0)
        return GenerationResult('{"answer":"ok","uncertain":false}', "m", stop_reason="stop")

    rig = Rig(SimpleNamespace(generate=generate), clock=clock)
    clock.advance(45.0)
    rig.ask_and_wait("first", 1)
    assert packets[0]["observed"]["age_s"] == 45.0
    answer = next(e for e in rig.events if e["kind"] == "answer")
    assert answer["age_s"] == 47.0 and answer["seconds"] == 2.0
    clock.advance(30.0)
    rig.ask_and_wait("second", 2)
    assert packets[1]["observed"]["age_s"] == 77.0  # the follow-up sees the pin has aged, not a fresh one
    rig.session.close()


def test_a_pin_is_not_answered_about_after_the_hard_limit_and_the_session_ends():
    clock = Clock()
    rig = Rig(SimpleNamespace(generate=lambda *a, **k: pytest.fail("an expired pin must not reach the model")), clock=clock)
    clock.advance(HARD_TTL_S)
    rig.session.ask("Explain")
    rig.wait_for("error")
    error = next(e for e in rig.events if e["kind"] == "error")
    assert error == {"kind": "error", "text": EXPIRED, "closed": True}
    assert not rig.session.available and rig.session.history == [] and rig.selection.target is None
    assert not rig.session.ask("again") and rig.spoken == []


def test_a_follow_up_just_inside_the_limit_is_answered_and_one_just_outside_is_not():
    clock = Clock()
    rig = Rig(clock=clock)
    clock.advance(HARD_TTL_S - 0.5)
    rig.ask_and_wait("inside", 1)
    clock.advance(0.5)
    rig.session.ask("outside")
    rig.wait_for("error")
    assert rig.events[-1]["text"] == EXPIRED and rig.count("answer") == 1


def test_pointing_again_forgets_the_old_exchanges_and_answers_about_the_new_item():
    packets = []

    def generate(request, cancel=None):
        packets.append(json.loads(request.text))
        return GenerationResult('{"answer":"ok","uncertain":false}', "m", stop_reason="stop")

    rig = Rig(SimpleNamespace(generate=generate))
    rig.ask_and_wait("about the first item", 1)
    assert len(rig.session.history) == 1
    clock = rig.clock
    clock.advance(100.0)  # the first pin is old; the new one is fresh
    rig.fake.target = type(rig.fake.target)("AXButton", "Retry")
    with using(rig.fake):
        second = capture_point(clock=clock)
    assert rig.session.repoint(second)
    assert rig.session.history == [] and rig.selection.target is None  # the old pin was closed
    rig.ask_and_wait("about the second item", 2)
    assert packets[1]["previous_exchanges"] == [] and packets[1]["observed"]["target"]["label"] == "Retry"
    assert packets[1]["observed"]["age_s"] == 0.0
    rig.session.close()


def test_pointing_again_after_the_session_ended_closes_the_new_selection():
    rig = Rig()
    rig.session.close()
    with using(rig.fake):
        late = capture_point(clock=rig.clock)
    assert rig.session.repoint(late) is False and late.target is None


# -- failures are masked, and nothing is stored ----------------------------------------------------------


def test_a_provider_error_is_masked_and_a_retry_is_allowed():
    outcomes = iter([WriterError(SECRET), None])

    def generate(request, cancel=None):
        failure = next(outcomes)
        if failure:
            raise failure
        return GenerationResult('{"answer":"fine","uncertain":false}', "m", stop_reason="stop")

    rig = Rig(SimpleNamespace(generate=generate))
    rig.session.ask("Explain")
    rig.wait_for("error")
    assert rig.session.failed and rig.session.available and not rig.spoken and not rig.session.history
    assert rig.events[-1] == {"kind": "error", "text": PROVIDER_FAILED, "closed": False}
    assert SECRET not in str(rig.events)
    rig.ask_and_wait("Retry", 1)
    assert not rig.session.failed and rig.spoken == ["fine"]
    rig.session.close()


def test_an_unexpected_exception_shows_a_fixed_sentence_never_its_message():
    def generate(request, cancel=None):
        raise RuntimeError(SECRET)

    rig = Rig(SimpleNamespace(generate=generate))
    rig.session.ask("Explain")
    rig.wait_for("error")
    assert rig.events[-1]["text"] == UNEXPECTED and SECRET not in str(rig.events) and rig.session.available
    rig.session.close()


def test_a_text_only_model_with_an_image_pin_says_so_in_our_own_words(monkeypatch):
    monkeypatch.setattr("glide.assistant.point_ask.writer_vision", lambda: False)
    rig = Rig(SimpleNamespace(generate=lambda *a, **k: pytest.fail("no image may be sent")), with_image=True)
    rig.session.ask("Explain")
    rig.wait_for("error")
    assert "text-only" in rig.events[-1]["text"] and rig.events[-1]["closed"] is False and rig.session.failed
    rig.session.close()


def test_nothing_is_written_or_logged_whatever_is_asked(tmp_path, monkeypatch, caplog):
    """Typed follow-ups, the pinned item's text and the answers exist in memory only (D3)."""
    monkeypatch.chdir(tmp_path)
    caplog.set_level(logging.DEBUG)
    rig = Rig(reply_writer('{"answer":"needle-answer","uncertain":false}'))
    rig.ask_and_wait("needle-question", 1)
    rig.session.close()
    assert [p for p in tmp_path.rglob("*") if p.is_file()] == []
    logged = " ".join(record.getMessage() for record in caplog.records)
    assert "needle" not in logged and "0007" not in logged

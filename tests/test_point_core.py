"""PointMode and the pet's point events: what is disclosed before anything is read, what is shown, how it ends."""

from __future__ import annotations

import json
import logging
import threading
from types import SimpleNamespace

import pytest
from point_fakes import GOOD, SyntheticDesktop, reply_writer
from test_assistant_fakes import WAIT, wait_until
from test_pet_core import FakeLoop, PetConfig, make_core
from test_point_session import Blocking

from glide.assistant.point_ask import PointUnavailable, capture_point
from glide.assistant.point_voice import PointAssistant
from glide.computer.generation import GenerationResult
from glide.computer.platform_adapter import using
from glide.computer.point_types import PointTarget
from glide.providers.config import ConfigError
from glide.ui.core import PetEvent, PetView
from glide.ui.point_core import NO_PROVIDER, TEXT_ONLY, UNREADABLE, PointMode

SECRET = "selected secret and private provider body"


class Rig:
    """A `PointMode` over a synthetic desktop, a config whose writer is fake, and fakes for the voice hooks."""

    def __init__(self, writer=None, *, delay=0.0, capture=None, vision=lambda: True, voice=True):
        self.fake = SyntheticDesktop()
        self.config = PetConfig(llm=None, writer=writer if writer is not None else reply_writer())
        self.events: list[dict] = []
        self.captured: list[dict] = []
        self.voice_started: list = []
        self.voice_ended = 0

        def real_capture(**options):
            self.captured.append(options)
            return capture_point(**options)

        def start_voice(factory, bind):
            assistant = factory(self.config, io=None)
            bind(assistant)
            self.voice_started.append(assistant)
            return voice

        def end_voice():
            self.voice_ended += 1

        self.mode = PointMode(
            self.config,
            lambda kind, **data: self.events.append({"kind": kind, **data}),
            start_voice=start_voice,
            end_voice=end_voice,
            capture=capture or real_capture,
            delay_s=delay,
            vision=vision,
        )

    def kinds(self):
        return [e["kind"] for e in self.events]

    def wait_for(self, kind, n=1):
        assert wait_until(lambda: self.kinds().count(kind) >= n), f"no {kind}: {self.kinds()}"

    def settle(self):
        assert wait_until(lambda: not self.mode._cancel), "the pin worker never finished"

    def start(self, *args, **kwargs):
        with using(self.fake):
            self.mode.start(*args, **kwargs)
            self.settle()


def test_a_local_preview_reads_the_item_and_sends_nothing_anywhere():
    rig = Rig(writer=SimpleNamespace(generate=lambda *a, **k: pytest.fail("a preview must not reach a model")))
    rig.start("Explain", share=False, with_image=True, voice=True)  # image and voice need sharing: both are ignored
    assert rig.kinds() == ["status", "selecting", "selected", "preview"]
    assert "Nothing is sent to a model" in rig.events[0]["text"]
    assert rig.events[2] == {"kind": "selected", "point": [200.0, 150.0], "text": "錯誤 0007: connection failed"}
    assert "label: 錯誤 0007: connection failed" in rig.events[3]["text"] and "role: AXStaticText" in rig.events[3]["text"]
    assert rig.captured == [{"with_image": False}] and rig.voice_started == [] and not rig.mode.holding
    assert rig.fake.calls == [("pointer",), ("target", (200.0, 150.0))]


def test_what_will_be_shared_is_said_before_anything_is_read():
    rig = Rig()
    rig.start("What is this?", share=True, with_image=True)
    kinds = rig.kinds()
    assert kinds[:4] == ["status", "status", "selecting", "selected"]
    assert rig.events[0]["text"].startswith("Answer provider: ") and "SimpleNamespace" in rig.events[0]["text"]
    assert rig.events[1]["text"] == "This request shares your question, the pointed item's text and a small image crop."
    rig.wait_for("answer")
    answer = next(e for e in rig.events if e["kind"] == "answer")
    assert answer["text"] == "This is error 0007. Check the connection." and answer["age_s"] >= 0 and rig.mode.holding
    assert rig.captured == [{"with_image": True}]
    rig.mode.close()


def test_an_unusable_provider_or_a_text_only_model_stops_before_the_screen_is_read():
    rig = Rig()
    rig.config._writer = None  # nothing usable in the chains
    rig.start("Explain", share=True)
    assert rig.events == [{"kind": "error", "text": NO_PROVIDER, "closed": True}] and rig.captured == [] and rig.fake.calls == []
    text_only = Rig(vision=lambda: False)
    text_only.start("Explain", share=True, with_image=True)
    assert text_only.events == [{"kind": "error", "text": TEXT_ONLY, "closed": True}]
    assert text_only.captured == [] and text_only.fake.calls == []


def test_a_follow_up_goes_to_the_same_pin_and_pointing_again_replaces_it_and_forgets_the_exchanges():
    packets = []

    def generate(request, cancel=None):
        packets.append(json.loads(request.text))
        return GenerationResult(GOOD, "m", stop_reason="stop")

    rig = Rig(SimpleNamespace(generate=generate))
    rig.start("first", share=True)
    rig.wait_for("answer")
    assert rig.mode.ask("second")
    rig.wait_for("answer", 2)
    assert [len(p["previous_exchanges"]) for p in packets] == [0, 1]
    session = rig.mode._session
    rig.fake.target = PointTarget("AXButton", "Retry")
    rig.start("third", share=True)
    rig.wait_for("answer", 3)
    assert rig.mode._session is session  # the same session: re-pointed, not rebuilt
    assert packets[2]["previous_exchanges"] == [] and packets[2]["observed"]["target"]["label"] == "Retry"
    assert rig.kinds().count("selected") == 2
    rig.mode.close()


def test_a_question_with_no_pin_says_to_select_a_point_first():
    rig = Rig()
    assert rig.mode.ask("hello") is False
    assert rig.events == [{"kind": "status", "text": "Select a point first."}]


def test_closing_during_the_countdown_cancels_it_and_reads_nothing():
    rig = Rig(delay=60.0)
    with using(rig.fake):
        rig.mode.start("Explain", share=True)
        rig.wait_for("selecting")
        assert rig.mode.active
        started = threading.Event()
        threading.Thread(target=lambda: (rig.mode.close(), started.set())).start()
        assert started.wait(WAIT)  # returned at once, not after sixty seconds
        rig.settle()
    assert rig.captured == [] and rig.fake.calls == [] and rig.kinds()[-1] == "closed" and not rig.mode.active


def test_pressing_ask_twice_during_the_countdown_starts_one_pin():
    rig = Rig(delay=60.0)
    with using(rig.fake):
        rig.mode.start("one", share=True)
        rig.mode.start("two", share=True)
        rig.wait_for("selecting")
        rig.mode.close()
        rig.settle()
    assert rig.kinds().count("selecting") == 1


@pytest.mark.parametrize(
    "raised,expected",
    [
        (PointUnavailable("Protected fields cannot be read or captured."), "Protected fields cannot be read or captured."),
        (RuntimeError(SECRET), UNREADABLE),
    ],
)
def test_a_capture_that_fails_says_only_a_fixed_sentence(raised, expected):
    def capture(**options):
        raise raised

    rig = Rig(capture=capture)
    rig.start("Explain", share=False)
    assert rig.events[-1] == {"kind": "error", "text": expected, "closed": True}
    assert SECRET not in str(rig.events) and not rig.mode.holding


def test_stop_cancels_the_answer_and_keeps_the_pin():
    blocking = Blocking()
    rig = Rig(blocking)
    rig.start("Explain", share=True)
    assert blocking.entered.wait(WAIT)
    rig.mode.stop()
    assert rig.kinds()[-1] == "stopped" and rig.mode.holding
    blocking.release.set()
    wait_until(lambda: False, timeout=0.2)
    assert "answer" not in rig.kinds()
    rig.mode.close()


def test_a_voice_pin_binds_the_session_to_the_assistant_before_listening_and_ends_the_microphone_with_the_pin():
    rig = Rig()
    rig.start("", share=True, voice=True)
    (assistant,) = rig.voice_started
    assert isinstance(assistant, PointAssistant) and assistant._session is rig.mode._session
    assert "thinking" not in rig.kinds()  # no question was given: it waits to be spoken to
    assert rig.mode.holding and rig.voice_ended == 0
    rig.mode._session_event("error", text="The pinned item expired. Select the point again.", closed=True)
    assert rig.voice_ended == 1
    rig.mode._session_event("error", text="again", closed=True)
    assert rig.voice_ended == 1  # once
    rig.mode.close()


def test_a_typed_pin_does_not_end_a_microphone_it_never_had():
    rig = Rig()
    rig.start("Explain", share=True)
    rig.mode._session_event("error", text="The pinned item expired.", closed=True)
    assert rig.voice_ended == 0
    rig.mode.close()


def test_asking_by_voice_while_a_typed_pin_is_held_replaces_it_with_a_spoken_session():
    rig = Rig()
    rig.start("Explain", share=True)
    rig.wait_for("answer")
    typed = rig.mode._session
    rig.start("", share=True, voice=True)
    assert typed.available is False and rig.mode._session is not typed and len(rig.voice_started) == 1
    rig.mode.close()


def test_a_voice_stack_that_cannot_start_leaves_no_pin_held():
    rig = Rig(voice=False)
    rig.start("", share=True, voice=True)
    assert not rig.mode.holding and "selected" in rig.kinds()


def test_a_voice_stack_that_cannot_start_takes_the_pin_off_the_view_too():
    """PR15-4175491845: `selected` had drawn the pin and 'Pinned: ...', then nothing said it was gone, though `holding` was False."""
    rig = Rig(voice=False)
    rig.start("", share=True, voice=True)
    rig.settle()
    assert not rig.mode.holding and rig.kinds()[-1] == "closed"
    view = PetView()
    for e in rig.events:
        view.apply(PetEvent("point", e))
    assert view.pin is None and view.target == ""


def test_nothing_is_written_or_logged(tmp_path, monkeypatch, caplog):
    monkeypatch.chdir(tmp_path)
    caplog.set_level(logging.DEBUG)
    rig = Rig(reply_writer('{"answer":"needle-answer","uncertain":false}'))
    rig.start("needle-question", share=True)
    rig.wait_for("answer")
    rig.mode.close()
    assert [p for p in tmp_path.rglob("*") if p.is_file()] == []
    assert "needle" not in " ".join(r.getMessage() for r in caplog.records)


# -- through the pet's core and its view ---------------------------------------------------------------------


def pet_with_point(tmp_path, **options):
    fake = SyntheticDesktop()
    core, config = make_core(tmp_path, capture=capture_point, point_delay_s=0.0, **options)
    config._writer = reply_writer()
    return core, config, fake


def test_point_events_reach_the_pet_as_point_events_and_the_view_pins_and_unpins(tmp_path):
    core, _, fake = pet_with_point(tmp_path)
    view = PetView()
    with using(fake):
        core.point.start("What is this?", share=True)
        assert wait_until(lambda: not core.point._cancel)
    assert wait_until(lambda: core.point.holding and core.point._session.history)
    for event in core.drain():
        assert event.type == "point"
        view.apply(event)
    assert view.pin == (200.0, 150.0) and view.target == "Pinned: 錯誤 0007: connection failed"
    assert view.answer == "This is error 0007. Check the connection." and "snapshot taken" in view.answer_note
    assert view.provider.startswith("Answer provider:") and view.mood == "happy" and view.status == "Answer ready"
    assert any(line.startswith("point answer: fixture | ") for line in view.lines)
    core.stop()
    for event in core.drain():
        view.apply(event)
    assert view.pin is None and view.target == "" and not core.point.active
    core.close()


def test_the_view_shows_what_the_provider_never_said_and_keeps_markup_as_text():
    view = PetView()
    view.apply(
        PetEvent(
            "point", {"kind": "answer", "text": "<b>x</b>" * 2000, "uncertain": True, "age_s": 47.4, "model": "m", "seconds": 1.5}
        )
    )
    assert len(view.answer) == 4096 and view.answer.startswith("<b>x</b>") and view.mood == "question"
    assert view.status == "More context needed · ask a follow-up" and "47 s ago" in view.answer_note
    view.apply(PetEvent("point", {"kind": "selected", "point": [1, 2], "text": "<i>" + "t" * 2000}))
    assert view.pin == (1, 2) and len(view.target) == len("Pinned: ") + 1024
    view.apply(PetEvent("point", {"kind": "error", "text": "Please try again.", "closed": False}))
    assert view.pin == (1, 2) and view.answer == "Please try again."  # a retryable error keeps the pin
    view.apply(PetEvent("point", {"kind": "error", "text": "expired", "closed": True}))
    assert view.pin is None
    view.apply(PetEvent("point", {"kind": "answer", "text": 7}))  # malformed: ignored
    assert view.answer == "expired"


def test_a_point_voice_session_uses_the_pets_voice_slot_and_stop_ends_it(tmp_path):
    loops = []

    def factory(config, settings, *, io, act, assistant_factory, **_):
        loop = FakeLoop(assistant_factory(config, io=io))
        loops.append((loop, act))
        return loop

    core, _, fake = pet_with_point(tmp_path, voice_factory=factory)
    with using(fake):
        core.point.start("", share=True, voice=True)
        assert wait_until(lambda: core.point.holding)
    ((loop, act),) = loops
    assert isinstance(loop.assistant, PointAssistant) and act is False  # reading a point never acts
    assert loop.calls == ["start"] and core.voice_active
    core.stop()
    assert wait_until(lambda: loop.calls[-1] == "stop")
    assert not core.voice_active and not core.point.active
    core.close()


def test_dismissing_the_voice_bar_during_the_countdown_means_the_microphone_never_opens(tmp_path):
    """PR15-4175616385: pause_voice only knew a session or one being built, so during the point countdown it did nothing and
    the microphone opened after the countdown for a bar that had been dismissed."""
    built = []

    def factory(config, settings, *, io, act, assistant_factory, **_):
        built.append(1)
        return FakeLoop(assistant_factory(config, io=io))

    core, config = make_core(tmp_path, capture=capture_point, point_delay_s=30.0, voice_factory=factory)
    config._writer = reply_writer()
    with using(SyntheticDesktop()):
        core.point.start("", share=True, voice=True)
        assert wait_until(lambda: core.point.active)
        core.pause_voice()  # the bar's "done" while the countdown runs
        assert wait_until(lambda: not core.point.active)
    assert built == [] and not core.voice_active
    assert not any(e.type == "mic" and e.data.get("open") for e in core.drain())
    core.close()


def test_pausing_does_not_cancel_a_countdown_that_was_not_for_the_microphone(tmp_path):
    core, config = make_core(tmp_path, capture=capture_point, point_delay_s=30.0)
    config._writer = reply_writer()
    with using(SyntheticDesktop()):
        core.point.start("", share=True)
        assert wait_until(lambda: core.point.active)
        core.pause_voice()
        assert core.point.active
    core.close()


def test_a_point_voice_cannot_start_while_an_ordinary_voice_session_is_open(tmp_path):
    def factory(config, settings, *, io, act, assistant_factory, **_):
        return FakeLoop(assistant_factory(config, io=io))

    core, _, fake = pet_with_point(tmp_path, voice_factory=factory)
    core.start_voice()
    assert wait_until(lambda: core.voice_active and core._voice is not None)
    with using(fake):
        core.point.start("", share=True, voice=True)
        assert wait_until(lambda: not core.point._cancel)
    notices = [e.data["message"] for e in core.drain() if e.type == "notice"]
    assert any("Stop the current voice session" in m for m in notices) and not core.point.holding
    core.close()


def test_a_stop_while_a_point_voice_session_is_still_opening_wins_and_nothing_stays_open(tmp_path):
    entered, release, loops = threading.Event(), threading.Event(), []

    def factory(config, settings, *, io, act, assistant_factory, **_):
        entered.set()
        assert release.wait(WAIT)
        loop = FakeLoop(assistant_factory(config, io=io))
        loops.append(loop)
        return loop

    core, _, fake = pet_with_point(tmp_path, voice_factory=factory)
    with using(fake):
        core.point.start("", share=True, voice=True)
        assert entered.wait(WAIT)
        core.stop()
        release.set()
        assert wait_until(lambda: loops and not core.point._cancel and not core._opening)
    (loop,) = loops
    assert loop.calls == ["stop"] and not core.voice_active and not core.point.holding
    assert not any(e.type == "mic" and e.data.get("open") for e in core.drain())
    core.close()


def test_a_slot_that_cannot_be_set_up_is_reported_as_that_not_as_a_permission_problem():
    rig = Rig()
    rig.config.secret = SECRET

    def broken(timeout=None):
        raise ConfigError(f"[llm.smart] local cannot be set up: bad option {SECRET}")

    rig.config.writer = broken
    rig.start("Explain", share=True)
    (event,) = rig.events
    assert event["kind"] == "error" and event["closed"] is True and "llm.smart" in event["text"] and "bad option" in event["text"]
    assert SECRET not in event["text"] and "permission" not in event["text"] and rig.captured == []


def test_an_image_crop_off_the_primary_display_says_so_instead_of_blaming_permissions():
    rig = Rig(capture=lambda **options: capture_point((5000, 5000), **options))
    rig.start("Explain", share=True, with_image=True)
    assert rig.events[-1] == {
        "kind": "error",
        "text": "The image crop works on the primary display only; point again.",
        "closed": True,
    }


def test_pointing_again_as_the_session_ends_is_a_failed_open_that_unpins_the_view():
    """audit2 finding 10: repoint() returned False and the new pin stayed drawn with nothing behind it."""
    rig = Rig()
    rig.start("first", share=True)
    rig.wait_for("answer")
    session = rig.mode._session
    asked = []
    session.ask = lambda question: asked.append(question) or False

    def ended_meanwhile(selection):
        session.close()
        selection.close()
        return False

    session.repoint = ended_meanwhile
    rig.start("second", share=True)
    rig.wait_for("closed")
    assert not rig.mode.holding and asked == []
    rig.mode.close()

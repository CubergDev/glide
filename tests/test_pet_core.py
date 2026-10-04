"""PetCore and PetView: the pet's adapter to the assistant, over the assistant tests' fakes. No Qt, no devices.

What these pin down: events follow PROTOCOL.md's names, a provider switch is always an event, a stop returns at
once, recording is opt-in (D3) and reaches the core, computer actions are off until asked for, and the pet writes
no file and keeps no text of its own.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
from test_assistant_fakes import WAIT, FakeConfig, FakeLLM, FakePlayer, FakeSTT, FakeTTS, route_json, wait_until

from glide.assistant.audio_io import chunked
from glide.assistant.tasks import OUTCOME_PHRASES, TaskResult
from glide.computer import runner
from glide.computer.platform_adapter import desktop
from glide.computer.runner import RunState
from glide.providers.chain import SwitchEvent
from glide.speech.audio import FullDuplexDevice
from glide.speech.session import build_voice
from glide.speech.settings import SpeechSettings
from glide.ui.core import MAX_TEXT, PetAssistant, PetCore, PetEvent, PetView, provider_lines

SECRET = "sk-SECRET-1234567890"


def tone(level: int, seconds: float) -> bytes:
    from array import array

    return array("h", [level if i % 2 == 0 else -level for i in range(int(seconds * 16000))]).tobytes()


class PetConfig(FakeConfig):
    """FakeConfig plus what the pet reads from a real GlideConfig: the speech settings and the slots."""

    voice = SpeechSettings()  # what a real GlideConfig holds for the voice stack (`config.speech` is the providers' view)

    def slots(self, role):
        return [SimpleNamespace(name=f"{role}-a", state="ready"), SimpleNamespace(name=f"{role}-b", state="skipped")]


def make_core(tmp_path, llm=None, **options):
    llm = llm if llm is not None else FakeLLM(route=route_json("answer", reply="Hello there."))
    config = PetConfig(llm=llm, tts=FakeTTS(), secret=SECRET)
    return PetCore(config, runs_dir=tmp_path / "runs", **options), config


def events_until(core, predicate, timeout=WAIT):
    """Drain until `predicate(every event so far)`; returns every event the core has made since this helper first saw it."""
    seen = core.__dict__.setdefault("seen", [])

    def done():
        seen.extend(core.drain())
        return predicate(seen)

    wait_until(done, timeout)
    return list(seen)


def types(events):
    return [(e.type, e.data.get("assistant") or e.data.get("phase") or e.data.get("role") or "") for e in events]


# -- text -------------------------------------------------------------------------------------------


def test_a_typed_answer_is_thinking_then_the_words_then_idle(tmp_path):
    core, _ = make_core(tmp_path)
    assert core.send_text("  hello  ")
    seen = events_until(core, lambda es: types(es)[-1:] == [("state", "idle")])
    assert types(seen) == [("transcript", "user"), ("state", "thinking"), ("transcript", "assistant"), ("state", "idle")]
    assert seen[0].data == {"role": "user", "text": "hello", "partial": False}
    assert seen[2].data["text"] == "Hello there."
    core.close()


@pytest.mark.parametrize("text", ["", "   ", "x" * (MAX_TEXT + 1)])
def test_an_empty_or_oversized_request_is_not_sent(tmp_path, text):
    core, config = make_core(tmp_path)
    assert not core.send_text(text)
    assert core.drain() == [] and config.calls.llm == 0
    core.close()


def test_a_computer_request_is_a_dry_run_until_actions_are_allowed(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(
        runner,
        "run",
        lambda cfg, f, classifier_factory=None, control=None: (
            seen.append((cfg.act, cfg.record_content)),
            RunState(outcome="dry run"),
        )[1],
    )
    monkeypatch.setattr(desktop, "accessibility_trusted", lambda: True)
    route = route_json("computer", reply="On it.", goal="open the calendar")
    core, _ = make_core(tmp_path, llm=FakeLLM(route=route))
    assert core.act is False
    core.send_text("open the calendar")
    events = events_until(core, lambda es: ("task", "completed") in types(es))
    assert seen == [(False, False)]  # act off, and content recording off: both defaults reach the runner
    assert ("task", "started") in types(events) and ("state", "acting") in types(events)
    done = next(e for e in events if e.type == "task" and e.data["phase"] == "completed")
    assert done.data["outcome"] == "dry run" and done.data["act"] is False
    events_until(core, lambda es: types(es)[-1:] == [("state", "idle")])
    core.act = True
    core.send_text("open the calendar again")
    events_until(core, lambda es: types(es).count(("task", "completed")) == 2)
    assert seen[-1] == (True, False)
    core.close()


def test_a_failed_task_is_reported_as_failed_and_a_stopped_one_as_stopped(tmp_path, monkeypatch):
    outcomes = iter([RunState(outcome="provider failure"), RunState(outcome="aborted (stopped by the user)")])
    monkeypatch.setattr(runner, "run", lambda cfg, f, classifier_factory=None, control=None: next(outcomes))
    core, _ = make_core(tmp_path, llm=FakeLLM(route=route_json("computer", goal="g")))
    core.send_text("do it")
    events_until(core, lambda es: any(e.type == "task" and e.data["phase"] == "failed" for e in es))
    assert wait_until(lambda: not core.busy)
    core.send_text("again")
    seen = events_until(core, lambda es: any(e.type == "task" and e.data["phase"] == "stopped" for e in es))
    assert [e.data["phase"] for e in seen if e.type == "task"] == ["started", "failed", "started", "stopped"]
    core.close()


class _EndedTask:
    """A computer task that has already ended with `result`."""

    act = True

    def __init__(self, result) -> None:
        self.result = result

    def wait(self, timeout=None) -> bool:
        return True


@pytest.mark.parametrize(
    ("result", "phase"),
    [
        (TaskResult("g", True, "done", achieved=True), "completed"),
        (TaskResult("g", False, "dry run"), "completed"),
        (TaskResult("g", True, "done", stopped=True), "stopped"),
        (TaskResult("g", True, "done", uncertain=True), "uncertain"),  # a write whose effect was never seen
        (TaskResult("g", True, "nothing helps", achieved=False), "failed"),
        (TaskResult("g", True, "done", achieved=False), "failed"),
        *[(TaskResult("g", True, outcome), "failed") for outcome in OUTCOME_PHRASES if outcome != "done"],
        (TaskResult("g", True, "aborted (something new)"), "failed"),  # an outcome nobody listed is not a success
    ],
)
def test_only_a_done_and_achieved_task_is_reported_as_completed(result, phase):
    """PR15-4175491842: stalled, stuck, step limit, low confidence, nothing helps and an unverified write were all 'completed'."""
    events = []
    assistant = PetAssistant(PetConfig(llm=FakeLLM()), report=events.append)
    assistant._await_task(_EndedTask(result))
    task = next(e for e in events if e.type == "task")
    assert task.data["phase"] == phase
    view = PetView()
    view.apply(task)
    assert (view.mood == "happy") == (phase == "completed")
    if phase == "uncertain":
        assert view.mood == "question" and "nothing will be repeated" in view.status


def test_stop_returns_at_once_while_a_task_is_running_and_the_task_ends_stopped(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()

    def blocked(cfg, f, classifier_factory=None, control=None):
        entered.set()
        release.wait(WAIT)
        return RunState(outcome="aborted (stopped by the user)")

    monkeypatch.setattr(runner, "run", blocked)
    core, _ = make_core(tmp_path, llm=FakeLLM(route=route_json("computer", goal="g")))
    core.send_text("do it")
    assert entered.wait(WAIT)
    assert core.busy
    core.stop()  # must not wait for the loop
    assert any(e.type == "state" and e.data["assistant"] == "idle" for e in core.drain())
    release.set()
    events_until(core, lambda es: any(e.type == "task" and e.data["phase"] == "stopped" for e in es))
    assert wait_until(lambda: not core.busy)
    core.close()


def test_a_provider_failure_is_an_error_state_and_a_notice_without_the_key(tmp_path):
    from glide.providers.errors import ProviderError

    llm = FakeLLM(route=route_json("answer"), deltas=["partial"])
    llm.stream_error, llm.stream_error_after = ProviderError(f"boom {SECRET}", kind="auth"), 0
    core, _ = make_core(tmp_path, llm=llm)
    core.send_text("hello")
    events = events_until(core, lambda es: any(e.type == "state" and e.data["assistant"] == "error" for e in es))
    notices = [e.data["message"] for e in events if e.type == "notice"]
    assert notices and all(SECRET not in m for m in notices)
    core.close()


# -- fallbacks are visible --------------------------------------------------------------------------


def test_every_provider_switch_is_an_event_with_the_key_scrubbed_and_the_reason_bounded(tmp_path):
    core, config = make_core(tmp_path)
    (listener,) = config.listeners
    listener(SwitchEvent("llm.fast", "a", "b", "timeout", f"slow {SECRET} " + "x" * 400))
    listener(SwitchEvent("stt", "c", None, "auth", "refused"))
    first, second = core.drain()
    assert first.type == "switch" and first.data["to_slot"] == "b" and first.data["kind"] == "timeout"
    assert SECRET not in first.data["reason"] and "***" in first.data["reason"] and len(first.data["reason"]) <= 160
    assert second.data["to_slot"] is None
    view = PetView()
    for event in (first, second):
        view.apply(event)
    assert view.lines[-1] == "fallback: stt c -> nothing left (auth: refused)"
    core.close()
    assert config.listeners == []  # closing stops listening


# -- recording is opt-in (D3) -----------------------------------------------------------------------


def test_recording_is_off_by_default_even_if_the_config_object_says_otherwise(tmp_path):
    config = PetConfig(llm=FakeLLM())
    config.record_content = True
    core = PetCore(config, runs_dir=tmp_path)
    assert config.record_content is False and core.record_content is False
    assert core.drain() == []  # nothing to show: there is no indicator when nothing is recorded
    core.close()


def test_recording_can_be_opted_into_at_start_or_later_and_each_change_is_an_event(tmp_path):
    config = PetConfig(llm=FakeLLM())
    core = PetCore(config, runs_dir=tmp_path, record_content=True)
    assert config.record_content is True
    assert [(e.type, e.data) for e in core.drain()] == [("recording", {"on": True})]
    core.record_content = False
    core.record_content = True
    assert [e.data["on"] for e in core.drain()] == [False, True] and config.record_content is True
    view = PetView()
    view.apply(PetEvent("recording", {"on": True}))
    assert view.recording and view.lines[-1] == "recording content: ON"
    core.close()


def test_the_pet_writes_nothing_and_keeps_no_text_when_recording_is_off(tmp_path, monkeypatch):
    """A typed request, its answer, a switch and a failure leave no file anywhere under the working folders."""
    monkeypatch.chdir(tmp_path)
    needle = "needle-4711 private request"
    core, config = make_core(tmp_path, llm=FakeLLM(route=route_json("answer", reply="needle-4711 private answer.")))
    config.listeners[0](SwitchEvent("llm.fast", "a", "b", "timeout", "slow"))
    core.send_text(needle)
    events_until(core, lambda es: types(es)[-1:] == [("state", "idle")])
    core.close()
    assert [p for p in tmp_path.rglob("*") if p.is_file()] == []


# -- voice ------------------------------------------------------------------------------------------


class FakeLoop:
    """The `VoiceLoop` surface the pet uses: start, pause, resume, stop, and the assistant it owns."""

    def __init__(self, assistant) -> None:
        self.assistant = assistant
        self.calls: list[str] = []
        self.on_failure = None  # set by the pet, called by a real loop when it ends on its own

    def start(self):
        self.calls.append("start")

    def pause(self):
        self.calls.append("pause")

    def resume(self):
        self.calls.append("resume")

    def stop(self):
        self.calls.append("stop")


def voice_core(tmp_path, **options):
    built = []

    def factory(config, settings, *, io, act, assistant_factory, **_):
        loop = FakeLoop(assistant_factory(config, io=io))
        built.append(SimpleNamespace(loop=loop, settings=settings, act=act, io=io))
        return loop

    core, config = make_core(tmp_path, voice_factory=factory, **options)
    return core, config, built


def test_voice_builds_once_then_pauses_and_resumes_the_same_loop(tmp_path):
    core, _, built = voice_core(tmp_path)
    core.headset, core.silence_ms = True, 900
    core.start_voice()
    events = events_until(core, lambda es: ("mic", "") in types(es))
    assert events[0].type == "mic" and events[0].data == {"open": True}
    (one,) = built
    assert one.settings.headset is True and one.settings.silence_ms == 900 and one.act is False
    assert isinstance(one.loop.assistant, PetAssistant)
    core.pause_voice()
    assert core.drain()[0].data == {"open": False, "detail": "paused"}
    core.start_voice()
    assert core.drain()[0].data == {"open": True}
    assert len(built) == 1 and one.loop.calls == ["start", "pause", "resume"]
    core.close()
    assert one.loop.calls[-1] == "stop"


def test_a_real_config_gives_the_voice_stack_its_own_settings_not_the_providers_view(tmp_path):
    """PR15-4175491833: the pet read `config.speech` (the providers' table, silence_ms None by default) where the voice stack
    takes `config.voice`; the fake config here had `speech` shaped like the voice settings and hid it."""
    from glide.providers.config import GlideConfig

    config = GlideConfig.from_toml('[speech]\nsilence_ms = 800\nvad = "energy"\n', env={})
    built = []

    def factory(config, settings, *, io, act, assistant_factory, **_):
        built.append(settings)
        return FakeLoop(assistant_factory(config, io=io))

    core = PetCore(config, runs_dir=tmp_path / "runs", voice_factory=factory)
    assert core.silence_ms == 800
    core.headset = True
    core.start_voice()
    events_until(core, lambda es: ("mic", "") in types(es))
    assert isinstance(built[0], SpeechSettings) and (built[0].headset, built[0].silence_ms, built[0].vad) == (True, 800, "energy")
    core.close()


def test_a_voice_stack_that_cannot_start_says_what_is_missing_and_never_the_key(tmp_path):
    def factory(config, settings, **kw):
        raise RuntimeError(f"sounddevice is not installed ({SECRET})")

    core, _ = make_core(tmp_path, voice_factory=factory)
    core.start_voice()
    events = events_until(core, lambda es: any(e.type == "notice" for e in es))
    assert ("mic", "") in types(events)
    assert events[0].data == {"open": False, "detail": "unavailable"}
    message = next(e.data["message"] for e in events if e.type == "notice")
    assert "sounddevice" in message and SECRET not in message and "RuntimeError" in message
    assert not core.voice_active
    core.close()


def test_stopping_a_voice_session_stops_the_work_ends_the_loop_and_returns_at_once(tmp_path):
    core, _, built = voice_core(tmp_path)
    core.start_voice()
    events_until(core, lambda es: ("mic", "") in types(es))
    (one,) = built
    core.stop()
    assert wait_until(lambda: one.loop.calls[-1] == "stop")
    events = core.drain()
    assert {"open": False, "detail": "stopped"} in [e.data for e in events if e.type == "mic"]
    assert not core.voice_active
    core.close()


def test_typed_text_during_a_voice_session_goes_to_the_voice_assistant(tmp_path):
    core, _, built = voice_core(tmp_path)
    core.start_voice()
    events_until(core, lambda es: ("mic", "") in types(es))
    (one,) = built
    assert core._assistant() is one.loop.assistant
    core.close()


def test_the_assistant_factory_passes_on_whatever_options_the_voice_stack_gives_it(tmp_path):
    core, config = make_core(tmp_path)
    assistant = core._make_assistant(config, history_turns=1)
    assert isinstance(assistant, PetAssistant) and assistant._history.maxlen == 2
    core.close()


def test_closing_twice_is_harmless(tmp_path):
    core, config = make_core(tmp_path)
    core.close()
    core.close()
    assert config.listeners == []


def test_what_is_heard_is_shown_as_it_arrives_and_a_spoken_answer_is_speaking(tmp_path):
    core, config, built = voice_core(tmp_path)
    config._stt = FakeSTT(final="hello glide", partials=["hello"])
    core.start_voice()
    events_until(core, lambda es: ("mic", "") in types(es))
    assistant = built[0].loop.assistant
    assistant.io.player = FakePlayer()
    assistant.handle_audio(chunked(tone(2000, 0.5)), wait=False)
    seen = core.drain()
    partial = next(e for e in seen if e.type == "transcript")
    assert partial.data == {"role": "user", "text": "hello", "partial": True}
    assert ("transcript", "user") in types(seen) and ("state", "listening") in types(seen)
    assert ("state", "thinking") in types(seen) and ("state", "speaking") in types(seen)
    assert types(seen)[-1] == ("state", "idle")
    core.close()


def blocked_voice(tmp_path, **options):
    """A core whose voice factory waits for the test, so a stop can arrive while the stack is still being built."""
    entered, release, loops = threading.Event(), threading.Event(), []

    def factory(config, settings, *, io, act, assistant_factory, **_):
        entered.set()
        assert release.wait(WAIT), "the test never released the voice factory"
        loop = FakeLoop(assistant_factory(config, io=io))
        loops.append(loop)
        return loop

    core, config = make_core(tmp_path, voice_factory=factory, **options)
    return core, config, entered, release, loops


@pytest.mark.parametrize("how", ["stop", "pause", "close"])
def test_a_stop_while_the_voice_stack_is_still_opening_wins_and_the_microphone_never_starts(tmp_path, how):
    core, _, entered, release, loops = blocked_voice(tmp_path)
    core.start_voice()
    assert entered.wait(WAIT) and core.voice_active
    {"stop": core.stop, "pause": core.pause_voice, "close": core.close}[how]()
    release.set()
    assert wait_until(lambda: loops and not core._opening)
    (loop,) = loops
    assert loop.calls == ["stop"]  # built, so ended; never started, so never listening
    assert not core.voice_active
    events = core.drain()
    assert not any(e.type == "mic" and e.data.get("open") for e in events)  # nothing ever said the microphone was open
    if how != "close":
        core.start_voice()  # and the slot is free for the next try
        assert wait_until(lambda: len(loops) == 2 and core.voice_active and core._voice is not None)
    core.close()


def test_a_stop_during_the_build_is_known_to_the_stack_before_it_opens_the_microphone(tmp_path):
    """PR15-4175491836: build_voice opens the device itself, so the epoch check after the build came after the microphone
    was already open. The factory is now given `ready()`, to ask just before it opens anything."""
    entered, release, asked = threading.Event(), threading.Event(), []

    def factory(config, settings, *, io, act, assistant_factory, ready):
        entered.set()
        assert release.wait(WAIT)
        asked.append(ready())  # what build_voice does right before device.start()
        return FakeLoop(assistant_factory(config, io=io))

    core, _ = make_core(tmp_path, voice_factory=factory)
    core.start_voice()
    assert entered.wait(WAIT)
    core.stop()
    release.set()
    assert wait_until(lambda: asked and not core._opening)
    assert asked == [False]
    core.start_voice()  # a later session is ready
    assert wait_until(lambda: len(asked) == 2 and core.voice_active)
    assert asked == [False, True]
    core.close()


def test_a_voice_loop_that_ends_on_its_own_closes_the_microphone_in_the_view_and_frees_the_slot(tmp_path):
    """PR15-4175491828: nothing watched the loop thread, so a device fault left the view saying the microphone was live and
    start_voice resumed a dead loop."""
    core, _, built = voice_core(tmp_path)
    core.start_voice()
    events_until(core, lambda es: ("mic", "") in types(es))
    loop = built[0].loop
    core.drain()
    loop.on_failure("Microphone overflow; the incomplete command was discarded.")
    assert wait_until(lambda: loop.calls[-1:] == ["stop"])
    assert not core.voice_active
    assert [e.data for e in core.drain() if e.type == "mic"] == [{"open": False, "detail": "failed"}]
    view = PetView()
    view.apply(PetEvent("mic", {"open": True}))
    view.apply(PetEvent("mic", {"open": False, "detail": "failed"}))
    assert not view.mic
    core.start_voice()  # a new session is built, not the dead one resumed
    assert wait_until(lambda: len(built) == 2 and core.voice_active)
    assert "resume" not in loop.calls
    core.close()


def test_a_failure_of_a_loop_that_was_already_replaced_or_stopped_is_ignored(tmp_path):
    core, _, built = voice_core(tmp_path)
    core.start_voice()
    events_until(core, lambda es: ("mic", "") in types(es))
    core.stop()
    core.drain()
    built[0].loop.on_failure("late")
    assert core.drain() == []
    core.close()


def test_a_voice_stack_that_fails_after_a_stop_says_nothing(tmp_path):
    entered, release = threading.Event(), threading.Event()

    def factory(config, settings, **kw):
        entered.set()
        release.wait(WAIT)
        raise RuntimeError("no microphone")

    core, _ = make_core(tmp_path, voice_factory=factory)
    core.start_voice()
    assert entered.wait(WAIT)
    core.stop()
    release.set()
    assert wait_until(lambda: not core._opening)
    assert not any(e.type == "notice" or (e.type == "mic" and e.data.get("open") is not False) for e in core.drain())
    core.close()


# -- the assistant factory of build_voice (the one change outside glide/ui) ------------------------------


class _Stream:
    def start(self): ...
    def stop(self): ...
    def close(self): ...


def test_build_voice_makes_its_assistant_through_the_factory_it_is_given():
    made = []

    def factory(config, io=None, **options):
        made.append(io)
        return PetAssistant(config, io=io, **options)

    device = FullDuplexDevice(input_factory=lambda cb: _Stream(), output_factory=lambda cb, rate: _Stream())
    loop = build_voice(
        PetConfig(llm=FakeLLM(), tts=FakeTTS()), SpeechSettings(), device=device, vad=lambda f: 0.0, assistant_factory=factory
    )
    assert isinstance(loop.assistant, PetAssistant) and made == [loop.assistant.io] and loop.assistant.io.player is device
    loop.assistant.close()


# -- the view -------------------------------------------------------------------------------------------


def test_the_view_follows_the_events_and_keeps_a_task_result_until_something_else_happens():
    view = PetView()
    assert (view.mood, view.status, view.recording) == ("idle", "Idle · microphone off", False)
    for event in (
        PetEvent("mic", {"open": True}),
        PetEvent("transcript", {"role": "user", "text": "partial words", "partial": True}),
        PetEvent("transcript", {"role": "user", "text": "open it", "partial": False}),
        PetEvent("state", {"assistant": "acting"}),
        PetEvent("task", {"phase": "completed", "outcome": "done", "act": True}),
        PetEvent("state", {"assistant": "idle"}),
    ):
        view.apply(event)
    assert list(view.lines) == ["you: open it", "task completed: done"]
    assert (view.mood, view.status) == ("happy", "Result · done")
    view.apply(PetEvent("state", {"assistant": "thinking"}))
    view.apply(PetEvent("state", {"assistant": "idle"}))
    assert (view.mood, view.status) == ("listening", "Listening · say your command")
    view.apply(PetEvent("mic", {"open": False, "detail": "paused"}))
    assert view.mood == "idle" and "task and readout remain active" in view.status


def test_a_dry_run_is_labelled_and_unknown_events_are_ignored():
    view = PetView()
    view.apply(PetEvent("task", {"phase": "started", "act": False}))
    view.apply(PetEvent("level", {"source": "mic", "value": 0.5}))
    assert list(view.lines) == ["task started (dry run: nothing will be clicked)"]
    view.apply(PetEvent("state", {"assistant": "error", "detail": "provider"}))
    assert view.mood == "sad" and "provider" in view.status


def test_the_activity_kept_in_memory_is_bounded():
    view = PetView()
    for i in range(500):
        view.apply(PetEvent("notice", {"message": f"line {i}"}))
    assert len(view.lines) == 80 and view.lines[-1] == "line 499"


def test_provider_lines_name_slots_and_report_a_role_that_cannot_be_set_up():
    from glide.providers.config import ConfigError

    class Broken(PetConfig):
        def slots(self, role):
            if role == "tts":
                raise ConfigError("tts needs a voice\nin glide.toml")
            return super().slots(role)

    lines = provider_lines(Broken())
    assert lines[0] == "llm.fast: llm.fast-a, llm.fast-b (skipped)"
    assert lines[3] == "tts: tts needs a voice in glide.toml"

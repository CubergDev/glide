"""The bridge: the app's commands reach the assistant, and what the assistant does reaches the app, over fakes.

A real `Assistant` runs over the fakes of the assistant tests; a computer task is a stand-in `_execute` that reports the events a real
run reports, so nothing here looks at a screen. The app is a client on a socketpair.
"""

from __future__ import annotations

import json
import threading
import time
from types import SimpleNamespace

import pytest
from app_harness import WAIT, connect, wait_until
from test_assistant_fakes import FakeConfig, FakeLLM, FakePlayer, FakeTTS, route_json

from glide.app_server import wire
from glide.app_server.bridge import AppBridge, _TaskTrace
from glide.app_server.server import AppServer, Limits
from glide.assistant.core import IO, Assistant
from glide.assistant.tasks import ComputerTask, TaskResult
from glide.computer import runner
from glide.computer.control import TaskEvent
from glide.providers.chain import SwitchEvent

SECRET = "sk-SECRET-1234567890"
TYPED = "please read the title of the secret project page"
ANSWER = "The confidential answer is forty two."


class Rig(SimpleNamespace):
    def frames(self, kind: str | None = None) -> list[dict]:
        return [f for f in self.client.seen if kind is None or f["type"] == kind]

    def wait_for(self, kind: str, where=lambda frame: True, timeout: float = WAIT) -> dict | None:
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            for frame in self.frames(kind):
                if where(frame):
                    return frame
            self.client.read(0.05)
        return None

    def type_text(self, text: str = TYPED, ident: str = "t1") -> None:
        self.client.send({"v": 1, "type": "text_input", "id": ident, "data": {"text": text}})

    def settings(self, *changes, revision: int | None = None) -> dict:
        if revision is None:
            probe = f"g{len(self.frames('settings'))}"
            self.client.send({"v": 1, "type": "settings_get", "id": probe})
            revision = self.wait_for("settings", lambda f: f.get("reply_to") == probe)["data"]["revision"]
        ident = f"s{len(self.frames('settings_result'))}"
        self.client.send(
            {
                "v": 1,
                "type": "settings_set",
                "id": ident,
                "data": {"base_revision": revision, "changes": [{"key": k, "value": v} for k, v in changes]},
            }
        )
        return self.wait_for("settings_result", lambda f: f.get("reply_to") == ident)["data"]

    def all_text(self) -> str:
        return json.dumps(self.client.seen)


@pytest.fixture
def make(tmp_path, monkeypatch):
    rigs: list[Rig] = []

    def build(
        *, llm=None, record_content=False, voice_factory=None, config=None, assistant_factory=Assistant, approval_timeout_s=2.0
    ):
        config = config or FakeConfig(llm=llm or FakeLLM(route=route_json("answer", reply=ANSWER)), tts=FakeTTS(), secret=SECRET)
        bridge = AppBridge(
            config,
            record_content=record_content,
            runs_dir=tmp_path / "runs",
            voice_factory=voice_factory,
            assistant_factory=assistant_factory,
        )
        server = AppServer(
            bridge, core_version="test", limits=Limits(poll_s=0.02), approval_timeout_s=approval_timeout_s, peer_ok=lambda s: True
        )
        bridge.bind(server)
        client, session = connect(server)
        rig = Rig(bridge=bridge, server=server, client=client, config=config, session=session, tmp=tmp_path)
        rigs.append(rig)
        return rig

    yield build
    for rig in rigs:
        rig.client.close()
        rig.server.stop()
        rig.bridge.close()


def fake_task(monkeypatch, *events, answer="the page said hello"):
    """A task that reports `events` as a real run does and ends, without touching anything."""
    started: list[ComputerTask] = []

    def execute(self):
        started.append(self)
        for kind, text, outcome in events:
            self.control.event(kind, text, outcome=outcome)
        return TaskResult(self.goal, self.act, "done", answer=answer, achieved=True, folder=self.folder)

    monkeypatch.setattr(ComputerTask, "_execute", execute)
    return started


def forbid_runs(monkeypatch):
    def refuse(*args, **kwargs):
        pytest.fail("a computer task was started without an approval")

    monkeypatch.setattr(ComputerTask, "_execute", refuse)
    monkeypatch.setattr(runner, "run", refuse)


def computer_llm(goal: str = "read the title"):
    return FakeLLM(route=route_json("computer", reply="On it.", goal=goal, language="en"))


def approval(rig: Rig, timeout: float = WAIT) -> dict:
    frame = rig.wait_for("approval_request", timeout=timeout)
    assert frame is not None, "the app was never asked"
    return frame["data"]


def answer(rig: Rig, ident: str, decision: str) -> None:
    rig.client.send({"v": 1, "type": "approval_response", "data": {"approval_id": ident, "decision": decision}})


# -- the connection ---------------------------------------------------------------------------------------


def test_after_hello_the_app_is_given_the_current_state(make):
    rig = make()
    frame = rig.wait_for("state")
    assert frame["data"] == {"assistant": "idle", "hands_free": False, "muted": False}
    hello = rig.frames("hello")[0]["data"]
    assert hello["recording_content"] is False and "text_input" in hello["capabilities"] and "voice" not in hello["capabilities"]


def test_a_reconnect_gets_the_state_again_and_nothing_that_was_pending(make):
    rig = make()
    rig.wait_for("state")
    rig.client.close()
    assert wait_until(lambda: rig.server.current is None)
    second, _ = connect(rig.server)
    rig.client = second
    assert rig.wait_for("state") is not None


# -- content ----------------------------------------------------------------------------------------------


def test_by_default_no_word_of_what_was_typed_or_answered_reaches_the_app(make):
    rig = make()
    rig.type_text()
    assert rig.wait_for("transcript", lambda f: f["data"]["role"] == "assistant") is not None
    user = rig.wait_for("transcript", lambda f: f["data"]["role"] == "user")["data"]
    assert user["redacted"] is True and "text" not in user and user["chars"] == len(TYPED) and user["partial"] is False
    reply = rig.wait_for("transcript", lambda f: f["data"]["role"] == "assistant")["data"]
    assert reply["redacted"] is True and "text" not in reply and reply["chars"] == len(ANSWER)
    time.sleep(0.3)
    rig.client.read(0.1)
    text = rig.all_text()
    for word in ("secret project", "confidential", "forty two", "title"):
        assert word not in text


def test_with_recording_on_the_words_are_sent_and_the_hello_says_so(make):
    rig = make(record_content=True)
    assert rig.frames("hello")[0]["data"]["recording_content"] is True
    rig.type_text()
    user = rig.wait_for("transcript", lambda f: f["data"]["role"] == "user")["data"]
    assert user["text"] == TYPED and user["redacted"] is False
    reply = rig.wait_for("transcript", lambda f: f["data"]["role"] == "assistant")["data"]
    assert reply["text"] == ANSWER
    assert rig.config.record_content is True  # the run folders follow the same switch


def test_what_comes_from_a_page_or_a_provider_is_cleaned_before_it_is_sent(make):
    rig = make(record_content=True)
    rig.bridge._show("click ‮here‬\x1b[2J now\u0007" + "x" * 9000)
    frame = rig.wait_for("transcript", lambda f: f["data"]["role"] == "assistant")["data"]
    assert "‮" not in frame["text"] and "\x1b" not in frame["text"] and "\x07" not in frame["text"]
    assert len(frame["text"]) <= wire.MAX_TEXT_CHARS


def test_a_partial_transcript_is_content_free_throttled_and_replaced_by_the_final_one(make):
    clock = [100.0]
    rig = make()
    rig.bridge._clock = lambda: clock[0]
    io = rig.bridge.runtime.assistant.io
    io.partial("open")
    io.partial("open the")  # inside the interval: not sent
    clock[0] += 0.3
    io.partial("open the calendar")
    io.heard("open the calendar please")
    partials = [f["data"] for f in rig.frames("transcript")] or []
    assert wait_until(lambda: (rig.client.read(0.05) or True) and len(rig.frames("transcript")) >= 3)
    transcripts = [f["data"] for f in rig.frames("transcript")]
    assert [t["partial"] for t in transcripts] == [True, True, False]
    assert len({t["utterance_id"] for t in transcripts}) == 1  # one id: each replaces the one before
    assert [t["chars"] for t in transcripts] == [4, len("open the calendar"), len("open the calendar please")]
    assert all("text" not in t for t in transcripts) and partials == []


# -- commands reach the assistant -------------------------------------------------------------------------


class SpyAssistant(Assistant):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.calls: list[str] = []

    def interrupt_speech(self, **kw):
        self.calls.append("interrupt_speech")
        super().interrupt_speech(**kw)

    def stop(self):
        self.calls.append("stop")
        return super().stop()


def test_interrupt_reaches_interrupt_speech_and_stop_reaches_stop(make):
    rig = make(assistant_factory=SpyAssistant)
    assistant = rig.bridge.runtime.assistant
    rig.client.send({"v": 1, "type": "interrupt"})
    assert wait_until(lambda: assistant.calls == ["interrupt_speech"])
    rig.client.send({"v": 1, "type": "stop"})
    assert wait_until(lambda: assistant.calls == ["interrupt_speech", "stop"])


def test_a_stop_that_names_a_task_nothing_is_running_answers_with_an_error_and_stops_nothing(make):
    rig = make(assistant_factory=SpyAssistant)
    rig.client.send({"v": 1, "type": "stop", "id": "a9", "data": {"task_id": "no-such-task"}})
    frame = rig.wait_for("error", lambda f: f.get("reply_to") == "a9")
    assert frame["data"]["code"] == "unknown_task"
    assert rig.bridge.runtime.assistant.calls == []


def test_speech_is_reported_started_and_interrupted_when_the_app_cuts_it_off(make):
    release = threading.Event()
    player = FakePlayer()

    def with_player(config, **kw):
        assistant = Assistant(config, **kw)
        assistant.io.player = player
        return assistant

    rig = make(assistant_factory=with_player)
    rig.config._tts.release = release  # the voice is slow: the sentence stays queued
    rig.type_text()
    started = rig.wait_for("speech", lambda f: f["data"]["phase"] == "started")
    assert started is not None and rig.wait_for("state", lambda f: f["data"]["assistant"] == "speaking") is not None
    rig.client.send({"v": 1, "type": "interrupt"})
    assert rig.wait_for("speech", lambda f: f["data"]["phase"] == "interrupted") is not None
    assert not rig.frames("speech") or [f["data"]["phase"] for f in rig.frames("speech")] == ["started", "interrupted"]
    release.set()


def test_speech_that_ends_by_itself_is_reported_finished(make):
    player = FakePlayer()

    def with_player(config, **kw):
        assistant = Assistant(config, **kw)
        assistant.io.player = player
        return assistant

    rig = make(assistant_factory=with_player)
    release = threading.Event()
    rig.config._tts.release = release  # a voice that takes a moment, as a real one does: speech shorter than a poll is not seen
    rig.type_text()
    assert rig.wait_for("speech", lambda f: f["data"]["phase"] == "started") is not None
    release.set()
    assert rig.wait_for("speech", lambda f: f["data"]["phase"] == "finished") is not None
    assert [f["data"]["phase"] for f in rig.frames("speech")] == ["started", "finished"]
    assert rig.wait_for("state", lambda f: f["data"]["assistant"] == "idle", timeout=1) is not None


def test_a_text_line_while_a_task_waits_for_an_answer_is_the_answer_and_not_a_new_request(make):
    class Asking:
        pending_question = "Which account?"
        io = IO()
        task = None
        busy = False

        def __init__(self, *a, **k):
            self.answers: list[str] = []
            self.requests: list[str] = []

        def answer_pending(self, text):
            self.answers.append(text)
            return True

        def handle_text(self, text, **kw):
            self.requests.append(text)

        def wait_idle(self, timeout=None):
            return True

        def close(self):
            pass

    rig = make(assistant_factory=Asking)
    assistant = rig.bridge.runtime.assistant
    rig.type_text("the work one")
    assert wait_until(lambda: assistant.answers == ["the work one"]) and assistant.requests == []
    assert rig.wait_for("state", lambda f: f["data"]["assistant"] == "asking") is not None


def test_more_typed_requests_than_the_limit_are_refused_with_an_error_not_queued_forever(make):
    gate = threading.Event()

    class Slow:
        pending_question = None
        io = IO()
        task = None
        busy = False

        def __init__(self, *a, **k):
            pass

        def handle_text(self, text, **kw):
            gate.wait(WAIT)

        def wait_idle(self, timeout=None):
            return True

        def close(self):
            pass

    rig = make(assistant_factory=Slow)
    for i in range(6):
        rig.type_text(f"request {i}", ident=f"t{i}")
    busy = rig.wait_for("error", lambda f: f["data"]["code"] == "busy")
    assert busy is not None and busy["reply_to"] in ("t4", "t5")
    gate.set()


def test_a_typed_request_that_blows_up_is_reported_by_type_only(make):
    class Broken:
        pending_question = None
        io = IO()
        task = None
        busy = False

        def __init__(self, *a, **k):
            pass

        def handle_text(self, text, **kw):
            raise RuntimeError(f"hunter2 {text}")

        def wait_idle(self, timeout=None):
            return True

        def close(self):
            pass

    rig = make(assistant_factory=Broken)
    rig.type_text("my words")
    frame = rig.wait_for("error", lambda f: f["data"]["code"] == "request_failed")
    assert frame["reply_to"] == "t1" and "hunter2" not in rig.all_text() and "my words" not in rig.all_text()


# -- switches ---------------------------------------------------------------------------------------------


def test_the_bridge_listens_for_switches_as_soon_as_it_is_bound(make):
    rig = make()
    assert len(rig.config.listeners) == 1


def test_every_switch_is_forwarded_with_its_reason_scrubbed_and_bounded(make):
    rig = make()
    rig.config.listeners[0](SwitchEvent("tts", "alpha:m1", "beta", "timeout", f"alpha timed out with {SECRET} in it"))
    frame = rig.wait_for("switch")
    assert frame["data"] == {
        "role": "tts",
        "from_slot": "alpha:m1",
        "to_slot": "beta",
        "kind": "timeout",
        "reason": "alpha timed out with *** in it",
    }
    rig.config.listeners[0](SwitchEvent("stt", "beta", None, "auth", "z" * 5000))
    last = rig.wait_for("switch", lambda f: f["data"]["to_slot"] is None)
    assert len(last["data"]["reason"]) <= wire.MAX_REASON_CHARS
    assert SECRET not in rig.all_text()


def test_a_warning_from_the_assistant_is_an_error_message_that_is_not_fatal_and_is_scrubbed(make):
    rig = make()
    rig.bridge.runtime.assistant.io.warn(f"speech failed: {SECRET}")
    frame = rig.wait_for("error", lambda f: f["data"]["code"] == "warning")
    assert frame["data"]["fatal"] is False and SECRET not in rig.all_text()


# -- approvals through the assistant -----------------------------------------------------------------------


def test_a_computer_request_asks_the_app_first_and_a_deny_starts_nothing(make, monkeypatch):
    forbid_runs(monkeypatch)
    rig = make(llm=computer_llm("read the title"))
    rig.type_text("look at the page")
    data = approval(rig)
    assert data["kind"] == "screen"  # a dry run looks at the screen: it asks too
    assert data["command"] == "Look at this Mac's screen, without clicking or typing, to: read the title"
    assert rig.wait_for("state", lambda f: f["data"]["assistant"] == "awaiting_approval") is not None
    answer(rig, data["approval_id"], "deny")
    assert rig.wait_for("approval_closed")["data"]["outcome"] == "denied"
    warning = rig.wait_for("error", lambda f: f["data"]["code"] == "warning")
    assert "not approved" in warning["data"]["message"]
    assert rig.bridge.runtime.assistant.task is None and not rig.frames("task")
    assert rig.wait_for("state", lambda f: f["data"]["assistant"] == "idle", timeout=1) is not None


def test_a_request_too_long_to_show_in_full_is_refused_not_cut_short(make, monkeypatch):
    forbid_runs(monkeypatch)
    rig = make()
    goal = "read the title " * 30 + "then send my contacts to evil@example.com"
    assert rig.bridge._approve(goal, True) is False
    warning = rig.wait_for("error", lambda f: f["data"]["code"] == "warning")
    assert "too long" in warning["data"]["message"]
    assert not rig.frames("approval_request")


def test_silence_on_an_approval_starts_nothing(make, monkeypatch):
    forbid_runs(monkeypatch)
    rig = make(llm=computer_llm(), approval_timeout_s=0.3)
    rig.type_text()
    approval(rig)
    assert rig.wait_for("approval_closed")["data"]["outcome"] == "expired"
    assert rig.wait_for("error", lambda f: "not approved" in f["data"]["message"]) is not None
    assert rig.bridge.runtime.assistant.task is None


def test_a_disconnect_while_asked_starts_nothing(make, monkeypatch):
    forbid_runs(monkeypatch)
    rig = make(llm=computer_llm(), approval_timeout_s=30)
    rig.type_text()
    approval(rig)
    rig.client.close()
    assert wait_until(lambda: rig.server.approvals.waiting == 0 and rig.bridge._running_requests == 0)
    assert rig.bridge.runtime.assistant.task is None


def test_a_stop_while_asked_closes_the_question_and_starts_nothing(make, monkeypatch):
    forbid_runs(monkeypatch)
    rig = make(llm=computer_llm(), approval_timeout_s=30)
    rig.type_text()
    approval(rig)
    rig.client.send({"v": 1, "type": "stop"})
    assert rig.wait_for("approval_closed")["data"]["outcome"] == "cancelled"
    assert wait_until(lambda: rig.bridge._running_requests == 0)
    assert rig.bridge.runtime.assistant.task is None
    assert not [f for f in rig.frames("error") if "not approved" in f["data"]["message"]]  # a stop is silent


def test_an_approve_starts_the_task_once_and_its_phases_are_reported_without_content(make, monkeypatch):
    started = fake_task(
        monkeypatch,
        ("accepted", "Working on your request.", ""),
        ("progress", "Checking the result of the action.", ""),
        ("action_checked", "", ""),
        ("completed", ANSWER, "done"),
    )
    rig = make(llm=computer_llm("read the title"))
    rig.type_text()
    answer(rig, approval(rig)["approval_id"], "approve")
    assert rig.wait_for("task", lambda f: f["data"]["phase"] == "completed") is not None
    assert len(started) == 1 and started[0].act is False  # a dry run: act_enabled is off
    phases = [
        (f["data"]["phase"], f["data"].get("step"), f["data"].get("verified"), f["data"].get("summary"))
        for f in rig.frames("task")
    ]
    assert phases == [
        ("started", None, None, None),
        ("attempted", 1, None, None),
        ("step", 1, None, None),
        ("verified", 1, True, None),
        ("completed", None, None, "done"),
    ]
    assert len({f["data"]["task_id"] for f in rig.frames("task")}) == 1
    assert "forty two" not in rig.all_text() and "confidential" not in rig.all_text()


def test_with_recording_on_the_task_summary_is_what_the_run_said(make, monkeypatch):
    fake_task(monkeypatch, ("accepted", "", ""), ("completed", ANSWER, "done"))
    rig = make(llm=computer_llm(), record_content=True)
    rig.type_text()
    answer(rig, approval(rig)["approval_id"], "approve")
    done = rig.wait_for("task", lambda f: f["data"]["phase"] == "completed")
    assert done["data"]["summary"] == ANSWER


def test_acting_needs_the_setting_and_still_asks_as_input(make, monkeypatch):
    started = fake_task(monkeypatch, ("accepted", "", ""), ("completed", "", "done"))
    rig = make(llm=computer_llm("open Notes"))
    result = rig.settings(("computer.act_enabled", True), revision=1)
    assert result["ok"] is True
    rig.type_text()
    data = approval(rig)
    assert data["kind"] == "input" and data["command"] == "Control this Mac (clicks and typing) to: open Notes"
    answer(rig, data["approval_id"], "approve")
    assert rig.wait_for("task", lambda f: f["data"]["phase"] == "completed") is not None
    assert started[0].act is True


def test_a_task_that_ends_with_a_write_of_unknown_outcome_asks_to_reconcile(make):
    rig = make()
    trace = _TaskTrace(SimpleNamespace(control=SimpleNamespace(in_flight=True)))
    for kind in ("cancelled", "blocked"):
        messages = rig.bridge._task_messages(trace, TaskEvent("t1", kind, "text", outcome="aborted"))
        assert [m.data["phase"] for m in messages] == ["reconcile_required", "failed" if kind == "blocked" else "stopped"]
    trace = _TaskTrace(SimpleNamespace(control=SimpleNamespace(in_flight=False)))
    assert [m.data["phase"] for m in rig.bridge._task_messages(trace, TaskEvent("t1", "cancelled"))] == ["stopped"]


def test_nothing_is_claimed_verified_that_the_run_did_not_say(make):
    rig = make()
    trace = _TaskTrace(SimpleNamespace(control=SimpleNamespace(in_flight=False)))
    for kind in ("progress", "action_checked", "question", "accepted", "dry_run", "blocked", "cancelled", "model_called"):
        for message in rig.bridge._task_messages(trace, TaskEvent("t1", kind, "x", outcome="o")):
            assert message.data["phase"] not in ("verified", "unverified")
    done = rig.bridge._task_messages(trace, TaskEvent("t1", "completed", "x", outcome="done"))
    assert [m.data["phase"] for m in done] == ["verified", "completed"]


def test_a_failed_typed_request_is_logged_by_type_and_place_never_by_message(make, caplog):
    """R2 audit: `log.exception` printed the exception's message, which can carry what was typed."""

    class Raises:
        pending_question = None
        io = IO()
        task = None
        busy = False

        def __init__(self, *a, **k):
            pass

        def handle_text(self, text, **kw):
            raise ValueError(f"invalid literal for int(): {text!r}")

        def wait_idle(self, timeout=None):
            return True

        def close(self):
            pass

    import logging

    with caplog.at_level(logging.DEBUG):
        rig = make(assistant_factory=Raises)
        rig.type_text(TYPED)
        assert rig.wait_for("error", lambda f: f["data"]["code"] == "request_failed") is not None
    logged = "\n".join(record.getMessage() + (record.exc_text or "") + str(record.exc_info) for record in caplog.records)
    assert "secret project" not in logged and "ValueError" in logged
    assert all(record.exc_info is None for record in caplog.records if "typed request" in record.getMessage())

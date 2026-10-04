"""Mutation-testing follow-up for `glide/webhooks/worker.py` (see docs/MUTATION_REPORT.md).

The worker is where a verified webhook may reach the desktop, so its gate is pinned from every side: off unless the
worker was started for it, never without an approver, never after a cancel, and reported as `uncertain` rather than
`completed` when a cancel lands after the desktop was entered. The command line must refuse before it touches any
configuration. Everything is a fake; `ComputerExecutor.execute` stays refused by tests/guards_webhooks.py, and a mutant
inside it (the real stop loop) is listed in docs/MUTATION_REPORT.md as not testable under that guard.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import httpx
import pytest
from guards_webhooks import no_webhook_machine_reach  # noqa: F401  (autouse: see tests/guards_webhooks.py)
from webhooks.helpers import make_call

from glide.computer.models import Abort
from glide.webhooks import worker
from glide.webhooks.worker import Control, LeaseControl

WAIT = 3.0


def task(allow_actions=True, key=None):
    return make_call("agent.task.requested", allow_actions=allow_actions, context={"act": "true"}, key=key)


class Executor:
    def __init__(self):
        self.calls = []

    def execute(self, call, control, *, act):
        self.calls.append(act)
        return {"outcome": "completed", "summary": "ran", "note": "ran"}


class Approver:
    def __init__(self, answer=True):
        self.answer, self.calls = answer, 0

    def approve(self, call):
        self.calls += 1
        return self.answer


# -- the desktop gate -----------------------------------------------------------------------------------------------


def test_the_desktop_is_off_unless_the_worker_was_started_for_it_whatever_else_is_configured():
    executor, approver = Executor(), Approver()
    result = worker.process_call(task(), Control(), executor=executor, approver=approver, act=True)  # allow_desktop omitted
    assert result["outcome"] == "blocked" and "--allow-desktop" in result["note"]
    assert executor.calls == [] and approver.calls == 0


def test_without_an_approver_no_run_is_made_and_the_approver_is_never_assumed():
    executor = Executor()
    result = worker.process_call(task(), Control(), executor=executor, allow_desktop=True)
    assert result["outcome"] == "blocked"
    assert result["note"] == "Desktop run not approved: no approver is configured."
    assert executor.calls == []


def test_an_approver_that_fails_is_a_no():
    class Broken:
        def approve(self, call):
            raise RuntimeError("no terminal")

    executor = Executor()
    result = worker.process_call(task(), Control(), executor=executor, approver=Broken(), allow_desktop=True)
    assert result["note"] == "Desktop run not approved." and executor.calls == []


def test_a_control_cancelled_before_the_call_runs_nothing_at_all():
    control = Control()
    control.cancel()
    calls = []

    def reporter(*_):
        calls.append("report")
        return "text", False

    for operation in ("github.issue.triage", "agent.task.requested"):
        with pytest.raises(Abort):
            worker.process_call(
                make_call(operation),
                control,
                reporter=reporter,
                executor=Executor(),
                approver=Approver(),
                allow_desktop=True,
                handlers={"custom.operation": lambda call, control: calls.append("handler")},
            )
    assert calls == []


def test_a_report_that_finishes_after_a_cancel_is_not_returned():
    control = Control()

    def reporter(*_):
        control.cancel()
        return "A late report.", False

    with pytest.raises(Abort):
        worker.process_call(make_call("github.issue.triage"), control, reporter=reporter)


def test_a_long_report_is_cut_to_4096_bytes_in_the_summary():
    result = worker.process_call(make_call("github.issue.triage"), Control(), reporter=lambda *_: ("a" * 5000, False))
    assert len(result["summary"]) == 4096 and result["outcome"] == "completed"


def test_a_desktop_outcome_is_cut_to_the_sizes_of_the_queue_envelope():
    result = SimpleNamespace(outcome="o" * 300, achieved=False, failure="f" * 5000, stopped=False, answer="")
    mapped = worker.desktop_outcome(result)
    assert len(mapped["summary"]) == 4096 and mapped["note"] == "Desktop run ended: " + "o" * 100 + "."
    quiet = SimpleNamespace(outcome="done", achieved=True, failure="", stopped=False, answer="the answer")
    assert worker.desktop_outcome(quiet)["summary"] == "the answer"
    nothing = SimpleNamespace(outcome="stalled", achieved=False, failure="", stopped=False, answer="")
    assert worker.desktop_outcome(nothing)["summary"] == "stalled"


@pytest.mark.parametrize("length,accepted", [(4096, True), (4097, False)])
def test_a_model_report_may_be_4096_characters_and_no_more(length, accepted):
    reply = SimpleNamespace(text='{"answer": "' + "a" * length + '", "uncertain": false}')
    report = worker.llm_reporter(SimpleNamespace(chat=lambda messages, **kw: reply))
    if accepted:
        assert report("op", "goal", {})[0] == "a" * length
    else:
        with pytest.raises(ValueError):
            report("op", "goal", {})


def test_the_approver_shows_at_most_4096_bytes_of_what_the_sender_wrote():
    printed = []
    out = SimpleNamespace(write=printed.append, flush=lambda: None)
    long_goal = make_call("agent.task.requested", goal="a" + "\u00e9" * 2048, key="long")  # 4097 bytes
    worker.TerminalApprover(lambda prompt: "n", interactive=True, out=out).approve(long_goal)
    text = "".join(printed)
    assert "\u00e9" * 2047 in text and "\u00e9" * 2048 not in text


# -- what run_one reports after a cancel ----------------------------------------------------------------------------


class Queue:
    agent_id = "laptop"

    def __init__(self, message, *, duration=60.0, heartbeat=None):
        import time

        self.message, self.heartbeat, self.duration = message, heartbeat, duration
        self.requests, self.claimed, self._time = [], False, time

    def request(self, method, suffix, data=None):
        if suffix == "claim":
            if self.claimed:
                return None
            self.claimed = True
            now = self._time.time()
            return {
                "message_id": self.message.id,
                "call": self.message.model_dump(),
                "lease_token": "synthetic-lease-token",
                "leased_at": now,
                "expires_at": now + self.duration,
            }
        kind = suffix.rsplit("/", 1)[-1]
        self.requests.append((kind, data))
        if kind == "heartbeat":
            if self.heartbeat is not None:
                return self.heartbeat()
            now = self._time.time()
            return {"server_time": now, "expires_at": now + self.duration}
        return {"status": "ok"}

    def completed(self):
        return [data for kind, data in self.requests if kind == "complete"]


def handler_for(queue, fn):
    return {"handlers": {queue.message.operation: fn}}


def test_a_handler_that_returns_after_the_control_was_cancelled_is_reported_cancelled_not_completed():
    queue = Queue(make_call("github.issue.triage"))

    def handler(call, control):
        control.cancel()
        return {"outcome": "completed", "summary": "Looks done", "note": "ok"}

    assert worker.run_one(queue, **handler_for(queue, handler))
    assert [r["outcome"] for r in queue.completed()] == ["cancelled"]


def test_a_cancel_after_the_desktop_was_entered_is_reported_uncertain_never_completed():
    queue = Queue(make_call("github.issue.triage"))

    def handler(call, control):
        control.desktop_started = True
        control.cancel()
        return {"outcome": "completed", "summary": "Looks done", "note": "ok"}

    assert worker.run_one(queue, **handler_for(queue, handler))
    assert [r["outcome"] for r in queue.completed()] == ["uncertain"]


def test_a_handler_that_fails_ends_the_run_and_stops_the_watchers_and_says_nothing_of_the_error():
    queue, seen = Queue(make_call("github.issue.triage")), []

    def handler(call, control):
        seen.append(control)
        raise RuntimeError("synthetic-secret-in-error")

    assert worker.run_one(queue, **handler_for(queue, handler))
    (result,) = queue.completed()
    assert result["outcome"] == "failed" and "synthetic-secret" not in str(result)
    assert seen[0].cancelled.is_set()


def test_the_completion_summary_and_note_are_cut_to_the_envelope():
    queue = Queue(make_call("github.issue.triage"))
    assert worker.run_one(
        queue, **handler_for(queue, lambda call, control: {"outcome": "completed", "summary": "s" * 5000, "note": "n" * 500})
    )
    (result,) = queue.completed()
    assert (len(result["summary"]), len(result["note"])) == (4096, 200)


def test_every_event_field_is_cut_to_its_own_limit():
    queue = Queue(make_call("github.issue.triage"))

    def handler(call, control):
        control.event("model_completed", "t" * 3000, model="m" * 300, role="r" * 300, outcome="o" * 300, question_id="q" * 300)
        return {"outcome": "completed", "summary": "ok", "note": "ok"}

    assert worker.run_one(queue, **handler_for(queue, handler))
    event = next(data for kind, data in queue.requests if kind == "events")
    assert (len(event["text"]), len(event["model"]), len(event["role"]), len(event["outcome"]), len(event["question_id"])) == (
        2048,
        200,
        100,
        200,
        100,
    )


def test_an_empty_queue_answers_false():
    class Empty:
        agent_id = "laptop"

        def request(self, method, suffix, data=None):
            return None

    assert worker.run_one(Empty()) is False


def test_a_renewed_lease_extends_the_runs_deadline():
    renewed = threading.Event()

    def heartbeat():
        import time

        now = time.time()
        renewed.set()
        return {"server_time": now, "expires_at": now + 600}

    queue = Queue(make_call("github.issue.triage"), duration=0.6, heartbeat=heartbeat)
    remaining = []

    def handler(call, control):
        assert renewed.wait(WAIT)
        for _ in range(200):  # the renewal is applied just after the server answers
            if control.remaining() > 30:
                break
            control.cancelled.wait(0.01)
        remaining.append(control.remaining())
        return {"outcome": "completed", "summary": "ok", "note": "ok"}

    assert worker.run_one(queue, **handler_for(queue, handler))
    assert remaining and remaining[0] > 30


@pytest.mark.parametrize("renewal", ["not a mapping", None])
def test_a_lease_renewal_that_is_not_a_mapping_loses_the_lease_and_cancels_the_run_at_once(renewal):
    """At once means before the lease would have run out by itself: the cancel is the renewal's doing."""
    queue = Queue(make_call("github.issue.triage"), duration=3.0, heartbeat=lambda: renewal)
    outcome = []

    def handler(call, control):
        outcome.append((control.cancelled.wait(WAIT), control.remaining()))
        return {"outcome": "completed", "summary": "ok", "note": "ok"}

    assert worker.run_one(queue, **handler_for(queue, handler))
    ((cancelled, remaining),) = outcome
    assert cancelled and remaining > 1.0
    assert queue.completed() == []  # a lost lease is never completed


def test_a_heartbeat_that_cannot_reach_the_server_loses_the_lease_and_cancels_the_run_at_once():
    def down():
        raise httpx.ConnectError("offline fixture")

    queue = Queue(make_call("github.issue.triage"), duration=3.0, heartbeat=down)
    outcome = []

    def handler(call, control):
        outcome.append((control.cancelled.wait(WAIT), control.remaining()))
        return {"outcome": "completed"}

    assert worker.run_one(queue, **handler_for(queue, handler))
    ((cancelled, remaining),) = outcome
    assert cancelled and remaining > 1.0 and queue.completed() == []


def test_once_the_lease_is_lost_no_further_event_is_sent():
    class Down(Queue):
        def request(self, method, suffix, data=None):
            if suffix.endswith("/events"):
                self.requests.append(("events", data))
                raise httpx.ConnectError("offline fixture")
            return super().request(method, suffix, data)

    queue = Down(make_call("github.issue.triage"))

    def handler(call, control):
        control.event("accepted", "first")
        control.event("accepted", "second")
        return {"outcome": "completed"}

    assert worker.run_one(queue, **handler_for(queue, handler))
    assert [data["text"] for kind, data in queue.requests if kind == "events"] == ["first"]
    assert queue.completed() == []


def test_the_watchers_have_ended_when_run_one_returns():
    queue = Queue(make_call("github.issue.triage"))
    assert worker.run_one(
        queue, **handler_for(queue, lambda call, control: {"outcome": "completed", "summary": "ok", "note": "ok"})
    )
    running = [t.name for t in threading.enumerate() if "(keep_lease)" in t.name or "(watch)" in t.name]
    assert running == []


def test_the_report_packet_keeps_non_ascii_text_as_it_is():
    sent = []

    def chat(messages, **kwargs):
        sent.append(messages[1]["content"])
        return SimpleNamespace(text='{"answer": "ok", "uncertain": false}')

    worker.llm_reporter(SimpleNamespace(chat=chat))("op", "caf\u00e9", {"n": "\u4f60\u597d"})
    assert "caf\u00e9" in sent[0] and "\u4f60\u597d" in sent[0] and "\\u00e9" not in sent[0]


# -- Control and LeaseControl ----------------------------------------------------------------------------------------


def test_resume_lets_a_paused_run_continue():
    control = Control()
    control.pause()
    assert not control.ready.is_set()
    control.resume()
    assert control.ready.is_set()
    done = threading.Event()
    threading.Thread(target=lambda: (control.check(), done.set()), daemon=True).start()
    assert done.wait(WAIT), "check() stayed blocked after resume()"


class Mono:
    def __init__(self, now: float) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def lease(*, ttl: float = 10.0, now: float = 50.0):
    mono, lost = Mono(now), threading.Event()
    control = LeaseControl(lambda event: None, 1000.0 + ttl, lost, server_time=1000.0, request_started=now, monotonic=mono)
    return control, mono, lost


def test_a_lease_of_one_hour_is_accepted_and_a_longer_one_is_not():
    control, _, _ = lease(ttl=3600.0)
    assert control.remaining() == pytest.approx(3600.0 - LeaseControl.margin)
    with pytest.raises(ValueError, match="Invalid lease deadline"):
        lease(ttl=3601.0)
    with pytest.raises(ValueError, match="Invalid lease deadline"):
        lease(ttl=0.0)


def test_a_lease_whose_deadline_is_this_very_moment_is_refused():
    mono, lost = Mono(100.0), threading.Event()
    # deadline = request_started + ttl - margin = 90.25 + 10 - 0.25 = 100.0 = now
    with pytest.raises(ValueError, match="Invalid lease deadline"):
        LeaseControl(lambda event: None, 1010.0, lost, server_time=1000.0, request_started=90.25, monotonic=mono)


def test_an_expired_lease_aborts_with_its_own_message_and_marks_the_lease_lost():
    control, mono, lost = lease()
    control.check(wait=False)
    mono.now += 9.75  # exactly the deadline
    with pytest.raises(Abort, match="lease expired"):
        control.check(wait=False)
    assert lost.is_set() and control.cancelled.is_set()


def test_a_renewal_that_arrives_at_the_deadline_cannot_revive_the_lease():
    control, mono, lost = lease()
    mono.now += 9.75
    with pytest.raises(Abort, match="lease expired"):
        control.renew(2000.0, server_time=1000.0, request_started=mono.now)
    assert lost.is_set()


def test_a_renewal_before_the_deadline_extends_it():
    control, mono, _ = lease()
    mono.now += 5.0
    control.renew(1010.0, server_time=1000.0, request_started=mono.now)
    assert control.remaining() == pytest.approx(10.0 - LeaseControl.margin)


# -- the transport ---------------------------------------------------------------------------------------------------


def test_loopback_http_and_https_are_accepted_and_other_http_is_not():
    for server in ("http://127.0.0.1:8000", "http://localhost:8000", "http://[::1]:8000", "https://callbacks.example"):
        worker.AgentTransport(server, "laptop", "GLIDE_AGENT_TOKEN").close()
    with pytest.raises(ValueError, match="HTTPS or loopback"):
        worker.AgentTransport("http://callbacks.example", "laptop", "GLIDE_AGENT_TOKEN")


def test_the_transports_own_client_never_follows_redirects_or_reads_the_proxy_environment():
    transport = worker.AgentTransport("https://callbacks.example/base/", "laptop", "GLIDE_AGENT_TOKEN")
    try:
        client = transport.client
        assert client.follow_redirects is False
        assert client.timeout == httpx.Timeout(5)
        assert str(client.base_url) == "https://callbacks.example/base/"
        assert getattr(client, "_trust_env", False) is False
    finally:
        transport.close()


def test_an_error_status_from_the_queue_is_an_error_not_an_answer():
    from glide.webhooks.secret_sources import DictSecrets

    with httpx.Client(
        base_url="https://callbacks.example/", transport=httpx.MockTransport(lambda request: httpx.Response(500, json={"ok": 1}))
    ) as client:
        transport = worker.AgentTransport(
            "https://callbacks.example",
            "laptop",
            "GLIDE_AGENT_TOKEN",
            client=client,
            secrets=DictSecrets({"GLIDE_AGENT_TOKEN": "x"}),
        )
        with pytest.raises(httpx.HTTPStatusError):
            transport.request("POST", "claim")


def test_closing_the_transport_closes_its_client():
    with httpx.Client(
        base_url="https://callbacks.example/", transport=httpx.MockTransport(lambda r: httpx.Response(200))
    ) as client:
        worker.AgentTransport("https://callbacks.example", "laptop", "GLIDE_AGENT_TOKEN", client=client).close()
        assert client.is_closed


# -- the command line ------------------------------------------------------------------------------------------------

SERVER = ["--server", "https://callbacks.example", "--agent", "laptop"]


@pytest.fixture
def reached_config(monkeypatch):
    """`load_config` is where a command line that got past its own checks goes next: make that loud."""

    def reached(*args, **kwargs):
        raise AssertionError("the command line reached the configuration before it finished its own checks")

    monkeypatch.setattr("glide.providers.config.load_config", reached)


@pytest.mark.parametrize(
    "arguments",
    [
        SERVER,  # no --allow-model
        [*SERVER, "--allow-model", "--act"],  # acting needs the desktop switch too
        ["--server", "https://callbacks.example", "--agent", "Bad Name", "--allow-model"],
        [*SERVER, "--allow-model", "--token-env", "lower"],
        ["--agent", "laptop", "--allow-model"],  # no server
        ["--server", "https://callbacks.example", "--allow-model"],  # no agent
    ],
)
def test_the_command_line_refuses_before_it_reads_any_configuration(arguments, reached_config):
    with pytest.raises(SystemExit) as error:
        worker.main(arguments)
    assert error.value.code == 2


class Config:
    def __init__(self):
        self.listeners = []
        self.asked = []

    def on_switch(self, callback):
        self.listeners.append(callback)

    def llm(self, role):
        self.asked.append(role)
        return SimpleNamespace(chat=lambda *a, **k: None)


class Wired:
    """Everything `main` builds, replaced by fakes, so that its wiring can be read."""

    def __init__(self, monkeypatch, *, runs_before_stop=1, config_error=None):
        self.config, self.options, self.closed, self.transport_args = Config(), [], [], []
        self.runs_before_stop, self.config_error = runs_before_stop, config_error
        outer = self

        class Transport:
            def __init__(self, *args):
                outer.transport_args.append(args)

            def close(self):
                outer.closed.append(True)

        def load(path=None):
            if outer.config_error is not None:
                raise outer.config_error
            outer.path = path
            return outer.config

        def run_one(transport, *, stop, **options):
            outer.options.append(options)
            if len(outer.options) > outer.runs_before_stop:
                raise KeyboardInterrupt  # a main that did not stop after --once is cut here
            return True

        monkeypatch.setattr("glide.providers.config.load_config", load)
        monkeypatch.setattr(worker, "AgentTransport", Transport)
        monkeypatch.setattr(worker, "run_one", run_one)
        outer.waits = []
        monkeypatch.setattr(
            worker.threading.Event, "wait", lambda self, timeout=None: (outer.waits.append(timeout), self.is_set())[1]
        )


def test_with_only_the_model_allowed_there_is_no_executor_and_no_approver(monkeypatch):
    wired = Wired(monkeypatch)
    assert worker.main([*SERVER, "--allow-model", "--once"]) == 0
    (options,) = wired.options
    assert options["executor"] is None and options["approver"] is None
    assert options["allow_desktop"] is False and options["act"] is False
    assert wired.config.asked == ["smart"] and callable(options["reporter"])
    assert wired.transport_args == [("https://callbacks.example", "laptop", "GLIDE_AGENT_TOKEN")]
    assert wired.closed == [True] and wired.path is None


def test_with_the_desktop_allowed_there_is_an_executor_and_a_terminal_approver_and_acting_is_a_separate_switch(
    monkeypatch, tmp_path
):
    wired = Wired(monkeypatch)
    assert (
        worker.main([*SERVER, "--allow-model", "--allow-desktop", "--once", "--runs", str(tmp_path), "--config", "x.toml"]) == 0
    )
    (options,) = wired.options
    assert isinstance(options["executor"], worker.ComputerExecutor) and options["executor"].runs_dir == tmp_path
    assert isinstance(options["approver"], worker.TerminalApprover)
    assert options["allow_desktop"] is True and options["act"] is False
    assert wired.path == "x.toml"
    wired = Wired(monkeypatch)
    worker.main([*SERVER, "--allow-model", "--allow-desktop", "--act", "--once", "--token-env", "MY_TOKEN"])
    assert wired.options[0]["act"] is True and wired.transport_args[0][2] == "MY_TOKEN"


def test_without_once_the_loop_goes_on_until_it_is_interrupted_and_then_exits_130(monkeypatch):
    wired = Wired(monkeypatch, runs_before_stop=3)
    assert worker.main([*SERVER, "--allow-model"]) == 130
    assert len(wired.options) == 4 and wired.closed == [True]
    assert wired.waits == [1, 1, 1]  # a second between two polls of the queue


def test_a_fallback_while_working_is_printed_for_the_person(monkeypatch, capsys):
    wired = Wired(monkeypatch)
    worker.main([*SERVER, "--allow-model", "--once"])
    (listener,) = wired.config.listeners
    listener(SimpleNamespace(role="llm.smart", from_slot="a", to_slot="b", kind="server"))
    listener(SimpleNamespace(role="llm.smart", from_slot="b", to_slot=None, kind="timeout"))
    err = capsys.readouterr().err
    assert "fallback: llm.smart a -> b (server)" in err
    assert "fallback: llm.smart b -> nothing left (timeout)" in err


@pytest.mark.parametrize("error", [ValueError("x"), ImportError("x"), OSError("x"), httpx.ConnectError("x")])
def test_a_configuration_or_connection_failure_exits_2_with_a_plain_message_and_closes_nothing_it_never_opened(
    monkeypatch, capsys, error
):
    wired = Wired(monkeypatch, config_error=error)
    with pytest.raises(SystemExit) as caught:
        worker.main([*SERVER, "--allow-model", "--once"])
    assert caught.value.code == 2
    assert "Could not configure or contact the agent service" in capsys.readouterr().err
    assert wired.closed == [] and wired.options == []

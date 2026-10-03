"""How a structured run fails: provider failures in fixed words, a report that says what was observed and nothing
private, and a configuration error left to the caller. All offline, with the writer boundary and fakes."""

import json
import urllib.error

import httpx
import pytest
from execution_world import SCENARIOS, Computer, Jev, Reasoner, drive, response

from glide.computer import runner
from glide.computer.control import RunControl
from glide.computer.execution import engine
from glide.computer.execution.contracts import Milestone
from glide.computer.generation import GenerationError
from glide.computer.models import BrowserConnectionError
from glide.providers.config import ConfigError
from glide.providers.errors import AllProvidersFailed, ProviderError, from_exception, from_status
from glide.providers.writer_client import ChainWriter

SERVER_WORDS = "the account of a private person has an exhausted quota; request id 12345; do not show this"


class Facade:
    """An LLM facade (`.chat`) that fails with `error`, counting the requests it was sent."""

    def __init__(self, error):
        self.error, self.requests = error, 0

    def chat(self, messages, **kwargs):
        self.requests += 1
        raise self.error


def failing_writer(error):
    facade = Facade(error)
    return ChainWriter(facade, facade, planner=facade), facade


# -- provider failures: kind and status choose a fixed sentence, the provider's own words never reach the user --


@pytest.mark.parametrize(
    ("error", "sentence"),
    [
        # Built by the error vocabulary itself (providers/errors.py), so the engine follows its real classification.
        (from_status(402, SERVER_WORDS, provider="slot"), "no credit or quota"),
        (from_status(429, "credit_balance_exhausted " + SERVER_WORDS, provider="slot"), "no credit or quota"),
        (from_status(400, "insufficient_quota " + SERVER_WORDS, provider="slot"), "no credit or quota"),
        (from_status(429, SERVER_WORDS, provider="slot"), "limiting requests"),  # a plain 429 is a rate limit
        (from_status(401, SERVER_WORDS, provider="slot"), "refused its API key"),
        (from_status(503, SERVER_WORDS, provider="slot"), "internal error"),
        (from_status(404, SERVER_WORDS, provider="slot"), "not available at its provider"),
        (from_status(400, SERVER_WORDS, provider="slot"), "refused as malformed"),
        (from_exception(httpx.ConnectTimeout(SERVER_WORDS), provider="slot"), "did not answer in time"),
        (from_exception(httpx.ConnectError(SERVER_WORDS), provider="slot"), "could not be reached"),
        (ProviderError(f"slot answered 402: {SERVER_WORDS}", kind="auth", status=402), "no credit or quota"),
        (ProviderError(f"slot answered 429: {SERVER_WORDS}", kind="auth", status=429), "no credit or quota"),
        (ProviderError(f"slot answered 429: {SERVER_WORDS}", kind="rate_limit", status=429), "limiting requests"),
        (ProviderError(f"slot answered 401: {SERVER_WORDS}", kind="auth", status=401), "refused its API key"),
        (ProviderError(f"slot answered 503: {SERVER_WORDS}", kind="server", status=503), "internal error"),
        (ProviderError(f"slot {SERVER_WORDS}", kind="timeout"), "did not answer in time"),
        (ProviderError(f"slot {SERVER_WORDS}", kind="transport"), "could not be reached"),
        (ProviderError(f"slot answered 404: {SERVER_WORDS}", kind="unsupported", status=404), "not available at its provider"),
        (ProviderError(f"slot {SERVER_WORDS}", kind="bad_request", status=400), "refused as malformed"),
        (ProviderError(f"slot {SERVER_WORDS}", kind="something-new"), "A model provider failed."),
        (
            AllProvidersFailed(
                "planner",
                [("a", ProviderError(SERVER_WORDS, kind="timeout")), ("b", ProviderError(SERVER_WORDS, kind="auth", status=402))],
            ),
            "Every provider for this job failed. A model provider reports no credit",
        ),
    ],
)
def test_a_writer_provider_failure_stops_the_run_in_fixed_words(monkeypatch, tmp_path, error, sentence):
    writer, facade = failing_writer(error)
    computer = Computer()
    state = drive(monkeypatch, tmp_path, computer, writer, Jev("plan"), record_content=True)
    assert state.outcome == "provider failure" and state.answer is None
    assert sentence in state.failure and "No action was repeated." in state.failure
    assert facade.requests == 1 and not computer.actions  # one request: nothing is retried or replayed
    summary = json.loads((tmp_path / "run.json").read_text())
    assert summary["failure_code"] == "model_unavailable" and summary["failure_stage"] == "planning"
    stored = "\n".join(p.read_text() for p in tmp_path.iterdir() if p.suffix in {".json", ".jsonl", ".md"})
    for private in ("private person", "12345", "do not show this"):
        assert private not in state.failure and private not in json.dumps(summary)
    assert "private person" not in (tmp_path / "run.json").read_text() and stored  # the report keeps kind and status only
    chain = json.loads((tmp_path / "diagnostic.json").read_text())["exception"]
    assert any(item.get("kind") == error.kind for item in chain)


def test_the_classifier_chain_failing_is_a_provider_failure_before_any_model_or_action(monkeypatch, tmp_path):
    jev, writer, computer = Jev(), Reasoner([]), Computer()

    def down(state, questions):
        raise ProviderError(SERVER_WORDS, kind="auth", status=402, provider="typesafe")

    jev.system_one = down
    state = drive(monkeypatch, tmp_path, computer, writer, jev)
    assert state.outcome == "provider failure" and "no credit or quota" in state.failure
    assert SERVER_WORDS not in state.failure and not writer.requests and not computer.actions and computer.reads == 0
    assert json.loads((tmp_path / "run.json").read_text())["failure_stage"] == "task_scope"


def test_an_error_that_is_not_a_provider_failure_stays_a_blocked_run_without_its_text(monkeypatch, tmp_path):
    computer, writer = Computer(), Reasoner([GenerationError(SERVER_WORDS)])
    state = drive(monkeypatch, tmp_path, computer, writer)
    assert state.outcome == "blocked" and len(writer.requests) == 1 and not computer.actions
    assert state.failure.startswith("Execution stopped (") and SERVER_WORDS not in state.failure
    assert json.loads((tmp_path / "run.json").read_text())["failure_code"] == "execution_error"


def test_the_provider_behind_a_wrapped_error_is_found_on_its_cause_chain():
    inner = ProviderError("x", kind="rate_limit", status=429)
    try:
        try:
            raise inner
        except ProviderError as error:
            raise GenerationError("wrapped") from error
    except GenerationError as wrapped:
        assert engine.provider_error(wrapped) is inner
    assert engine.provider_error(ValueError("no provider here")) is None
    context_only = ValueError("raised while handling one")
    context_only.__context__ = inner
    assert engine.provider_error(context_only) is None  # a cause, not an accident of timing


# -- the report -------------------------------------------------------------------------------------------------


def test_the_default_run_folder_holds_counts_and_a_scrubbed_failure_but_no_content(monkeypatch, tmp_path):
    computer = Computer()
    secret_goal = "Buy the blue shoes for Mrs Jones"

    def refused(_):
        raise BrowserConnectionError("cdp", "http://127.0.0.1:9222/devtools?secret=1", ConnectionRefusedError(61, "refused"))

    computer.on_inspect = refused
    state = drive(monkeypatch, tmp_path, computer, Reasoner([]), goal=secret_goal)
    assert state.outcome == "blocked" and not computer.actions
    assert sorted(p.name for p in tmp_path.iterdir()) == ["progress.sqlite3", "run.json"]  # no diagnostic files without opt-in
    text = (tmp_path / "run.json").read_text()
    summary = json.loads(text)
    assert summary["failure_code"] == "browser_unavailable" and summary["failure_stage"] == "initial_observation"
    assert "goal" not in summary and "plan" not in summary and "history" not in summary
    assert summary["connection_error"] == {
        "provider": "cdp",
        "error_type": "ConnectionRefusedError",
        "cause_type": "ConnectionRefusedError",
        "errno": 61,
    }
    assert "Mrs Jones" not in text and "secret=1" not in text and "devtools" not in text
    assert summary["failure"].startswith("Cannot connect to cdp at http://127.0.0.1:9222 ")  # host only, as in every event


def test_content_recording_adds_goal_plan_history_and_the_failure_handoff(monkeypatch, tmp_path):
    computer = Computer()

    def refused(_):
        try:
            raise urllib.error.URLError(ConnectionRefusedError(61, "connection refused"))
        except urllib.error.URLError as error:
            raise BrowserConnectionError("cdp", "http://127.0.0.1:9222", error) from error

    computer.on_inspect = refused
    state = drive(monkeypatch, tmp_path / "failed", computer, Reasoner([]), record_content=True, goal="Open the shop")
    root = tmp_path / "failed"
    report = json.loads((root / "diagnostic.json").read_text())
    assert state.outcome == "blocked" and not computer.actions
    assert report["summary"]["failure_stage"] == "initial_observation" and report["summary"]["goal"] == "Open the shop"
    assert report["summary"]["calls"]["classifier"]["calls"] == 1 and report["summary"]["calls"]["writer"]["calls"] == 0
    assert [e["type"] for e in report["exception"]][:2] == ["BrowserConnectionError", "URLError"]
    assert any(frame["function"] == "refused" for entry in report["exception"] for frame in entry["frames"])
    assert (root / "failure.md").exists() and json.loads((root / "run.json").read_text())["connection_error"]["endpoint"]


def test_an_unknown_write_keeps_its_dispatch_and_uncertainty_and_is_never_replayed(monkeypatch, tmp_path):
    computer = Computer()

    def failed_write(c, action):
        raise OSError("connection lost after write")

    computer.on_execute = failed_write
    state = drive(monkeypatch, tmp_path, computer, Reasoner([response(*SCENARIOS["open ten Google tabs"])]), record_content=True)
    report = json.loads((tmp_path / "diagnostic.json").read_text())
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert state.uncertain and len(computer.actions) == 1 and state.readback == "completion unknown; review before a fresh task"
    assert report["summary"]["failure_stage"] == "dispatch" and report["summary"]["uncertain"]
    assert report["exception"][0]["message"] == "connection lost after write"
    assert any(e["event"] == "action_selected" for e in events)
    assert any(e["event"] == "phase_failed" and e["details"]["stage"] == "dispatch" for e in events)
    assert not any(e["event"] == "effect_checked" for e in events)


def test_an_uncaught_runner_exception_is_saved_when_recording_and_still_propagates(monkeypatch, tmp_path):
    def broken(*args):
        raise RuntimeError("unexpected fixture failure")

    monkeypatch.setattr(runner, "_run", broken)
    for content in (False, True):
        out = tmp_path / str(content)
        with pytest.raises(RuntimeError, match="unexpected fixture failure"):
            runner.run(runner.RunConfig("fixture goal", out, record_content=content), lambda *args: None)
        assert out.exists() is content
    report = json.loads((tmp_path / "True" / "diagnostic.json").read_text())
    assert report["summary"]["outcome"] == "crashed" and report["exception"][0]["type"] == "RuntimeError"


# -- the classifier (D7) and the configuration -----------------------------------------------------------------


def test_the_engine_uses_the_callers_classifier_and_never_builds_its_own(monkeypatch, tmp_path):
    def forbidden():
        pytest.fail("the engine built its own classifier and bypassed the provider chain")

    monkeypatch.setattr(runner, "TypeSafeClient", forbidden)
    jev = Jev("navigate")
    state = drive(monkeypatch, tmp_path, Computer(), Reasoner([]), jev, goal="Open https://different.test")
    assert state.answer.achieved and len(jev.requests) == 2
    assert json.loads((tmp_path / "run.json").read_text())["calls"]["classifier"]["calls"] == 2


def test_the_structured_engine_refuses_to_start_without_a_classifier_factory(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "TypeSafeClient", lambda: pytest.fail("the hosted classifier was built for the engine"))
    cfg = runner.RunConfig("Task", tmp_path, act=True, engine="structured")
    with pytest.raises(ValueError, match="classifier_factory"):
        runner.run(cfg, lambda *a: None, control=RunControl("offline"))


def test_a_configuration_error_is_left_to_the_caller_but_the_run_folder_is_still_written(monkeypatch, tmp_path):
    def unconfigured():
        raise ConfigError("no usable provider for the classifier; set EXAMPLE_KEY")

    monkeypatch.setattr(engine, "make_backend", lambda _: pytest.fail("no browser before the classifier"))
    cfg = runner.RunConfig("Task", tmp_path, act=True, engine="structured", readiness_timeout=0)
    with pytest.raises(ConfigError, match="EXAMPLE_KEY"):
        runner.run(cfg, lambda *a: None, classifier_factory=unconfigured, control=RunControl("offline"))
    assert json.loads((tmp_path / "run.json").read_text())["outcome"] == "crashed"


def test_invalid_budgets_block_before_any_model_browser_or_journal(monkeypatch, tmp_path):
    for kwargs in (
        {"readiness_timeout": 31},
        {"readiness_timeout": float("nan")},
        {"steps": 0},
        {"steps": 501},
        {"min_confidence": 2},
    ):
        computer, writer, jev = Computer(), Reasoner([]), Jev()
        state = drive(monkeypatch, tmp_path / str(kwargs), computer, writer, jev, **kwargs)
        assert state.outcome == "blocked" and "Invalid execution budget" in state.failure
        assert not computer.actions and not writer.requests and not jev.requests and computer.reads == 0


def test_a_step_limit_names_what_remains(monkeypatch, tmp_path):
    computer = Computer()
    steps = [Milestone("tabs", "Open tabs", "tab_created", value="https://example.net", quantity=3)]
    state = drive(monkeypatch, tmp_path, computer, Reasoner([response(*steps)]), Jev("plan"), steps=2)
    assert state.outcome == "step limit" and len(computer.actions) == 2
    assert "budget ended" in state.failure and "Verified 2 effect(s); 1 remain." in state.failure

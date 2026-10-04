"""The file-backed recorder: private, bounded and credential-free, and only when content recording is opted into (D3).

Adapted from variant-6's recorder tests. What it dropped with its code: the latest-failure pointer (it needed a file
lock), the terminal tee, and the git and browser-settings metadata (a subprocess, and a module of another package).
"""

import json
import os
import stat
from urllib.error import URLError

import pytest

from glide.computer import diagnostics
from glide.computer.diagnostics import Diagnostics
from glide.providers.errors import ProviderError


def report(recorder):
    return json.loads((recorder.root / "diagnostic.json").read_text())


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def modes(root):
    return {p.name: stat.S_IMODE(p.stat().st_mode) for p in [root, *root.iterdir()]}


def test_default_recording_writes_nothing_and_finish_returns_no_report(tmp_path):
    recorder = Diagnostics(tmp_path / "run")
    with recorder.activate():
        diagnostics.event("started", goal="buy shoes")
        diagnostics.exception(ValueError("a page said https://a.test/secret"), stage="observe")
    assert recorder.finish({"outcome": "blocked"}) is None
    assert not (tmp_path / "run").exists()
    assert recorder.error == {"type": "ValueError", "message": "ValueError"}


@pytest.mark.skipif(os.name != "posix", reason="POSIX permissions")
def test_opted_in_files_are_private_and_complete(tmp_path):
    recorder = Diagnostics(tmp_path / "run", record_content=True, task_id="t1")
    recorder.event("stage_started", stage="inspect")
    assert lines(recorder.root / "events.jsonl")[0]["event"] == "stage_started"  # appended as it happens
    path = recorder.finish({"outcome": "done", "goal_achieved": True})
    assert path == recorder.root / "diagnostic.json"
    assert not (recorder.root / "failure.md").exists()
    assert modes(recorder.root) == {"run": 0o700, "events.jsonl": 0o600, "diagnostic.json": 0o600}
    assert report(recorder)["task_id"] == "t1" and report(recorder)["summary"]["outcome"] == "done"


def test_failure_report_names_stage_chain_and_frames_without_locals_or_source(tmp_path):
    recorder = Diagnostics(tmp_path / "run", record_content=True, task_id="task-9")
    private_local = "a-field-that-only-lives-in-locals"
    try:
        try:
            assert private_local
            raise ConnectionRefusedError(61, "connection refused")
        except ConnectionRefusedError as cause:
            raise RuntimeError("browser observation failed") from cause
    except RuntimeError as error:
        with recorder.activate():
            diagnostics.exception(error, stage="browser_preflight")
    path = recorder.finish({"outcome": "blocked", "failure_stage": "browser_preflight"})
    assert path == recorder.root / "failure.md"
    chain = report(recorder)["exception"]
    assert [e["type"] for e in chain] == ["RuntimeError", "ConnectionRefusedError"] and chain[1]["errno"] == 61
    assert chain[0]["frames"][-1]["file"] == __file__ and report(recorder)["error_stage"] == "browser_preflight"
    text = path.read_text()
    assert "browser_preflight" in text and "A cause has not been inferred" in text
    for item in recorder.root.iterdir():
        assert private_local not in item.read_text() and "raise RuntimeError" not in item.read_text()


def test_provider_error_kind_and_status_are_recorded_as_codes(tmp_path):
    recorder = Diagnostics(tmp_path / "run", record_content=True)
    recorder.finish({"outcome": "provider failure"}, error=ProviderError("x", kind="rate_limit", status=429))
    assert report(recorder)["exception"][0]["kind"] == "rate_limit" and report(recorder)["exception"][0]["status"] == 429


def test_urlerror_reason_is_not_lost_when_there_is_no_python_cause(tmp_path):
    recorder = Diagnostics(tmp_path / "run", record_content=True)
    recorder.finish({"outcome": "blocked"}, error=URLError(ConnectionRefusedError(61, "refused")))
    assert [e["type"] for e in report(recorder)["exception"]] == ["URLError", "ConnectionRefusedError"]


@pytest.mark.parametrize("content", [False, True])
def test_credentials_never_reach_memory_or_files(tmp_path, monkeypatch, content):
    secret = "private-api-secret-only-for-test"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    recorder = Diagnostics(tmp_path / "run", record_content=content)
    details = {
        "reason": f"key={secret}",
        "authorization": "Bearer arbitrary-secret",
        "nested": {"api_key": "another-secret", "password": "a-password"},
        "headers": {"Cookie": "sessionid=super-secret"},
        "endpoint": "https://alice:passcode@example.test/path?access_token=hidden-token",
        "input_tokens": 17,
    }
    recorder.event("failed", **details)
    recorder.finish({"outcome": "blocked", **details}, error=ValueError(f"Provider refused {secret}"))
    everything = str(list(recorder.events)) + str(recorder.error)
    if content:
        everything += "\n".join(p.read_text() for p in (tmp_path / "run").rglob("*") if p.is_file())
    for value in (
        secret,
        "arbitrary-secret",
        "another-secret",
        "a-password",
        "super-secret",
        "alice",
        "passcode",
        "hidden-token",
    ):
        assert value not in everything
    assert recorder.events[0]["details"]["input_tokens"] == 17


def test_without_content_urls_lose_their_path_and_content_fields_are_marked(tmp_path):
    recorder = Diagnostics(tmp_path / "run")
    recorder.event(
        "observation", url="https://example.test/search?q=my-query#f", page_text="private", action_params={"value": "v"}
    )
    details = recorder.events[0]["details"]
    assert details == {
        "url": "https://example.test",
        "page_text": "[content recording disabled]",
        "action_params": "[content recording disabled]",
    }


def test_events_rotate_and_the_report_keeps_the_last_ones(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostics, "MAX_EVENTS_BYTES", 1000)
    recorder = Diagnostics(tmp_path / "run", record_content=True)
    for index in range(80):
        recorder.event("observation", sequence=index, output="x" * 100)
    recorder.event("failed", reason="last observed failure")
    recorder.finish({"outcome": "blocked"})
    for name in ("events.jsonl", "events.previous.jsonl"):
        assert (recorder.root / name).stat().st_size <= 1000 and lines(recorder.root / name)
    assert report(recorder)["recent_events"][-1]["details"]["reason"] == "last observed failure"


def test_huge_events_and_summaries_stay_bounded_and_valid_json(tmp_path):
    recorder = Diagnostics(tmp_path / "run", record_content=True)
    nested = {str(i): ["x" * 50_000] * 100 for i in range(100)}
    recorder.event("huge", payload=nested)
    recorder.finish({"outcome": "blocked", "failure_stage": "verify", "history": nested}, error=ValueError("y" * 10_000_000))
    assert (recorder.root / "events.jsonl").stat().st_size <= diagnostics.MAX_EVENT_BYTES
    assert lines(recorder.root / "events.jsonl")[0]["details"]["truncated"]
    assert (recorder.root / "diagnostic.json").stat().st_size <= diagnostics.MAX_REPORT_BYTES
    assert (recorder.root / "failure.md").stat().st_size <= diagnostics.MAX_REPORT_BYTES
    assert report(recorder)["summary"]["failure_stage"] == "verify"


def test_a_write_that_fails_does_not_mask_the_original_error(tmp_path):
    blocked = tmp_path / "file"
    blocked.write_text("not a folder")
    recorder = Diagnostics(blocked / "run", record_content=True)
    error = ValueError("original observed error")
    with recorder.activate():
        diagnostics.event("started")
        diagnostics.exception(error, stage="inspect")
    assert recorder.finish({"outcome": "blocked"}, error=error) is None
    assert recorder.write_error and recorder.error["message"] == "original observed error"


@pytest.mark.parametrize("outcome", ["done", "cancelled", "dry run", "aborted (Ctrl-C)"])
def test_only_a_failed_run_gets_failure_md(tmp_path, outcome):
    recorder = Diagnostics(tmp_path / "run", record_content=True)
    assert recorder.finish({"outcome": outcome}) == recorder.root / "diagnostic.json"
    assert not (recorder.root / "failure.md").exists()


@pytest.mark.parametrize("outcome", ["blocked", "unsupported", "provider failure", "crashed", "done"])
def test_an_unachieved_goal_or_a_blocked_run_is_a_failure(tmp_path, outcome):
    recorder = Diagnostics(tmp_path / "run", record_content=True)
    assert recorder.finish({"outcome": outcome, "goal_achieved": False}) == recorder.root / "failure.md"


def test_finish_is_idempotent(tmp_path):
    recorder = Diagnostics(tmp_path / "run", record_content=True)
    first = recorder.finish({"outcome": "blocked", "failure_stage": "one"})
    assert recorder.finish({"outcome": "blocked", "failure_stage": "two"}) == first
    assert report(recorder)["summary"]["failure_stage"] == "one"


def test_a_folder_in_use_is_never_overwritten(tmp_path):
    base = tmp_path / "run"
    first = Diagnostics(base, record_content=True)
    first.event("first")
    first.finish({"outcome": "blocked"})
    before = (base / "diagnostic.json").read_bytes()
    second, third = Diagnostics(base, record_content=True), Diagnostics(base, record_content=True)
    second.event("second")
    third.event("third")
    assert len({second.root, third.root, base}) == 3 and second.requested_root == base
    second.finish({"outcome": "done"})
    third.finish({"outcome": "done"})
    assert (base / "diagnostic.json").read_bytes() == before
    assert [e["event"] for e in lines(second.root / "events.jsonl")] == ["second"]
    assert [e["event"] for e in lines(third.root / "events.jsonl")] == ["third"]


def test_runtime_lists_glide_setting_names_never_values(tmp_path, monkeypatch):
    # PR10-4175413698: glide.toml allows any capitals-only api_key_env, so a GLIDE_* value can be a provider key
    for name in [n for n in os.environ if n.startswith("GLIDE_")]:
        monkeypatch.delenv(name)
    monkeypatch.setenv("GLIDE_ENGINE_SETTING", "visible-value")
    monkeypatch.setenv("GLIDE_ACME", "sk-live-ABCDEF123456")
    monkeypatch.setenv("SOME_PRIVATE_ENV", "never-collect-this-variable")
    recorder = Diagnostics(tmp_path / "run", record_content=True)
    recorder.finish({"outcome": "provider failure"})
    for path in (recorder.root / "diagnostic.json", recorder.root / "failure.md"):
        text = path.read_text()
        assert "GLIDE_ACME" in text and "GLIDE_ENGINE_SETTING" in text
        assert "sk-live-ABCDEF123456" not in text and "visible-value" not in text
        assert "never-collect-this-variable" not in text and "SOME_PRIVATE_ENV" not in text
    assert report(recorder)["runtime"]["settings"] == ["GLIDE_ACME", "GLIDE_ENGINE_SETTING"]


@pytest.mark.skipif(os.name != "posix", reason="POSIX permissions")
def test_write_private_replaces_whole_and_is_0600(tmp_path):
    path = tmp_path / "run.json"
    path.write_text("old")
    diagnostics.write_private(path, '{"a": 1}')
    assert path.read_text() == '{"a": 1}' and stat.S_IMODE(path.stat().st_mode) == 0o600
    assert [p.name for p in tmp_path.iterdir()] == ["run.json"]


def test_public_redactor_handles_json_and_every_cookie_pair(monkeypatch):
    monkeypatch.setenv("PROVIDER_API_KEY", "known-secret")
    result = diagnostics.redact('known-secret {"token": "unknown-secret"}\nCookie: session=abc; user=def\nBearer abc.def')
    for value in ("known-secret", "unknown-secret", "session=abc", "user=def", "abc.def"):
        assert value not in result

"""What the browser backends record when the Playwright CLI or a CDP endpoint fails, using only fake processes and sockets.

Ported from variant-6's transport diagnostics. The process is replaced at the one seam (`execution.spawn.start`), and
the recorder is the in-memory `Diagnostics` of this tree rather than a file.
"""

import json
import urllib.error
from contextlib import nullcontext

import pytest

from glide.computer import diagnostics
from glide.computer.browser import cdp
from glide.computer.execution import dom, playwright_cli, spawn
from glide.computer.models import BrowserConnectionError

GET_JSON = cdp._get_json  # the real function, taken before the guard replaces it; each test feeds it a fake opener


@pytest.fixture
def events(monkeypatch):
    captured = []

    def collect(name, **details):
        captured.append((name, details))

    for module in (cdp, dom, playwright_cli):
        monkeypatch.setattr(module, "event", collect)
    return captured


def test_http_error_is_recorded_before_browser_connection_wrapping(monkeypatch, events):
    error = urllib.error.URLError(ConnectionRefusedError(61, "connection refused"))

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(cdp._DIRECT, "open", fail)  # loopback is asked through the no-proxy opener
    monkeypatch.setattr(dom, "_get_json", GET_JSON)
    backend = dom.BrowserBackend("http://127.0.0.1:9417")
    with pytest.raises(BrowserConnectionError) as caught:
        backend.connect()
    assert caught.value.__cause__ is error
    http = next(details for name, details in events if name == "browser_http_failed")
    assert http["endpoint"] == "http://127.0.0.1:9417/json/version"
    assert http["exception"] is error and http["elapsed_s"] >= 0
    assert events[-1][0] == "browser_connection_failed"


def fake_cli(monkeypatch, raw, code=1):
    monkeypatch.setattr(playwright_cli, "executable", lambda cli="": "/fixture/playwright-cli")
    monkeypatch.setattr(playwright_cli, "operation_gate", lambda: nullcontext("http://fixture/gate"))

    class Process:
        returncode = code

        def __init__(self, *args, **kwargs):
            kwargs["stdout"].write(raw)

        def poll(self):
            return self.returncode

    monkeypatch.setattr(spawn, "start", Process)


@pytest.mark.parametrize("raw", [b"daemon disconnected", b'{"isError":true,"error":"Browser is missing"}'])
def test_cli_failure_retains_exit_code_and_original_provider_output(monkeypatch, events, raw):
    fake_cli(monkeypatch, raw)
    with pytest.raises(cdp.CDPError) as caught:
        playwright_cli.CliSession("fixture").call("Target.getTargets")
    name, detail = events[-1]
    assert name == "browser_request_failed" and detail["provider"] == "playwright-cli"
    assert detail["endpoint"] == "playwright:fixture" and detail["exit_code"] == 1
    assert detail["exception"] is caught.value
    assert detail["output"] == raw.decode()


def test_cli_failure_output_is_bounded_and_success_omits_it(monkeypatch, events):
    raw = b"x" * 10000 + b"actual failure"
    fake_cli(monkeypatch, raw)
    with pytest.raises(cdp.CDPError):
        playwright_cli.CliSession("fixture").call("Target.getTargets")
    assert len(events[-1][1]["output"]) <= playwright_cli.DIAGNOSTIC_OUTPUT
    assert events[-1][1]["output"].endswith("actual failure")

    fake_cli(monkeypatch, b'{"result":"{}"}', code=0)
    assert playwright_cli.CliSession("fixture").call("Target.getTargets") == {}
    assert events[-1][0] == "browser_request_completed"
    assert "output" not in events[-1][1]


def test_cli_timeout_retains_partial_output_after_stopping_client(monkeypatch, events):
    monkeypatch.setattr(playwright_cli, "executable", lambda cli="": "/fixture/playwright-cli")
    monkeypatch.setattr(playwright_cli, "operation_gate", lambda: nullcontext("http://fixture/gate"))
    monkeypatch.setattr(playwright_cli, "MAX_OUTPUT", 8)
    stopped = []

    class Process:
        returncode = None

        def __init__(self, *args, **kwargs):
            kwargs["stdout"].write(b"daemon blocked")

        def poll(self):
            return self.returncode

        def terminate(self):
            stopped.append(True)
            self.returncode = -15

        def wait(self, **kwargs):
            return self.returncode

    monkeypatch.setattr(spawn, "start", Process)
    with pytest.raises(cdp.CDPError, match="completion unknown"):
        playwright_cli.CliSession("fixture").call("Target.getTargets")
    assert stopped == [True]
    assert events[-1][1]["output"] == ""
    assert events[-1][1]["output_omitted"] == "partial_line"
    assert events[-1][1]["exit_code"] == -15


@pytest.mark.parametrize("kind", ["known", "bearer"])
def test_cli_redacts_credentials_before_cutting_diagnostic_tail(monkeypatch, kind):
    monkeypatch.setattr(playwright_cli, "DIAGNOSTIC_OUTPUT", 64)
    secret = "fixture-tail-boundary-credential-value"
    if kind == "known":
        monkeypatch.setenv("GLIDE_WRITER_API_KEY", secret)
        prefix = "provider notice\n"
    else:
        prefix = "provider notice\nAuthorization: Bearer "
    suffix = "\nfinal provider failure"
    padding = " " * (64 - len(secret) // 2 - len(suffix))
    raw = (prefix + secret + padding + suffix).encode()
    # Without redaction first, the retained text begins inside the credential.
    assert raw[-64:].startswith(secret[len(secret) // 2 :].encode())
    fake_cli(monkeypatch, raw)
    recorder = diagnostics.Diagnostics(record_content=True)
    with recorder.activate(), pytest.raises(cdp.CDPError):
        playwright_cli.CliSession("fixture").call("Target.getTargets")
    log = json.dumps(list(recorder.events))
    assert secret not in log and secret[len(secret) // 2 :] not in log
    assert "final provider failure" in log
    assert "REDACTED" in log


@pytest.mark.parametrize("complete_line", [False, True])
def test_oversized_cli_output_discards_partial_credential_line_before_redaction(monkeypatch, complete_line):
    monkeypatch.setattr(playwright_cli, "MAX_OUTPUT", 64)
    monkeypatch.setattr(playwright_cli, "DIAGNOSTIC_OUTPUT", 48)
    secret = "fixture-oversized-output-credential-value"
    monkeypatch.setenv("GLIDE_WRITER_API_KEY", secret)
    raw = b"discarded prefix\n" * 5 + b"Authorization: Bearer " + secret.encode() + b" " * 15
    if complete_line:
        raw += b"\nprovider completion unknown"
    assert secret[-10:].encode() in raw[-64:]
    fake_cli(monkeypatch, raw)
    recorder = diagnostics.Diagnostics(record_content=True)
    with recorder.activate(), pytest.raises(cdp.CDPError, match="exceeded its limit"):
        playwright_cli.CliSession("fixture").call("Target.getTargets")
    log = json.dumps(list(recorder.events))
    details = recorder.events[-1]["details"]
    assert secret[-10:] not in log and secret not in log
    assert details["exit_code"] == 1
    assert details["exception"]["type"] == "CDPError"
    if complete_line:
        assert details["output"] == "provider completion unknown"
    else:
        assert details["output"] == "" and details["output_omitted"] == "partial_line"


def test_without_content_opt_in_the_cli_output_is_not_recorded_at_all(monkeypatch):
    fake_cli(monkeypatch, b"user typed: my unseen query")
    recorder = diagnostics.Diagnostics(record_content=False)
    with recorder.activate(), pytest.raises(cdp.CDPError):
        playwright_cli.CliSession("fixture").call("Input.insertText", {"text": "my unseen query"}, session_id="tab")
    assert "my unseen query" not in json.dumps(list(recorder.events))

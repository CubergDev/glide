"""Provider selection, fallback, the fixed-code Playwright CLI transport and cancellation, all without a browser.

Nothing here starts a process, opens a socket to a browser, or runs Node. The Playwright CLI is replaced at the one
process seam (`execution.spawn.start`) by a fake that reads the generated command file like the CLI would.
"""

import json
import shutil
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from urllib.error import URLError

import pytest

from glide.computer import browser_settings, desktop_access, diagnostics
from glide.computer.browser.cdp import CDPError
from glide.computer.control import RunControl, controlled
from glide.computer.execution import dom, playwright_cli, providers, spawn
from glide.computer.execution.contracts import Action, Element, InvalidAction, Observation, UnsupportedCapability
from glide.computer.execution.obscura import ObscuraBackend
from glide.computer.models import Abort, BrowserConnectionError, DesktopError

ENV_NAMES = (
    "GLIDE_BROWSER_PROVIDER",
    "GLIDE_BROWSER_ENDPOINT",
    "GLIDE_BROWSER_TARGET",
    "GLIDE_PLAYWRIGHT_SESSION",
    "GLIDE_PLAYWRIGHT_CLI",
)
REAL_MAKE_BACKEND = providers.make_backend  # taken at import, before the autouse guard replaces it
CDP = "http://127.0.0.1:9415"
OBSCURA = "http://127.0.0.1:9742"


@pytest.fixture(autouse=True)
def clean_settings(monkeypatch):
    for name in ENV_NAMES:
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    monkeypatch.setattr(browser_settings, "_table", {})
    monkeypatch.setattr(providers, "make_backend", REAL_MAKE_BACKEND)  # this file tests the door the guard shuts


def configure(**table):
    table.setdefault("cdp", {"endpoint": CDP})
    table.setdefault("obscura", {"endpoint": OBSCURA})
    table.setdefault("playwright", {"session": "fixture"})
    browser_settings.use(table)


# -- selection --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "provider,transport", [("native", "native"), ("cdp", "cdp"), ("obscura", "obscura"), ("playwright", "playwright-cli")]
)
def test_selection_is_passive_and_names_the_configured_provider(monkeypatch, provider, transport):
    configure(provider=provider, target="new")
    monkeypatch.setattr(dom, "_get_json", lambda *a, **kw: pytest.fail("Selecting a provider connected to it"))
    backend = providers.make_backend("Brave Browser")
    assert backend.transport == transport


def test_the_unconfigured_default_is_the_native_desktop_with_the_configured_browser(monkeypatch):
    backend = providers.make_backend()
    assert backend.transport == "native" and backend.browser  # the default browser name comes from the core config


def test_the_selected_provider_and_where_it_came_from_are_recorded():
    configure(provider="obscura")
    with diagnostics.Diagnostics().activate() as recorder:
        providers.make_backend()
    events = [e for e in recorder.events if e["event"] == "browser_provider_selected"]
    assert events and events[0]["details"]["provider"] == "obscura" and events[0]["details"]["source"] == "glide.toml"


def test_obscura_uses_its_endpoint_and_explicit_selection_wins(monkeypatch):
    configure(provider="obscura")
    assert providers.make_backend().origin == OBSCURA
    monkeypatch.setenv("GLIDE_BROWSER_ENDPOINT", "http://127.0.0.1:9555")
    assert providers.make_backend().origin == "http://127.0.0.1:9555"


def test_playwright_session_ignores_stale_cdp_endpoint():
    configure(provider="playwright", playwright={"session": "firefox-work"})
    backend = providers.make_backend()
    assert backend.origin == "playwright:firefox-work" and backend.transport == "playwright-cli"
    assert not hasattr(backend, "port")


def test_playwright_gets_the_configured_cli_and_permission_to_act():
    configure(provider="playwright", playwright={"session": "s", "cli": "/opt/playwright-cli"})
    backend = providers.make_backend(act=True)
    assert backend.cli == "/opt/playwright-cli" and backend.allow_actions
    assert not providers.make_backend(act=False).allow_actions


@pytest.mark.parametrize(
    "table",
    [
        {"provider": "bogus"},
        {"provider": "cdp", "cdp": {}},
        {"provider": "obscura", "obscura": {"endpoint": "https://remote.test"}},
        {"provider": "playwright", "playwright": {"session": "-s=bad;echo"}},
    ],
)
def test_bad_config_cannot_fall_back_to_native(monkeypatch, table):
    monkeypatch.setattr(browser_settings, "_table", table)
    built = []
    monkeypatch.setattr(providers, "_build", lambda *args: built.append(args))
    with pytest.raises(DesktopError):
        providers.make_backend("Brave Browser")
    assert built == []


@pytest.mark.parametrize("provider", ["cdp", "obscura", "playwright"])
def test_provider_is_independent_of_native_browser_and_desktop_permissions(monkeypatch, provider):
    configure(provider=provider, target="new")
    monkeypatch.setattr(desktop_access, "prepare_desktop", lambda *a, **kw: pytest.fail("Browser provider reached desktop"))
    built = []
    real = providers._build

    def build(*args):
        backend = real(*args)
        seen = []
        monkeypatch.setattr(backend, "inspect", lambda: seen.append("inspect"))
        monkeypatch.setattr(backend, "close", lambda: seen.append("close"))
        built.append((backend, seen))
        return backend

    monkeypatch.setattr(providers, "_build", build)
    providers.prepare(True)
    ((backend, seen),) = built
    assert seen == ["inspect", "close"] and backend.transport in {provider, "playwright-cli"}


def test_native_prepare_is_the_desktop_permission_preflight(monkeypatch):
    asked = []
    monkeypatch.setattr(desktop_access, "prepare_desktop", lambda act, **kw: asked.append(act))
    providers.prepare(True)
    assert asked == [True]


def test_voice_preparation_defers_the_browser_check_without_connecting(monkeypatch):
    configure(provider="cdp")
    monkeypatch.setattr(dom, "_get_json", lambda *a, **kw: pytest.fail("Voice preparation connected to the browser"))
    with diagnostics.Diagnostics().activate() as recorder:
        providers.prepare_voice(True)
    assert [e["event"] for e in recorder.events if e["event"] == "browser_check_deferred"]
    with pytest.raises(KeyboardInterrupt):
        providers.prepare_voice(True, cancelled=lambda: True)


# -- failure is actionable, takes no action, and does not switch ------------------------------------------------


@pytest.mark.parametrize("factory", [dom.BrowserBackend, ObscuraBackend])
def test_refused_connection_is_actionable_and_takes_no_action(monkeypatch, factory):
    seen = []

    def refused(url, **kw):
        seen.append(url)
        raise URLError(ConnectionRefusedError(61, "secret page content must not be logged"))

    monkeypatch.setattr(dom, "_get_json", refused)
    backend = factory(CDP, "new")
    monkeypatch.setattr(backend, "execute", lambda *a, **kw: pytest.fail("An action was taken without a connection"))
    with diagnostics.Diagnostics().activate() as recorder, pytest.raises(BrowserConnectionError) as caught:
        backend.inspect()
    assert seen == [CDP + "/json/version"]
    assert "9415" in str(caught.value) and "Start or reconnect" in str(caught.value)
    assert caught.value.details == {
        "provider": backend.transport,
        "endpoint": backend.origin,
        "error_type": "URLError",
        "cause_type": "ConnectionRefusedError",
        "errno": 61,
    }
    assert "secret page content" not in str(caught.value) + json.dumps(caught.value.details) + json.dumps(list(recorder.events))


def test_preflight_failure_closes_connection_and_does_not_fall_back(monkeypatch):
    configure(provider="cdp")
    closed = []

    def failed(self):
        raise BrowserConnectionError("cdp", self.origin, ConnectionRefusedError(61, "refused"))

    monkeypatch.setattr(dom.BrowserBackend, "inspect", failed)
    monkeypatch.setattr(dom.BrowserBackend, "close", lambda self: closed.append(True))
    monkeypatch.setattr(desktop_access, "prepare_desktop", lambda *a, **kw: pytest.fail("Fell back to the native desktop"))
    with pytest.raises(BrowserConnectionError):
        providers.prepare(True)
    assert closed == [True]


# -- the optional, visible fallback ----------------------------------------------------------------------------


class Reachable:
    """Replaces `dom._get_json`: which origins answer."""

    def __init__(self, *answering):
        self.answering, self.asked = answering, []

    def __call__(self, url, **kw):
        origin = url.removesuffix("/json/version")
        self.asked.append(origin)
        if origin not in self.answering:
            raise URLError(ConnectionRefusedError(61, "refused"))
        return {"webSocketDebuggerUrl": "ws://" + origin.removeprefix("http://") + "/devtools/browser/x"}


@pytest.fixture
def sessions(monkeypatch):
    opened = []
    monkeypatch.setattr(dom, "Session", lambda url, **kw: opened.append(url) or SimpleNamespace(close=lambda: None))
    return opened


def test_unavailable_provider_is_replaced_by_the_next_one_and_the_switch_is_visible(monkeypatch, sessions):
    configure(provider="cdp", fallback=["obscura"])
    monkeypatch.setattr(dom, "_get_json", Reachable(OBSCURA))
    heard = []
    with diagnostics.Diagnostics().activate() as recorder:
        backend = providers.make_backend(on_switch=heard.append)
    assert backend.transport == "obscura"
    (switch,) = heard
    assert (switch.role, switch.from_slot, switch.to_slot, switch.kind) == ("browser", "cdp", "obscura", "BrowserConnectionError")
    assert "Cannot connect to cdp" in switch.reason and "secret" not in switch.reason
    kinds = [e["event"] for e in recorder.events]
    assert "browser_provider_switched" in kinds and kinds.index("browser_provider_selected") < kinds.index(
        "browser_provider_switched"
    )


def test_a_fallback_is_always_said_on_the_runs_channel_whether_or_not_the_caller_listens(monkeypatch, sessions):
    """Audit finding 2: with no `on_switch` the switch used to reach only the in-memory recorder."""
    configure(provider="cdp", fallback=["obscura"])
    monkeypatch.setattr(dom, "_get_json", Reachable(OBSCURA))
    heard = []
    with diagnostics.Diagnostics().activate() as recorder, controlled(RunControl("t", heard.append)):
        assert providers.make_backend().transport == "obscura"
    (said,) = [e for e in heard if e.kind == "switch"]
    assert said.text == "fallback: browser cdp -> obscura (BrowserConnectionError)"
    assert "127.0.0.1" not in said.text and "browser_provider_switched" in [e["event"] for e in recorder.events]


def test_the_first_available_provider_is_used_without_a_switch(monkeypatch, sessions):
    configure(provider="cdp", fallback=["obscura"])
    probe = Reachable(CDP, OBSCURA)
    monkeypatch.setattr(dom, "_get_json", probe)
    heard = []
    assert providers.make_backend(on_switch=heard.append).transport == "cdp"
    assert heard == [] and probe.asked == [CDP]


def test_when_nothing_is_available_the_last_error_is_raised_and_each_switch_was_reported(monkeypatch, sessions):
    configure(provider="cdp", fallback=["obscura"])
    monkeypatch.setattr(dom, "_get_json", Reachable())
    heard = []
    with pytest.raises(BrowserConnectionError, match="obscura"):
        providers.make_backend(on_switch=heard.append)
    assert [(s.from_slot, s.to_slot) for s in heard] == [("cdp", "obscura")]


def test_playwright_as_a_fallback_must_be_installed_and_allowed_to_act(monkeypatch, sessions):
    configure(provider="cdp", fallback=["playwright"])
    monkeypatch.setattr(dom, "_get_json", Reachable())
    monkeypatch.setattr(shutil, "which", lambda _: None)
    with pytest.raises(DesktopError, match="not installed"):
        providers.make_backend(act=True)
    monkeypatch.setattr(shutil, "which", lambda _: "/fixture/playwright-cli")
    with pytest.raises(DesktopError, match="not enabled"):
        providers.make_backend(act=False)
    assert providers.make_backend(act=True).transport == "playwright-cli"


def test_no_fallback_configured_means_no_probe_and_no_switch(monkeypatch):
    configure(provider="cdp")
    monkeypatch.setattr(dom, "_get_json", lambda *a, **kw: pytest.fail("make_backend probed a provider with no fallback"))
    assert providers.make_backend().transport == "cdp"


def test_the_native_desktop_is_never_a_fallback(monkeypatch):
    monkeypatch.setattr(browser_settings, "_table", {"provider": "cdp", "cdp": {"endpoint": CDP}, "fallback": ["native"]})
    with pytest.raises(DesktopError, match="never a fallback"):
        providers.make_backend()


def test_a_provider_that_failed_after_selection_is_not_swapped_for_another(monkeypatch, sessions):
    """The fallback applies before anything is done. After that a failure is the caller's to see, never replayed elsewhere."""
    configure(provider="cdp", fallback=["obscura"])
    probe = Reachable(CDP, OBSCURA)
    monkeypatch.setattr(dom, "_get_json", probe)

    def failed(self):
        raise BrowserConnectionError("cdp", self.origin, ConnectionResetError(54, "reset"))

    monkeypatch.setattr(dom.BrowserBackend, "inspect", failed)
    heard = []
    with pytest.raises(BrowserConnectionError):
        providers.prepare(True, on_switch=heard.append)
    assert heard == [] and OBSCURA not in probe.asked


# -- the Playwright CLI: where it is, and what it is -----------------------------------------------------------


@pytest.mark.parametrize("package", ["@playwright/cli", "playwright", "playwright-core"])
def test_cli_resolves_microsoft_agent_package_not_the_test_runner(monkeypatch, tmp_path, package):
    executable = tmp_path / "entry.js"
    executable.write_text("// offline fixture")
    (tmp_path / "package.json").write_text(json.dumps({"name": package}))
    monkeypatch.setattr(shutil, "which", lambda _: str(executable))
    if package == "@playwright/cli":
        assert playwright_cli.executable() == str(executable)
    else:
        with pytest.raises(DesktopError, match="Microsoft's @playwright/cli"):
            playwright_cli.executable()


def test_cli_is_found_only_at_the_configured_path_or_on_path(monkeypatch, tmp_path):
    asked = []
    monkeypatch.setattr(shutil, "which", lambda name: asked.append(name))
    with pytest.raises(DesktopError, match=r"not installed.*GLIDE_PLAYWRIGHT_CLI"):
        playwright_cli.executable()
    with pytest.raises(DesktopError):
        playwright_cli.executable("/configured/playwright-cli")
    assert asked == ["playwright-cli", "/configured/playwright-cli"]
    # A binary left in the working directory is never picked up.
    local = tmp_path / ".private/browser-providers/node_modules/.bin"
    local.mkdir(parents=True)
    (local / "playwright-cli").write_text("#!/bin/sh\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(DesktopError, match="not installed"):
        playwright_cli.executable()


def test_cli_cannot_recreate_a_tab_when_actions_are_disabled():
    backend = playwright_cli.PlaywrightBackend("fixture", "new")
    with pytest.raises(UnsupportedCapability, match="passive inspection"):
        backend.inspect()
    with pytest.raises(UnsupportedCapability, match="disabled"):
        backend.execute(Action("inspect", "x"), Observation("browser", "x", capabilities={"inspect"}))
    assert backend.browser is None


def test_cli_backend_names_what_to_enable(monkeypatch):
    backend = playwright_cli.PlaywrightBackend("fixture", "new")
    with pytest.raises(DesktopError, match=r"not enabled.*Allow computer actions"):
        backend.check()
    monkeypatch.setattr(shutil, "which", lambda _: None)
    backend.allow_actions = True
    with pytest.raises(DesktopError, match="not installed"):
        backend.check()


# -- the Playwright CLI transport, through the process seam ----------------------------------------------------


def command_of(args):
    return Path(next(a.removeprefix("--filename=") for a in args if a.startswith("--filename="))).read_text()


def fake_cli(monkeypatch, replies, *, control=None, returncode=0, output=None):
    """Replace the seam. Each started process answers with the next of `replies` (a value, written as JSON)."""
    seen = []
    monkeypatch.setattr(playwright_cli, "executable", lambda cli="": cli or "/fixture/playwright-cli")

    class Process:
        def __init__(self, args, **options):
            code = command_of(args)
            seen.append((args, options, code))
            if control:
                control.cancel()
            reply = replies[min(len(seen) - 1, len(replies) - 1)]
            options["stdout"].write(output if output is not None else json.dumps(reply).encode())
            self.returncode, self.polls = returncode, 0

        def poll(self):
            self.polls += 1
            return None if self.polls == 1 else self.returncode

    monkeypatch.setattr(spawn, "start", Process)
    return seen


def test_cli_uses_fixed_file_json_data_no_shell_or_secrets(monkeypatch):
    text = "你好'; process.exit(7); //\n$(echo danger)"
    monkeypatch.setenv("GLIDE_WRITER_API_KEY", "fixture-secret")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "fixture-secret")
    seen = fake_cli(monkeypatch, [{"result": "{}"}])
    playwright_cli.CliSession("fixture").call("Input.insertText", {"text": text}, session_id="tab-1")
    args, options, code = seen[0]
    assert args[1:3] == ["-s=fixture", "run-code"] and "--json" in args
    assert not options.get("shell") and all(text not in a for a in args)
    assert options["stdin"] is spawn.DEVNULL
    assert not any("fixture-secret" in v for v in options["env"].values())
    assert set(options["env"]) <= {*playwright_cli.CHILD_ENVIRONMENT, "NO_UPDATE_NOTIFIER"}
    request = json.loads(code.split("const request = ", 1)[1].split(";\n", 1)[0])
    assert request["params"]["text"] == text and request["target"] == "tab-1" and request["method"] == "Input.insertText"
    assert options["env"]["NO_UPDATE_NOTIFIER"] == "1"
    assert not Path(args[3].removeprefix("--filename=")).exists()


def test_cli_passes_the_configured_program_and_session(monkeypatch):
    seen = fake_cli(monkeypatch, [{"result": "{}"}])
    playwright_cli.CliSession("work_1", "/opt/playwright-cli").call("Target.getTargets")
    assert seen[0][0][:2] == ["/opt/playwright-cli", "-s=work_1"]


def test_cancelled_cli_waits_for_dispatched_write_then_aborts(monkeypatch):
    control = RunControl()
    seen = fake_cli(monkeypatch, [{"result": "{}"}], control=control)
    with controlled(control), pytest.raises(Abort):
        playwright_cli.CliSession("fixture").call("Input.insertText", {"text": "value"}, session_id="tab")
    assert len(seen) == 1
    with controlled(control), pytest.raises(Abort):
        playwright_cli.CliSession("fixture").call("Page.navigate", {"url": "https://example.test"})
    assert len(seen) == 1


@pytest.mark.parametrize("value", [{"isError": True, "error": "fixture failure"}, {"result": "bad JSON"}, {"result": "null"}])
def test_cli_terminal_errors_cannot_produce_executable_results(monkeypatch, value):
    fake_cli(monkeypatch, [value])
    with pytest.raises(CDPError):
        playwright_cli.CliSession("fixture").call("Target.getTargets")


def test_a_failing_exit_code_is_a_failure_even_with_a_well_formed_reply(monkeypatch):
    fake_cli(monkeypatch, [{"result": "{}"}], returncode=1)
    with pytest.raises(CDPError, match="Playwright operation failed"):
        playwright_cli.CliSession("fixture").call("Target.getTargets")


def test_cli_output_limit_terminates_only_its_client(monkeypatch):
    stopped = []
    monkeypatch.setattr(playwright_cli, "executable", lambda cli="": "/fixture/playwright-cli")
    monkeypatch.setattr(playwright_cli, "MAX_OUTPUT", 32)

    class Process:
        def __init__(self, args, **kw):
            self.returncode = None
            kw["stdout"].write(b"x" * 64)

        def poll(self):
            return self.returncode

        def terminate(self):
            stopped.append("client")
            self.returncode = -15

        def wait(self, **kw):
            return self.returncode

    monkeypatch.setattr(spawn, "start", Process)
    with pytest.raises(CDPError, match="completion unknown"):
        playwright_cli.CliSession("fixture").call("Target.getTargets")
    assert stopped == ["client"]


def test_cli_rejects_arbitrary_methods_and_does_not_close_daemon():
    session = playwright_cli.CliSession("fixture")
    with pytest.raises(CDPError, match="Unsupported"):
        session.call("Browser.close")
    assert session.call("Target.detachFromTarget") == {}
    session.close()


def test_cli_process_remains_guarded_offline(monkeypatch):
    monkeypatch.setattr(playwright_cli, "executable", lambda cli="": "/fixture/playwright-cli")
    with pytest.raises(RuntimeError, match="real machine"):
        playwright_cli.CliSession("fixture").call("Target.getTargets")


def test_a_read_that_hits_a_navigation_is_retried_through_the_cli(monkeypatch):
    """The CLI's own error text for a destroyed context is told apart from a dead session, so the read is retried."""
    snapshot = {"result": json.dumps({"result": {"value": {"url": "https://x.test", "document_id": "9", "ready": True}}})}
    navigating = {"isError": True, "error": "page.evaluate: Execution context was destroyed, most likely because of a navigation"}
    targets = {"result": json.dumps({"targetInfos": [{"type": "page", "targetId": "glide-1", "url": "https://x.test"}]})}
    attach = {"result": json.dumps({"sessionId": "glide-1"})}
    seen = fake_cli(monkeypatch, [targets, attach, navigating, snapshot])
    monkeypatch.setattr(dom.time, "sleep", lambda _: None)
    backend = playwright_cli.PlaywrightBackend("fixture", "glide-1", allow_actions=True)
    observed = backend.inspect(controls=False)
    assert observed.owner == "glide-1:9" and len(seen) == 4


def test_a_dead_cli_session_is_a_connection_failure_with_the_playwright_hint(monkeypatch):
    fake_cli(monkeypatch, [{"isError": True, "error": "no session named fixture is open"}])
    backend = playwright_cli.PlaywrightBackend("fixture", "new", allow_actions=True)
    with pytest.raises(BrowserConnectionError, match="Open the selected Playwright CLI session"):
        backend.inspect()


def test_cli_diagnostics_carry_no_credentials_or_typed_text(monkeypatch):
    leaking = (b"Authorization: Bearer abcdef123456secret\napi_key=hunter2hunter2\nuser typed: my unseen query\n",)
    fake_cli(monkeypatch, [None], output=leaking[0], returncode=1)
    for record_content in (False, True):
        with diagnostics.Diagnostics(record_content=record_content).activate() as recorder, pytest.raises(CDPError):
            playwright_cli.CliSession("fixture").call("Input.insertText", {"text": "my unseen query"}, session_id="tab")
        blob = json.dumps(list(recorder.events))
        assert "abcdef123456secret" not in blob and "hunter2hunter2" not in blob
        if not record_content:
            assert "my unseen query" not in blob


def test_operation_gate_closes_after_stop_and_wrong_identity():
    control = RunControl()
    with controlled(control), playwright_cli.operation_gate() as gate:

        def read(url):
            with urllib.request.urlopen(url) as reply:
                return json.load(reply)["allowed"]

        assert read(gate)
        assert not read(gate + "wrong")
        control.cancel()
        assert not read(gate)


# -- Obscura and CDP ---------------------------------------------------------------------------------------------


def test_obscura_omits_browser_origin_without_changing_loopback_validation(monkeypatch):
    seen = []
    monkeypatch.setattr(dom, "_get_json", lambda *a, **kw: {"webSocketDebuggerUrl": "ws://127.0.0.1:9417/devtools/browser"})
    monkeypatch.setattr(dom, "Session", lambda *a, **kw: seen.append(kw) or SimpleNamespace())
    ObscuraBackend("http://127.0.0.1:9417").connect()
    assert seen[0]["suppress_origin"]
    dom.BrowserBackend("http://127.0.0.1:9417").connect()
    assert seen[1]["suppress_origin"]  # Chrome refuses any Origin its launch flags did not allow, so none is sent


@pytest.mark.parametrize("backend_type", [dom.BrowserBackend, ObscuraBackend])
@pytest.mark.parametrize(
    "endpoint", ["https://127.0.0.1:9222", "http://example.test:9222", "http://u:p@127.0.0.1:9222", "http://127.0.0.1"]
)
def test_only_loopback_http_origins_are_accepted(backend_type, endpoint):
    with pytest.raises(DesktopError, match="loopback"):
        backend_type(endpoint)


@pytest.mark.parametrize(
    "factory", [lambda: dom.BrowserBackend("http://127.0.0.1:9417"), lambda: ObscuraBackend("http://127.0.0.1:9417")]
)
def test_cdp_providers_remain_guarded_offline(factory):
    with pytest.raises(RuntimeError, match="real machine"):
        factory().inspect()


def test_obscura_replaces_observed_input_and_rejects_unsupported_editors():
    calls = []
    obs = Observation(
        "browser", "document", elements={"field": Element("field", "Search", "input", "old", True)}, capabilities={"type", "key"}
    )
    backend = ObscuraBackend("http://127.0.0.1:9417")
    backend.page = SimpleNamespace(
        evaluate=lambda expr: calls.append(expr) or True,
        call=lambda method, data: calls.append((method, data)) or {},
    )
    backend.execute(Action("type", obs.identity, "field", "新しい query"), obs)
    assert any(isinstance(c, str) and "setSelectionRange(0,e.value.length)" in c for c in calls)
    assert calls[-1] == ("Input.insertText", {"text": "新しい query"})
    assert not any(isinstance(c, tuple) and "commands" in c[1] for c in calls)
    backend.page.evaluate = lambda expr: False
    with pytest.raises(UnsupportedCapability, match="outside input"):
        backend.execute(Action("type", obs.identity, "field", "new"), obs)
    with pytest.raises(UnsupportedCapability, match="keyboard shortcut"):
        backend.execute(Action("key", obs.identity, value="a", modifiers=("command",), parameter_source="request"), obs)
    with pytest.raises(InvalidAction):
        backend.execute(Action("type", "stale", "field", "new"), obs)

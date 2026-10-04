"""The seams where the platform adapters meet the rest of the machine: the shared escape hatch, the
AppleScript argv boundary, the osascript deadline and cancel, the dispatch gate, and loopback-only CDP.

Everything here runs offline against fakes. `tests/conftest.py` refuses the real functions in every test
(`macos.osascript`, `windows.open_url`, `cdp._get_json`, `websocket.create_connection`, `subprocess.Popen`),
so the real ones are captured at import time, before any guard runs, and each test fakes only the layer
below the one it exercises.
"""

import contextvars
import json
import socket
import subprocess
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from glide.computer import macos, platform_adapter, windows
from glide.computer.browser import cdp
from glide.computer.control import RunControl, controlled, dispatch
from glide.computer.models import Abort, DesktopError
from glide.computer.platform_adapter import Desktop, NotDispatched, dispatched, dispatching, using

# The real functions, read at collection, before the guard replaces them for each test.
REAL_OSASCRIPT = macos.osascript
REAL_ACTIVATE = macos.activate
REAL_OPEN_URL = macos.open_url
REAL_BROWSER_URL = macos.browser_url
REAL_BROWSER_JXA = macos._browser_jxa
REAL_WINDOWS_OPEN_URL = windows.open_url
REAL_GET_JSON = cdp._get_json
REAL_CREATE_CONNECTION = cdp.websocket.create_connection

ADAPTERS = pytest.mark.parametrize("adapter", [macos, windows], ids=["macos", "windows"])


@pytest.fixture(autouse=True)
def inside_dispatch():
    """Every test but the gate's own stands where `control.dispatch` runs the adapter (see the gate tests)."""
    with dispatching():
        yield


# ------------------------------------------------------------------ the shared escape hatch (finding 12)


def test_the_escape_hatch_is_defined_once():
    assert macos.sleep_watching is windows.sleep_watching is platform_adapter.sleep_watching
    assert macos.abort_hint is windows.abort_hint is platform_adapter.abort_hint


@ADAPTERS
def test_check_abort_reads_the_adapters_pointer_when_it_is_called(monkeypatch, adapter):
    monkeypatch.setattr(adapter, "mouse_location", lambda: (0.0, 0.0))
    with pytest.raises(Abort, match="top-left corner"):
        adapter.check_abort()
    monkeypatch.setattr(adapter, "mouse_location", lambda: (900.0, 900.0))
    adapter.check_abort()


@ADAPTERS
def test_sleep_watching_asks_the_check_in_use_before_it_sleeps(monkeypatch, adapter):
    """`abort_on` replaces `check_abort` on the adapter; a wait must reach that replacement."""
    monkeypatch.setattr(platform_adapter.time, "sleep", lambda _: pytest.fail("slept past a stop"))

    def stopped():
        raise Abort("stopped")

    with using(adapter):
        monkeypatch.setattr(adapter, "check_abort", stopped)
        with pytest.raises(Abort, match="stopped"):
            adapter.sleep_watching(30)


# ------------------------------------------------------------------ a fake osascript child


class FakeOsascript:
    """Stands in for `subprocess.Popen` as macos.py calls it. `script` decides what each wait does:
    a list of callables, one per `communicate`, each returning (stdout, stderr) or raising."""

    def __init__(self, steps=None, returncode=0):
        self.steps = list(steps or [])
        self.returncode_when_done = returncode
        self.runs: list[SimpleNamespace] = []

    def __call__(self, argv, **kwargs):
        run = SimpleNamespace(argv=list(argv), killed=False, reaped=False, returncode=None, parent=self)
        run.poll = lambda: run.returncode
        run.kill = lambda: (setattr(run, "killed", True), setattr(run, "returncode", -9))
        run.communicate = lambda timeout=None: self._wait(run, timeout)
        self.runs.append(run)
        return run

    def _wait(self, run, timeout):
        if run.killed:
            run.reaped = True
            return "", ""
        step = self.steps.pop(0) if self.steps else (lambda: ("", ""))
        out = step()  # may raise TimeoutExpired (still running), or anything else
        run.returncode = self.returncode_when_done
        return out


def still_running():
    raise subprocess.TimeoutExpired("osascript", 0.05)


def answer(text):
    return lambda: (text + "\n", "")


@pytest.fixture
def child(monkeypatch):
    fake = FakeOsascript()
    monkeypatch.setattr(macos.subprocess, "Popen", fake)
    return fake


# ------------------------------------------------------------------ osascript deadline and cancel (finding 11)


def test_a_call_that_outlives_its_deadline_is_killed_and_reported_as_unknown(child):
    child.steps = [still_running] * 5
    with pytest.raises(DesktopError, match="outcome is unknown") as caught:
        REAL_OSASCRIPT("on run argv\nend run", timeout=0.0)
    run = child.runs[0]
    assert run.killed and run.reaped
    assert "on run" not in str(caught.value)


def test_a_cancelled_run_kills_the_call_that_is_running(child):
    control = RunControl("task")
    child.steps = [still_running, lambda: control.cancel("stop") or still_running(), still_running]
    with controlled(control), pytest.raises(Abort, match="stop"):
        REAL_OSASCRIPT("on run argv\nend run")
    assert child.runs[0].killed and child.runs[0].reaped


def test_a_call_made_after_the_cancel_never_starts(child):
    control = RunControl("task")
    control.cancel()
    with controlled(control), pytest.raises(Abort):
        REAL_OSASCRIPT("on run argv\nend run")
    assert child.runs == []


def test_a_script_that_fails_raises_without_the_script_in_the_error(child):
    child.steps = [answer("")]
    child.returncode_when_done = 1
    with pytest.raises(subprocess.CalledProcessError) as caught:
        REAL_OSASCRIPT("on run argv\nreturn secret-script\nend run")
    assert "secret-script" not in str(caught.value)


def test_the_jxa_helper_shares_the_deadline_and_the_cancel(child, monkeypatch):
    control = RunControl("task")
    child.steps = [lambda: control.cancel() or still_running()]
    with controlled(control), pytest.raises(Abort):
        REAL_BROWSER_JXA("Google Chrome", "JSON.stringify({})")
    assert child.runs[0].killed


def test_an_unknown_first_activate_is_not_followed_by_the_system_events_retry(monkeypatch):
    calls = []

    def first_call_times_out(script, *args, **kwargs):
        calls.append(script)
        raise DesktopError("outcome is unknown")

    monkeypatch.setattr(macos, "osascript", first_call_times_out)
    with pytest.raises(DesktopError):
        REAL_ACTIVATE("Finder")
    assert calls == [macos.ACTIVATE_SCRIPT]


def test_browser_url_gives_none_for_a_slow_or_failing_browser_but_never_hides_a_stop(monkeypatch):
    def raises(error):
        def call(script, *args, **kwargs):
            raise error

        return call

    for error in (DesktopError("slow"), subprocess.CalledProcessError(1, "osascript")):
        monkeypatch.setattr(macos, "osascript", raises(error))
        assert REAL_BROWSER_URL("Google Chrome") is None
    monkeypatch.setattr(macos, "osascript", raises(Abort("stopped")))
    with pytest.raises(Abort):
        REAL_BROWSER_URL("Google Chrome")


# ------------------------------------------------------------------ hostile text stays data (finding 10)

BENIGN_URL = "https://example.com/page"
# Shapes the URL check lets through (no space, no control character): only argv can keep these inert.
SMUGGLED_URLS = [
    'https://example.com/"&(run(script"id"))&"',
    'https://example.com/\\";tell(application"Finder")to(quit)',
    "https://example.com/';quit#",
    "https://example.com/?q=\\",
]
# Shapes it refuses outright: nothing may be run for these.
REFUSED_URLS = [
    'https://example.com/" & (do shell script "id") & "x y',
    'https://example.com/\nend tell\ndo shell script "id"',
    "https://example.com/\r",
    "javascript:alert(1)",
    "file:///etc/passwd",
    "--renderer-cmd-prefix=calc",
    "-e do shell script",
    "",
    "https://user:pw@example.com/",
]
HOSTILE_APPS = [
    'Chrome" to quit\ntell application "Finder',
    'Chrome" & (do shell script "id") & "',
    "Chrome\\",
    "Chrome\nFinder",
    "-e",
    "-l JavaScript",
    "",
    "x" * 200,
]


@pytest.fixture
def mac_child(monkeypatch, child):
    child.steps = [answer("")] * 8
    monkeypatch.setattr(macos, "osascript", REAL_OSASCRIPT)  # the layer below is faked: `child` is its Popen
    monkeypatch.setattr(macos, "frontmost_app", lambda: "Google Chrome")
    return child


def test_a_url_with_quotes_and_applescript_in_it_is_only_ever_an_argument(mac_child):
    REAL_OPEN_URL("Google Chrome", BENIGN_URL)
    reference = mac_child.runs[0].argv
    for hostile in SMUGGLED_URLS:
        mac_child.runs.clear()
        mac_child.steps = [answer("")] * 8
        assert REAL_OPEN_URL("Google Chrome", hostile) is True
        argv = mac_child.runs[0].argv
        assert argv[:-1] == reference[:-1], "the program is the same fixed one whatever the URL"
        assert argv[-2:] == ["Google Chrome", hostile], "the URL travels as the last argv item, untouched"
        assert hostile not in " ".join(argv[:-1])


@pytest.mark.parametrize("url", REFUSED_URLS)
def test_a_url_that_is_not_a_plain_web_address_runs_nothing(mac_child, url):
    with pytest.raises(DesktopError, match="Invalid URL"):
        REAL_OPEN_URL("Google Chrome", url)
    assert mac_child.runs == []


@pytest.mark.parametrize("app", HOSTILE_APPS)
def test_an_application_name_cannot_carry_applescript_or_an_option(mac_child, app):
    for call in (
        lambda: REAL_ACTIVATE(app),
        lambda: REAL_OPEN_URL(app, BENIGN_URL),
    ):
        with pytest.raises(DesktopError, match="Invalid application name"):
            call()
    assert REAL_BROWSER_URL(app) is None
    assert mac_child.runs == []


def test_every_real_application_name_shape_is_accepted(mac_child, monkeypatch):
    for app in (
        "Google Chrome",
        "Brave Browser",
        "Microsoft Edge",
        "Safari",
        "Arc",
        "Visual Studio Code",
        "1Password 7",
        "Caf\u00e9",
    ):
        mac_child.steps = [answer("")] * 8
        monkeypatch.setattr(macos, "frontmost_app", lambda app=app: app)
        assert REAL_ACTIVATE(app) is True
        assert mac_child.runs[-1].argv[-1] == app


def test_activate_passes_the_app_to_both_of_its_fixed_programs(mac_child, monkeypatch):
    monkeypatch.setattr(macos, "frontmost_app", lambda: "Something Else")
    monkeypatch.setattr(macos.time, "sleep", lambda _: None)
    assert REAL_ACTIVATE("Google Chrome", timeout=0.0) is False  # no time to wait, so it goes straight to the retry
    first, second = (run.argv for run in mac_child.runs)
    assert "-e" in first and first[-1] == "Google Chrome" and second[-1] == "Google Chrome"
    assert "System Events" in " ".join(second) and "Google Chrome" not in " ".join(second[:-1])


def test_a_call_that_names_an_option_as_an_argument_is_refused_by_osascript_itself(child):
    with pytest.raises(DesktopError, match="Invalid AppleScript argument"):
        REAL_OSASCRIPT("on run argv\nend run", "-e")
    assert child.runs == []


# ------------------------------------------------------------------ Windows: the same URL rule


@pytest.fixture
def windows_launch(monkeypatch):
    launched = []
    monkeypatch.setattr(windows.subprocess, "Popen", lambda argv, **kw: launched.append(("popen", argv)))
    monkeypatch.setattr(windows.webbrowser, "open", lambda url: launched.append(("webbrowser", url)))
    monkeypatch.setattr(windows.shutil, "which", lambda exe: f"C:/{exe}.exe")
    monkeypatch.setattr(windows, "activate", lambda app, timeout=3.0: True)
    return launched


@pytest.mark.parametrize("url", REFUSED_URLS)
def test_windows_refuses_what_macos_refuses(windows_launch, url):
    with pytest.raises(DesktopError, match="Invalid URL"):
        REAL_WINDOWS_OPEN_URL("Google Chrome", url)
    assert windows_launch == []


def test_windows_launches_a_plain_web_address(windows_launch):
    assert REAL_WINDOWS_OPEN_URL("Google Chrome", BENIGN_URL) is True
    assert windows_launch == [("popen", ["chrome", BENIGN_URL])]


# ------------------------------------------------------------------ CDP ignores environment proxies on loopback (finding 4)


class Listener:
    """A loopback TCP listener that counts the connections it is dealt and closes each at once."""

    def __init__(self):
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen()
        self.sock.settimeout(0.05)
        self.port = self.sock.getsockname()[1]
        self.accepted = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def _serve(self):
        while not self._stop.is_set():
            try:
                client, _ = self.sock.accept()
            except TimeoutError:
                continue
            self.accepted += 1
            client.close()

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.sock.close()


class Answers(BaseHTTPRequestHandler):
    def do_GET(self):
        body = json.dumps({"Browser": "fake"}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def proxy_in_the_environment(monkeypatch):
    """What the conftest guard switched off, switched back on: a proxy named in the environment, no
    `no_proxy`, and urllib reading the environment again. The proxy is a loopback listener that counts hits."""
    with Listener() as proxy:
        monkeypatch.setenv("http_proxy", f"http://127.0.0.1:{proxy.port}")
        monkeypatch.setenv("HTTP_PROXY", f"http://127.0.0.1:{proxy.port}")
        for name in ("no_proxy", "NO_PROXY"):
            monkeypatch.delenv(name, raising=False)
        monkeypatch.setattr(urllib.request, "getproxies", urllib.request.getproxies_environment)
        monkeypatch.setattr(urllib.request, "_opener", None)  # the shared opener is built once, from the environment
        yield proxy


def test_the_cdp_http_endpoint_is_asked_directly_whatever_the_environment_says(proxy_in_the_environment):
    server = ThreadingHTTPServer(("127.0.0.1", 0), Answers)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        reply = REAL_GET_JSON(f"http://127.0.0.1:{server.server_address[1]}/json/version", timeout=2.0)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert reply == {"Browser": "fake"}
    assert proxy_in_the_environment.accepted == 0


def test_the_cdp_websocket_is_dialled_directly_whatever_the_environment_says(proxy_in_the_environment, monkeypatch):
    monkeypatch.setattr(cdp.websocket, "create_connection", REAL_CREATE_CONNECTION)
    with (
        Listener() as target,
        pytest.raises((cdp.websocket.WebSocketException, OSError)),
    ):  # the listener hangs up: only where it dialled matters
        cdp.Session(f"ws://127.0.0.1:{target.port}/devtools/page/1", timeout=1.0)
    assert proxy_in_the_environment.accepted == 0
    assert target.accepted == 1


def test_an_address_that_is_not_loopback_keeps_the_environment_proxy_rules(monkeypatch):
    """Only this machine's own addresses are exempt; the helper that decides is exact."""
    assert all(cdp.is_loopback(host) for host in ("127.0.0.1", "::1", "localhost", "127.1.2.3"))
    assert not any(cdp.is_loopback(host) for host in ("example.com", "10.0.0.5", "0.0.0.0", "", None, "127.0.0.1.example.com"))
    seen = {}
    monkeypatch.setattr(cdp.websocket, "create_connection", lambda url, **kw: seen.update(url=url, **kw))
    cdp.Session("ws://example.com:9222/devtools/page/1")
    assert "http_proxy_host" not in seen and "http_no_proxy" not in seen


# ------------------------------------------------------------------ input only through dispatch (finding 7b)

# What the adapters expose that neither sends input nor acts on another app. Everything else on the Desktop
# surface must be gated, so a new primitive has to be classified here before the suite passes.
OBSERVATION = {
    "check_abort", "abort_hint", "sleep_watching", "accessibility_trusted", "screen_capture_trusted",
    "request_permissions", "frontmost_app_and_pid", "browser_url", "open_path", "frontmost_window_bounds",
    "screenshot", "display_scale", "recognize_text", "focused_field", "actionable_elements", "ax_value",
    "execution_tabs", "execution_scrolls", "execution_labels",
}  # fmt: skip
SURFACE = {name for name, value in vars(Desktop).items() if callable(value) and not name.startswith("_")}
GATED = sorted(SURFACE - OBSERVATION)
CALLS = {
    "click_at": ((1.0, 2.0),),
    "press": ("a",),
    "type_text": ("x",),
    "clear_field": (),
    "scroll": (1,),
    "activate": ("Finder",),
    "open_url": ("Google Chrome", "https://example.com"),
    "ax_press": (None,),
    "ax_focus": (None,),
    "ax_set_value": (None, "x"),
    "execution_tab": ("Google Chrome", "navigate", "", "https://example.com"),
    "execution_shortcut": ("a", ()),
    "execution_scroll": (None, "up"),
}


# Read at collection: by the time a test runs, the guard has put refusals in place of several of these.
PRIMITIVES = {(adapter, name): getattr(adapter, name) for adapter in (macos, windows) for name in CALLS}


def outside_any_dispatch(call):
    """Run `call` in a fresh context, where nothing has marked a dispatch."""
    return contextvars.Context().run(call)


def test_every_input_primitive_on_the_desktop_surface_is_classified():
    assert set(GATED) == set(CALLS), "a new Desktop method is either an observation or needs a gate and a call here"
    assert OBSERVATION <= SURFACE


@ADAPTERS
@pytest.mark.parametrize("name", GATED)
def test_an_input_primitive_refuses_to_run_outside_a_dispatch(adapter, name):
    primitive = PRIMITIVES[adapter, name]
    assert hasattr(primitive, "__wrapped__"), f"{adapter.__name__}.{name} is not gated"
    with pytest.raises(NotDispatched, match=name):
        outside_any_dispatch(lambda: primitive(*CALLS[name]))


def test_the_gate_error_is_not_one_the_runner_absorbs_as_a_failed_action():
    assert not issubclass(NotDispatched, (DesktopError, RuntimeError, Abort))


def test_a_gated_primitive_runs_inside_the_block_and_the_mark_ends_with_it():
    @dispatched
    def probe(value):
        return value

    def run():
        with pytest.raises(NotDispatched):
            probe(1)
        with dispatching():
            assert probe(2) == 2
        with pytest.raises(NotDispatched):
            probe(3)
        with pytest.raises(KeyError), dispatching():
            raise KeyError
        with pytest.raises(NotDispatched):
            probe(4)

    outside_any_dispatch(run)


def test_the_primitives_that_call_each_other_pass_inside_one_block(monkeypatch):
    pressed = []
    monkeypatch.setattr(macos, "_down_then_up", lambda event: pressed.append(event(True)))
    monkeypatch.setattr(macos, "Quartz", SimpleNamespace(
        CGEventCreateKeyboardEvent=lambda _, code, down: code,
        CGEventSetFlags=lambda event, flags: None,
        kCGEventFlagMaskCommand=1,
    ))  # fmt: skip

    def run():
        with dispatching():
            macos.clear_field()  # presses command-a, then delete

    outside_any_dispatch(run)
    assert pressed == [0, 51]


@pytest.mark.xfail(
    strict=True,
    reason="control.dispatch must enter platform_adapter.dispatching(): the patch is in the report. "
    "Once it is applied this passes, strict xfail turns that into a failure, and the marker goes.",
)
def test_control_dispatch_marks_the_dispatch_for_the_adapters():
    @dispatched
    def probe():
        return "reached"

    assert outside_any_dispatch(lambda: dispatch(probe)) == "reached"

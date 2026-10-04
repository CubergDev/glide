"""The seams where the platform adapters meet the rest of the machine: the shared escape hatch, the
AppleScript argv boundary, the osascript deadline and cancel, the dispatch gate, and loopback-only CDP.

Everything here runs offline against fakes. `tests/conftest.py` refuses the real functions in every test
(`macos.osascript`, `windows.open_url`, `cdp._get_json`, `websocket.create_connection`, `subprocess.Popen`),
so the real ones are captured at import time, before any guard runs, and each test fakes only the layer
below the one it exercises.
"""

import subprocess
from types import SimpleNamespace

import pytest

from glide.computer import macos, platform_adapter, windows
from glide.computer.control import RunControl, controlled
from glide.computer.models import Abort, DesktopError
from glide.computer.platform_adapter import using

# The real functions, read at collection, before the guard replaces them for each test.
REAL_OSASCRIPT = macos.osascript
REAL_ACTIVATE = macos.activate
REAL_OPEN_URL = macos.open_url
REAL_BROWSER_URL = macos.browser_url
REAL_BROWSER_JXA = macos._browser_jxa
REAL_WINDOWS_OPEN_URL = windows.open_url

ADAPTERS = pytest.mark.parametrize("adapter", [macos, windows], ids=["macos", "windows"])

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

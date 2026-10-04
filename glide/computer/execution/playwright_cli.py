"""Fixed Playwright CLI bridge for an explicitly opened browser session.

The model still chooses validated primitives. Only this package's fixed driver
code can reach run-code; task text is JSON data, never a program or shell command.

The one process this starts is the user's own `playwright-cli`, through `spawn.start` (the package's single seam).
It attaches to a session the user opened; it never opens a browser.
"""

import json
import os
import shutil
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .. import browser_settings
from ..browser.cdp import CDPError
from ..control import checkpoint, current_control
from ..diagnostics import event, redact
from ..models import Abort, DesktopError
from . import spawn
from .contracts import UnsupportedCapability
from .dom import DOCUMENT_CHANGED, BrowserBackend

METHODS = {
    "Target.getTargets",
    "Target.createTarget",
    "Target.attachToTarget",
    "Target.activateTarget",
    "Target.closeTarget",
    "Runtime.evaluate",
    "Page.navigate",
    "Page.getFrameTree",
    "Input.dispatchKeyEvent",
    "Input.dispatchMouseEvent",
    "Input.insertText",
    "Glide.releasePair",
    "Glide.selectAll",
}
MAX_OUTPUT = 1024 * 1024
COMMAND_SECONDS = 5.0
DIAGNOSTIC_OUTPUT = 4096
DRIVER = Path(__file__).with_name("playwright_driver.js")
# The only environment the CLI gets. Model, speech and other credentials are never forwarded to a browser helper.
CHILD_ENVIRONMENT = (
    "PATH",
    "HOME",
    "TMPDIR",
    "SYSTEMROOT",
    "LOCALAPPDATA",
    "APPDATA",
    "USERPROFILE",
    "XDG_CONFIG_HOME",
    "XDG_CACHE_HOME",
    "XDG_RUNTIME_DIR",
    "PLAYWRIGHT_BROWSERS_PATH",
)


@contextmanager
def operation_gate():
    """A queued daemon request must recheck live cancellation before each mutation.

    The CLI daemon outlives its client. Closing this per-command gate makes a late
    request fail closed instead of executing after the Python process stopped.
    """
    control = current_control()
    deadline = time.monotonic() + COMMAND_SECONDS
    path = "/" + uuid.uuid4().hex
    closed = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            while control and not control.ready.is_set() and not control.cancelled.is_set() and not closed.is_set():
                if time.monotonic() >= deadline:
                    break
                closed.wait(0.02)
            allowed = (
                self.path == path
                and not closed.is_set()
                and time.monotonic() < deadline
                and (control is None or (not control.cancelled.is_set() and control.ready.is_set()))
            )
            body = json.dumps({"allowed": allowed}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}{path}"
    finally:
        closed.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=0.5)


def executable(configured=""):
    """The user's `playwright-cli`: the configured path, else the one on PATH. Nothing is searched for in the working
    directory, and nothing is installed or fetched."""
    found = shutil.which(configured or "playwright-cli")
    if not found:
        raise DesktopError(
            "Playwright CLI is not installed: install @playwright/cli, or set its path "
            "(glide.toml [browser.playwright] cli, or GLIDE_PLAYWRIGHT_CLI)"
        )
    # Check a recognizable npm entry point. Do not accidentally invoke the test CLI.
    manifest = Path(found).resolve().parent / "package.json"
    if manifest.is_file():
        try:
            package = json.loads(manifest.read_text())
        except (OSError, ValueError) as error:
            raise DesktopError("Cannot read Playwright CLI package metadata") from error
        if package.get("name") != "@playwright/cli":
            raise DesktopError("Select Microsoft's @playwright/cli executable (playwright-cli), not the playwright test CLI")
    return found


def keep_diagnostic_tail(output, evidence):
    """For a client that did not finish cleanly: keep the redacted tail of what it wrote, bounded.

    Redaction comes before truncation. When the output was larger than the limit, the window can start halfway
    through a credential or its label, so only complete lines are kept (and `output_omitted` says when none is).
    """
    try:
        output.seek(0, os.SEEK_END)
        size = output.tell()
        offset = max(0, size - MAX_OUTPUT)
        output.seek(offset)
        raw = output.read(min(size, MAX_OUTPUT))
        if offset:
            raw = raw.partition(b"\n")[2]
            if not raw:
                evidence["output_omitted"] = "partial_line"
        evidence["output"] = redact(raw.decode("utf-8", errors="replace"))[-DIAGNOSTIC_OUTPUT:]
    except (OSError, ValueError):
        pass  # Diagnostic reads must preserve the original command failure.


class CliSession:
    def __init__(self, session, cli=""):
        browser_settings.validate_provider("playwright", "", "", session)
        self.session, self.cli = session, cli

    def call(self, method, params=None, *, session_id=None):
        started = time.perf_counter()
        details = {"provider": "playwright-cli", "method": method, "endpoint": "playwright:" + self.session}
        evidence = {}
        event("browser_request_started", **details)
        try:
            result = self._call(method, params, session_id=session_id, evidence=evidence)
        except BaseException as error:
            event("browser_request_failed", **details, elapsed_s=time.perf_counter() - started, exception=error, **evidence)
            raise
        event(
            "browser_request_completed", **details, elapsed_s=time.perf_counter() - started, exit_code=evidence.get("exit_code")
        )
        return result

    def _call(self, method, params=None, *, session_id=None, evidence):
        if method == "Target.detachFromTarget":
            return {}  # Leave the user's daemon and browser alive between tasks.
        if method not in METHODS:
            raise CDPError("Unsupported Playwright bridge method")
        checkpoint()
        program = executable(self.cli)
        env = {k: v for k, v in os.environ.items() if k in CHILD_ENVIRONMENT}
        env["NO_UPDATE_NOTIFIER"] = "1"
        with operation_gate() as gate, tempfile.TemporaryDirectory(prefix="glide-playwright-") as directory:
            request = {
                "method": method,
                "params": params or {},
                "target": session_id or "",
                "deadline": (time.time() + COMMAND_SECONDS) * 1000,
                "gate": gate,
            }
            path = Path(directory) / "command.js"
            path.write_text(DRIVER.read_text().replace("__GLIDE_REQUEST__", json.dumps(request, ensure_ascii=True), 1))
            path.chmod(0o600)
            raw = self._run(
                [program, "-s=" + self.session, "run-code", "--filename=" + str(path), "--raw", "--json"], env, evidence
            )
        return self._result(raw, method, evidence["exit_code"])

    def _run(self, args, env, evidence):
        """Run the CLI to completion and return its output; the client is killed only after it has settled or timed out."""
        with tempfile.TemporaryFile() as output:
            process = spawn.start(args, stdin=spawn.DEVNULL, stdout=output, stderr=output, env=env)
            deadline = time.monotonic() + COMMAND_SECONDS + 1
            interrupted = False
            try:
                while process.poll() is None:
                    # A dispatched write must settle before Stop reconciliation. Killing
                    # the CLI client alone cannot cancel its daemon's operation.
                    try:
                        checkpoint(wait=False)
                    except (Abort, KeyboardInterrupt):
                        interrupted = True
                    if output.tell() > MAX_OUTPUT or time.monotonic() >= deadline:
                        raise CDPError("Playwright request timed out; completion unknown, no automatic replay")
                    time.sleep(0.02)
                if interrupted:
                    checkpoint(wait=False)
                    raise KeyboardInterrupt
                size = output.tell()
                if size > MAX_OUTPUT:
                    raise CDPError("Playwright response exceeded its limit")
                output.seek(0)
                raw = output.read(MAX_OUTPUT).decode("utf-8", errors="replace")
                # Redacted before truncation, so a credential's prefix cannot vanish at the tail and leave its suffix.
                evidence["output"] = redact(raw)[-DIAGNOSTIC_OUTPUT:]
                return raw
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=0.5)
                    except spawn.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=0.5)
                evidence["exit_code"] = process.returncode
                if "output" not in evidence:
                    keep_diagnostic_tail(output, evidence)

    def _result(self, raw, method, exit_code):
        try:
            value = json.loads(raw)
        except ValueError as error:
            raise CDPError("Invalid Playwright CLI response; check the configured session") from error
        if exit_code or not isinstance(value, dict) or value.get("isError"):
            if any(marker in raw for marker in DOCUMENT_CHANGED):
                # The page navigated under a read: a page problem, which the read retries, not a lost session.
                raise CDPError(f"{method}: Execution context was destroyed")
            raise CDPError("Playwright operation failed; open the configured browser session or check its activity")
        try:
            result = json.loads(value["result"])
        except (KeyError, TypeError, ValueError) as error:
            raise CDPError("Invalid Playwright command result") from error
        if not isinstance(result, dict):
            raise CDPError("Invalid Playwright command result")
        checkpoint(wait=False)
        return result

    def close(self):
        pass


class PlaywrightBackend(BrowserBackend):
    transport = "playwright-cli"
    passive_inspection = False  # the CLI's ensureTab() can recreate a page before our script runs, so it may act

    def __init__(self, session=browser_settings.DEFAULT_SESSION, target="", *, allow_actions=False, cli=""):
        browser_settings.validate_provider("playwright", "", target, session)
        self._initialize(target)
        self.origin = "playwright:" + session
        self.session_name = session
        self.cli = cli
        self.allow_actions = allow_actions

    def check(self):
        """Usable only when allowed to act and the CLI is installed. Nothing is started to find out."""
        if not self.allow_actions:
            raise DesktopError(
                "Playwright CLI is not enabled: it needs Allow computer actions (its inspection can recreate a tab); "
                "use CDP for passive inspection"
            )
        executable(self.cli)

    def connection_lost(self, error):
        # Every failure of the bridge is a failure of the CLI session; a document change is told apart in `_result`.
        return isinstance(error, CDPError)

    def inspect(self, *, controls=True):
        # CLI ensureTab() recreates an empty/crashed selected page before invoking
        # run-code, outside our driver. It cannot guarantee passive observation.
        if not self.allow_actions:
            raise UnsupportedCapability(
                ["Playwright CLI inspection requires Allow computer actions; use CDP for passive inspection"]
            )
        return super().inspect(controls=controls)

    def execute(self, action, observed):
        if not self.allow_actions:
            raise UnsupportedCapability(["Playwright CLI actions are disabled"])
        return super().execute(action, observed)

    def connect(self):
        if not self.browser:
            self.browser = CliSession(self.session_name, self.cli)

    def select_all(self):
        # Selection works in any engine and does not depend on the host OS shortcut.
        self.page.call("Glide.selectAll")

    def paired(self, method, down, up):
        checkpoint()
        # One daemon operation owns both edges. A killed/timed-out CLI client must
        # not leave a key held while a second command waits to release it.
        self.page.call("Glide.releasePair", {"method": method, "down": down, "up": up})

"""The wizard's local page. Standard library only, bound to 127.0.0.1 on a random port.

Every request needs the run's secret: the first page load spends the one-time URL token and is handed a session token
that every later request must send in a header. Host and Origin must name this server, mutations are POST with a JSON
body under a size limit, and the page is one document under a strict CSP. Pasted keys live in `self.keys` only: never
echoed, logged or written. The server stops after it has been idle for a while.
"""

from __future__ import annotations

import hmac
import json
import os
import secrets
import subprocess
import threading
import time
from collections.abc import Callable, Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from glide.providers import doctor
from glide.providers.config import PRESETS, GlideConfig

from . import model
from .page import render_page

MAX_BODY = 64 * 1024
IDLE_S = 15 * 60
OUTPUT_CAP = 20_000
HEADER = "X-Glide-Session"


def run_process(argv: list[str], env: Mapping[str, str], *, wait: bool) -> str:
    """Start Glide for the person who clicked Start. Waits and returns capped output for `ask`, else detaches."""
    if not wait:
        subprocess.Popen(argv, env=dict(env), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return "started"
    done = subprocess.run(argv, env=dict(env), capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL)
    return (done.stdout + done.stderr)[:OUTPUT_CAP]


def _live_probe(cfg: GlideConfig, role: str, info: object) -> doctor.Row:
    return doctor._probe(cfg, role, info, 20.0, time.monotonic)


class SetupServer:
    def __init__(
        self,
        config_path: Path,
        *,
        env: Mapping[str, str] | None = None,
        idle_s: float = IDLE_S,
        runner: Callable[..., str] = run_process,
        prober: Callable[[GlideConfig, str, object], doctor.Row] | None = None,
    ) -> None:
        self.config_path = config_path
        self.env = os.environ if env is None else env
        self.idle_s = idle_s
        self.runner = runner
        self.prober = prober or (lambda c, r, i: _live_probe(c, r, i))
        self.keys: dict[str, str] = {}  # env var name -> pasted value, in memory only
        self.url_token = secrets.token_urlsafe(24)
        self.session = secrets.token_urlsafe(24)
        self.spent = False
        self.last = time.monotonic()
        self.test_lock = threading.Lock()
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(self))
        self.port = self.httpd.server_address[1]
        self.nonce = secrets.token_urlsafe(16)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/?t={self.url_token}"

    def serve(self) -> None:
        self.httpd.timeout = 1.0
        while time.monotonic() - self.last < self.idle_s and not getattr(self, "stopped", False):
            self.httpd.handle_request()
        self.httpd.server_close()

    def stop(self) -> None:
        self.stopped = True

    def child_env(self) -> dict[str, str]:
        return {**self.env, **self.keys}

    # Hooks a subclass (the control panel) overrides: the page, what a GET may read, and the sweep over every response.
    def render(self) -> str:
        return render_page(self.session, self.nonce)

    def read(self, name: str, query: dict) -> dict:
        raise model.SetupError("unknown call")

    def scrub_response(self, text: str) -> str:
        for value in self.keys.values():
            text = text.replace(value, "[key]").replace(json.dumps(value)[1:-1], "[key]")
        return text

    # -- API ------------------------------------------------------------------------------------

    def state(self) -> dict:
        rows = model.provider_rows()
        for row in rows:
            row["status"] = model.key_status(row["name"], self.keys, self.env)
            row["export"] = model.export_line(row["name"])
        roles = {
            r: {"candidates": model.candidates(r), "defaults": {c: model.default_slot(r, c) for c in model.candidates(r)}}
            for r in model.ROLES
        }
        return {
            "providers": rows,
            "roles": roles,
            "features": model.DEFAULT_FEATURES,
            "path": str(self.config_path),
            "exists": self.config_path.exists(),
            "modes": model.MODES,
        }

    def api(self, name: str, body: dict) -> dict:
        if name == "preset":
            return {"roles": model.preset(str(body.get("provider")))}
        if name == "render":
            return {"toml": model.render_toml(body)}
        if name == "key":
            return self._key(body)
        if name == "test":
            return self._test(body)
        if name == "write":
            text = model.render_toml(body)
            backup = model.write_config(self.config_path, text)
            return {"written": str(self.config_path), "backup": str(backup) if backup else ""}
        if name == "launch":
            return self._launch(body)
        raise model.SetupError("unknown call")

    def _key(self, body: dict) -> dict:
        provider, value = body.get("provider"), body.get("value", "")
        if provider not in PRESETS or not PRESETS[provider].api_key_env:
            raise model.SetupError("that provider takes no key")
        var = PRESETS[provider].api_key_env
        if value == "":
            self.keys.pop(var, None)
        elif isinstance(value, str) and model.KEY_TEXT.fullmatch(value):
            self.keys[var] = value
        else:
            raise model.SetupError("the key must be 4 to 512 printable characters with no spaces")
        return {"status": model.key_status(provider, self.keys, self.env)}  # presence only

    def _scrub(self, text: str) -> str:
        for value in self.keys.values():
            text = text.replace(value, "[key]")
        return text

    def _test(self, body: dict) -> dict:
        if body.get("confirm") is not True:
            raise model.SetupError("a test runs only after an explicit click")
        provider, role = body.get("provider"), body.get("role")
        state = model.clean_state(body.get("state"))
        slots = [s for s in state["roles"].get(role, []) if s["provider"] == provider]
        if not slots:
            raise model.SetupError("that provider is not in this role's chain")
        if not self.test_lock.acquire(blocking=False):
            raise model.SetupError("a test is already running")
        try:
            config = GlideConfig.from_toml(model.render_toml(state), env=self.child_env(), source="glide setup test")
            try:
                info = next(i for i in config.slots(role) if i.provider == provider)
                if info.state != "ready":
                    return {
                        "status": f"skipped({info.short})",
                        "detail": self._scrub(config.scrub(info.reason)),
                        "latency_s": None,
                    }
                row = self.prober(config, role, info)
                return {"status": row.status, "detail": self._scrub(config.scrub(row.detail)), "latency_s": row.latency_s}
            finally:
                config.close()
        finally:
            self.test_lock.release()

    def _launch(self, body: dict) -> dict:
        mode = body.get("mode")
        act = bool(model.clean_state(body.get("state"))["features"]["computer"]) if body.get("state") else False
        if act and body.get("confirm_act") is not True:
            raise model.SetupError("computer control is on: confirm that Glide may click and type")
        if not self.config_path.is_file():
            raise model.SetupError("write glide.toml first")
        text = str(body.get("text") or "")
        argv = model.launch_argv(mode, self.config_path, act=act, text=text)
        equivalent = model.equivalent_command(mode, self.config_path, act=act)
        if mode == "chat":
            return {"command": equivalent, "output": "Chat needs a terminal: run the command above there."}
        out = self.runner(argv, self.child_env(), wait=mode == "ask")
        return {"command": equivalent, "output": self._scrub(out)}


def _make_handler(app: SetupServer):
    class Handler(BaseHTTPRequestHandler):
        server_version = "GlideSetup"
        sys_version = ""

        def log_message(self, *args):  # no access log: URLs carry a token
            pass

        def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, data: dict) -> None:
            self._send(code, app.scrub_response(json.dumps(data)).encode(), "application/json")

        def _local(self) -> bool:
            own = f"127.0.0.1:{app.port}"
            if self.headers.get("Host") != own:
                return False
            origin = self.headers.get("Origin")
            return origin is None or origin == f"http://{own}"

        def _authed(self) -> bool:
            return hmac.compare_digest(self.headers.get(HEADER, ""), app.session)

        def do_GET(self):
            app.last = time.monotonic()
            if not self._local():
                return self._json(403, {"error": "not this server"})
            parts = urlsplit(self.path)
            if parts.path == "/api/state":
                return self._json(200, app.state()) if self._authed() else self._json(403, {"error": "no session"})
            if parts.path.startswith("/api/read/"):
                if not self._authed():
                    return self._json(403, {"error": "no session"})
                try:
                    return self._json(200, app.read(parts.path.removeprefix("/api/read/"), parse_qs(parts.query)))
                except model.SetupError as e:
                    return self._json(400, {"error": str(e)})
                except Exception as e:
                    return self._json(500, {"error": type(e).__name__})
            token = (parse_qs(parts.query).get("t") or [""])[0]
            if parts.path != "/" or app.spent or not hmac.compare_digest(token, app.url_token):
                return self._json(403, {"error": "this link was already used; run the command again"})
            app.spent = True
            csp = (
                f"default-src 'none'; script-src 'nonce-{app.nonce}'; style-src 'nonce-{app.nonce}'; "
                "connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
            )
            page = app.render().encode()
            self._send(200, page, "text/html; charset=utf-8", {"Content-Security-Policy": csp})

        def do_POST(self):
            app.last = time.monotonic()
            if not self._local() or not self._authed():
                return self._json(403, {"error": "not allowed"})
            name = urlsplit(self.path).path.removeprefix("/api/")
            if not self.path.startswith("/api/") or self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                return self._json(400, {"error": "JSON only"})
            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                return self._json(411, {"error": "length required"})
            if not 0 <= length <= MAX_BODY:
                return self._json(413, {"error": "too large"})
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(body, dict):
                    raise ValueError
            except ValueError:
                return self._json(400, {"error": "bad JSON"})
            try:
                self._json(200, app.api(name, body))
            except model.SetupError as e:
                self._json(400, {"error": str(e)})
            except Exception as e:  # never echo more than the type: a message could carry request text
                self._json(500, {"error": type(e).__name__})

        def do_PUT(self):
            self._json(405, {"error": "POST only"})

        do_DELETE = do_PATCH = do_PUT

    return Handler

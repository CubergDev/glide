"""A small set: the wizard's safety behaviours and one happy path each for the page, the file and the text mode."""

from __future__ import annotations

import http.client
import json
import threading
from pathlib import Path

import pytest

from glide.providers.config import GlideConfig
from glide.providers.doctor import Row
from glide.setup import model, text
from glide.setup.server import SetupServer

KEY = "sk-test-SECRET-123456"


@pytest.fixture
def app(tmp_path):
    ran = []
    srv = SetupServer(
        tmp_path / "glide.toml",
        env={"PATH": "x"},
        runner=lambda argv, env, wait: ran.append((argv, env)) or "ok",
        prober=lambda cfg, role, info: Row(role, info.name, "ok", f"answered with {KEY}", 0.1),
    )
    srv.ran = ran
    t = threading.Thread(target=srv.httpd.serve_forever, daemon=True)
    t.start()
    yield srv
    srv.httpd.shutdown()
    srv.httpd.server_close()


def req(app, method, path, body=None, headers=None, host=None):
    c = http.client.HTTPConnection("127.0.0.1", app.port)
    h = {"Host": host or f"127.0.0.1:{app.port}", **(headers or {})}
    data = None
    if body is not None:
        data = json.dumps(body)
        h.setdefault("Content-Type", "application/json")
    c.request(method, path, data, h)
    r = c.getresponse()
    return r.status, r.read().decode(), r


def auth(app):
    return {"X-Glide-Session": app.session}


def open_page(app):
    return req(app, "GET", f"/?t={app.url_token}")


def test_token_is_one_time_and_page_is_locked_down(app):
    assert req(app, "GET", "/")[0] == 403
    code, body, r = open_page(app)
    assert code == 200 and "Glide setup" in body and app.session in body
    csp = r.getheader("Content-Security-Policy")
    assert "default-src 'none'" in csp and "'unsafe-inline'" not in csp and "frame-ancestors 'none'" in csp
    assert open_page(app)[0] == 403  # spent


def test_requests_need_session_host_and_origin(app):
    assert req(app, "GET", "/api/state")[0] == 403
    assert req(app, "GET", "/api/state", headers=auth(app))[0] == 200
    assert req(app, "GET", "/api/state", headers=auth(app), host="evil.example")[0] == 403
    assert req(app, "POST", "/api/render", {}, {**auth(app), "Origin": "http://evil.example"})[0] == 403
    assert req(app, "POST", "/api/render", {})[0] == 403
    assert req(app, "PUT", "/api/render", {}, auth(app))[0] == 405


def test_body_limits_and_content_type(app):
    big = {"x": "a" * (70 * 1024)}
    assert req(app, "POST", "/api/render", big, auth(app))[0] == 413
    assert req(app, "POST", "/api/render", None, {**auth(app), "Content-Type": "text/plain", "Content-Length": "2"})[0] == 400


def test_pasted_key_is_presence_only_and_never_written(app):
    code, body, _ = req(app, "POST", "/api/key", {"provider": "openai", "value": KEY}, auth(app))
    assert json.loads(body) == {"status": "pasted"} and KEY not in body
    assert KEY not in req(app, "GET", "/api/state", headers=auth(app))[1]
    state = {"features": {}, "roles": model.preset("openai")}
    code, body, _ = req(app, "POST", "/api/write", state, auth(app))
    assert code == 200 and KEY not in app.config_path.read_text()
    assert req(app, "POST", "/api/key", {"provider": "openai", "value": "has space"}, auth(app))[0] == 400


def test_write_loads_backs_up_and_defaults_are_safe(app):
    state = {"features": {}, "roles": model.preset("openai")}
    text_ = model.render_toml(state)
    assert "[memory]" not in text_ and "[webhooks]" not in text_
    GlideConfig.from_toml(text_, env={}, source="t").close()
    app.config_path.write_text("old = 1\n")
    _, body, _ = req(app, "POST", "/api/write", state, auth(app))
    backup = Path(json.loads(body)["backup"])
    assert backup.read_text() == "old = 1\n" and app.config_path.read_text() == text_
    on = model.render_toml({"features": {"memory": True, "webhooks": True}, "roles": {}})
    assert "enabled = true" in on and 'config = "webhooks.json"' in on


def test_bad_state_is_refused(app):
    assert (
        req(app, "POST", "/api/render", {"roles": {"llm.fast": [{"provider": "elevenlabs", "model": "m"}]}}, auth(app))[0] == 400
    )
    assert (
        req(app, "POST", "/api/render", {"roles": {"llm.fast": [{"provider": "openai", "model": 'x"\nevil'}]}}, auth(app))[0]
        == 400
    )


def test_test_needs_explicit_click_and_scrubs_key(app):
    state = {"features": {}, "roles": model.preset("openai")}
    body = {"provider": "openai", "role": "llm.fast", "state": state}
    assert req(app, "POST", "/api/test", body, auth(app))[0] == 400
    req(app, "POST", "/api/key", {"provider": "openai", "value": KEY}, auth(app))
    code, out, _ = req(app, "POST", "/api/test", {**body, "confirm": True}, auth(app))
    assert code == 200 and json.loads(out)["status"] == "ok" and KEY not in out


def test_launch_needs_file_and_act_confirmation(app):
    state = {"features": {"computer": True}, "roles": {}}
    assert req(app, "POST", "/api/launch", {"mode": "ask", "text": "hi", "state": state}, auth(app))[0] == 400  # no file yet
    app.config_path.write_text("")
    assert req(app, "POST", "/api/launch", {"mode": "ask", "text": "hi", "state": state}, auth(app))[0] == 400  # no act confirm
    req(app, "POST", "/api/key", {"provider": "openai", "value": KEY}, auth(app))
    code, out, _ = req(app, "POST", "/api/launch", {"mode": "ask", "text": "hi", "state": state, "confirm_act": True}, auth(app))
    argv, env = app.ran[0]
    assert code == 200 and "--act" in argv and env["OPENAI_API_KEY"] == KEY and KEY not in out
    assert "glide --config" in json.loads(out)["command"]


def test_text_mode_happy_path(tmp_path):
    answers = iter(["n", "n", "n", "openai", "y", "", ""])
    lines = []
    secrets_ = iter([KEY])
    code = text.run_text(
        tmp_path / "glide.toml",
        ask=lambda p: next(answers),
        secret=lambda p: next(secrets_),
        out=lines.append,
        runner=lambda *a, **k: 0,
        env={},
    )
    written = (tmp_path / "glide.toml").read_text()
    assert code == 0 and "openai" in written and KEY not in written and KEY not in "\n".join(lines)
    assert "OPENAI_API_KEY" in "\n".join(lines)

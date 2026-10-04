"""A small set: the panel's safety behaviours and one happy path per feature."""

from __future__ import annotations

import http.client
import json
import threading
import time
import tomllib
from types import SimpleNamespace

import pytest

from glide.assistant.core import Reply
from glide.panel.server import PanelServer
from glide.providers.config import load_config
from glide.providers.doctor import Row
from tests.files_world import build_world

KEY = "sk-test-SECRET-123456"


class FakeAssistant:
    """Stands in for `Assistant`: asks the approver like the real one does, and ends the way the script says."""

    def __init__(self, config, io, runs_dir, script):
        self.io, self.script, self.stopped = io, script, False

    def handle_text(self, text, *, act=False, wait=True, hint_language=None):
        if text.startswith("do "):
            if not self.io.approve(text, act):
                return Reply("computer", error="the task was not approved: nothing was done")
            result = SimpleNamespace(
                outcome="dry run" if not act else "aborted (stopped)", act=act, uncertain=act and self.script.get("uncertain", False),
                readback="unavailable; completion unknown" if act else "not needed", steps=1, seconds=0.1,
                would_do="click Save", failure=None, answer=None, stopped=False,
            )  # fmt: skip
            return Reply("computer", "on it", task=SimpleNamespace(result=result))
        self.io.show(f"answer to: {text} {KEY}")
        return Reply("answer", f"answer to: {text} {KEY}")

    def stop(self):
        self.stopped = True
        return True

    def close(self):
        return True


@pytest.fixture
def make(tmp_path):
    started = []

    def build(toml: str = "", *, script=None):
        cfg = tmp_path / "glide.toml"
        if toml:
            cfg.write_text(toml, encoding="utf-8")
        world = build_world(tmp_path)
        srv = PanelServer(
            cfg, scope="test", env={"PATH": "x"}, runs_dir=tmp_path / "runs", state_dir=tmp_path / "state",
            file_runs_dir=world.runs, files_home=world.home,
            prober=lambda config, role, info: Row(role, info.name, "ok", f"answered with {KEY}", 0.1),
            assistant_factory=lambda config, io, runs: FakeAssistant(config, io, runs, script or {}),
        )  # fmt: skip
        srv.world = world
        threading.Thread(target=srv.httpd.serve_forever, daemon=True).start()
        started.append(srv)
        return srv

    yield build
    for srv in started:
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


def api(app, name, body=None):
    code, text, _ = req(app, "POST", f"/api/{name}", body or {}, {"X-Glide-Session": app.session})
    return code, json.loads(text)


def get(app, path):
    code, text, _ = req(app, "GET", path, None, {"X-Glide-Session": app.session})
    return code, json.loads(text)


def wait_chat(app, until="done"):
    for _ in range(100):
        _, r = get(app, "/api/read/chat")
        if r["status"] == until or r.get("approval"):
            return r
        time.sleep(0.02)
    raise AssertionError("chat did not settle")


def test_token_host_origin_and_methods(make):
    app = make()
    assert req(app, "GET", "/")[0] == 403
    code, body, r = req(app, "GET", f"/?t={app.url_token}")
    assert code == 200 and "Glide panel" in body and app.session in body
    assert "default-src 'none'" in r.getheader("Content-Security-Policy")
    assert req(app, "GET", f"/?t={app.url_token}")[0] == 403  # one-time
    assert req(app, "GET", "/api/state")[0] == 403
    assert req(app, "GET", "/api/read/status")[0] == 403
    assert req(app, "GET", "/api/state", headers={"X-Glide-Session": app.session}, host="evil.example")[0] == 403
    assert req(app, "POST", "/api/preview", {}, {"X-Glide-Session": app.session, "Origin": "http://evil.example"})[0] == 403
    assert req(app, "PUT", "/api/preview", {}, {"X-Glide-Session": app.session})[0] == 405


def test_no_key_in_any_response_or_log(make, tmp_path, caplog, capsys):
    app = make('[llm.fast]\nchain = ["openai:some-model"]\n')
    assert api(app, "key", {"env_var": "OPENAI_API_KEY", "value": KEY})[1] == {"status": "pasted"}
    assert api(app, "key", {"env_var": "PATH", "value": KEY})[0] == 400  # only a variable some provider names
    bodies = [
        req(app, "GET", "/api/state", None, {"X-Glide-Session": app.session})[1],
        json.dumps(api(app, "preview", {"changes": {}})[1]),
        json.dumps(api(app, "test", {"provider": "openai", "confirm": True})[1]),
        json.dumps(get(app, "/api/read/status")[1]),
    ]
    api(app, "chat_start", {"text": "hello"})
    bodies.append(json.dumps(wait_chat(app)))
    assert all(KEY not in b for b in bodies)
    assert "[key]" in bodies[2] or "ok" in bodies[2]
    assert KEY not in caplog.text and KEY not in capsys.readouterr().err
    assert not any(KEY in p.read_text(errors="ignore") for p in tmp_path.rglob("*") if p.is_file())


def test_test_needs_explicit_click(make):
    app = make('[llm.fast]\nchain = ["openai:m"]\n')
    assert api(app, "test", {"provider": "openai"})[0] == 400


def test_custom_provider_round_trips_through_the_real_loader(make, tmp_path):
    app = make()
    changes = {
        "providers": {"mygw": {"kind": "openai_compat", "base_url": "https://gw.example.com/v1", "api_key_env": "MYGW_KEY",
                               "options": {"extra_headers": {"X-Org": "acme"}}},
                      "systemone": {"kind": "typesafe", "base_url": "http://127.0.0.1:9/v1/systemone", "api_key_env": "S1_KEY"}},
        "roles": {"llm.fast": {"chain": [{"provider": "mygw", "model": "any-model-id"}]},
                  "classifier": {"chain": [{"provider": "systemone"}, {"provider": "llm.fast"}]}},
    }  # fmt: skip
    code, prev = api(app, "preview", {"changes": changes})
    assert code == 200 and "mygw" in prev["toml"] and "MYGW_KEY" in prev["toml"]
    assert api(app, "write", {"changes": changes, "expect": prev["expect"], "confirm": True})[0] == 200
    config = load_config(tmp_path / "glide.toml", env={"MYGW_KEY": "k-12345", "S1_KEY": "k-12345"})
    assert config.providers["mygw"].base_url == "https://gw.example.com/v1"
    assert config.providers["systemone"].api_key_env == "S1_KEY"
    assert [i.provider for i in config.slots("llm.fast")] == ["mygw"]
    row = next(p for p in get(app, "/api/state")[1]["providers"] if p["name"] == "mygw")
    assert row["builtin"] is False and row["status"] == "missing"
    config.close()
    bad = {"providers": {"x": {"kind": "openai_compat", "api_key_env": "not a var name"}}}
    code, err = api(app, "preview", {"changes": bad})
    assert code == 400 and "api_key_env" in err["error"] and "\n" not in err["error"]


def test_chain_edit_round_trips_and_bad_edit_shows_loader_error(make):
    app = make('[llm.smart]\nchain = ["openai:a"]\n')
    role = {"chain": [{"provider": "openai", "model": "m1"}, {"provider": "deepseek", "model": "m2", "options": {"max_tokens": 50}}],
            "policy": {"order": "latency", "hedge_after_s": 2.5}}  # fmt: skip
    _, prev = api(app, "preview", {"changes": {"roles": {"llm.smart": role}}})
    assert api(app, "write", {"changes": {"roles": {"llm.smart": role}}, "expect": prev["expect"], "confirm": True})[0] == 200
    view = get(app, "/api/state")[1]["roles"]["llm.smart"]
    assert [s["provider"] for s in view["chain"]] == ["openai", "deepseek"]
    assert view["chain"][1]["options"] == {"max_tokens": 50} and view["policy"] == {"order": "latency", "hedge_after_s": 2.5}
    code, err = api(app, "preview", {"changes": {"roles": {"llm.smart": {"chain": [{"provider": "nope", "model": "x"}]}}}})
    assert code == 400 and "nope" in err["error"]


def test_engine_change_writes_the_key_and_defaults_to_legacy(make, tmp_path):
    app = make()
    assert get(app, "/api/state")[1]["features"]["engine"] == "legacy"
    ch = {"features": {"engine": "structured"}}
    _, prev = api(app, "preview", {"changes": ch})
    assert "[computer]" in prev["diff"] or "+engine" in prev["diff"]
    api(app, "write", {"changes": ch, "expect": prev["expect"], "confirm": True})
    assert tomllib.loads((tmp_path / "glide.toml").read_text())["computer"] == {"engine": "structured"}
    assert api(app, "preview", {"changes": {"features": {"engine": "turbo"}}})[0] == 400
    assert get(app, "/api/state")[1]["features"]["computer"] is False  # default OFF


def test_write_needs_confirm_and_makes_a_backup(make, tmp_path):
    app = make('[llm.fast]\nchain = ["openai:a"]\n')
    ch = {"features": {"memory": True}}
    _, prev = api(app, "preview", {"changes": ch})
    assert api(app, "write", {"changes": ch, "expect": prev["expect"]})[0] == 400  # no confirm
    assert api(app, "write", {"changes": ch, "expect": "stale", "confirm": True})[0] == 400  # file moved on
    code, done = api(app, "write", {"changes": ch, "expect": prev["expect"], "confirm": True})
    assert code == 200 and done["backup"]
    assert (tmp_path / done["backup"].rsplit("/", 1)[1]).read_text() == '[llm.fast]\nchain = ["openai:a"]\n'
    assert tomllib.loads((tmp_path / "glide.toml").read_text())["memory"]["enabled"] is True


def test_chat_answer_and_computer_gating(make):
    app = make()
    api(app, "chat_start", {"text": "hello"})
    r = wait_chat(app)
    assert r["result"]["route"] == "answer" and "[key]" not in r["result"]["text"]
    assert api(app, "chat_start", {"text": "do it", "act": True, "confirm_act": True})[0] == 400  # computer off
    api(app, "chat_start", {"text": "do the thing"})  # computer off: the approver refuses, nothing starts
    r = wait_chat(app)
    assert r["result"]["error"] and any("off" in e["text"] for e in r["events"])


def test_chat_dry_run_then_real_run_with_uncertain_outcome(make):
    app = make("[panel]\ncomputer = true\n", script={"uncertain": True})
    api(app, "chat_start", {"text": "do the thing"})
    r = wait_chat(app, "approval")
    assert r["approval"]["act"] is False
    api(app, "chat_decide", {"approve": True})
    r = wait_chat(app)
    task = r["result"]["task"]
    assert task["outcome"] == "dry run" and task["can_run_for_real"] and not task["uncertain"]
    assert api(app, "chat_start", {"text": "do the thing", "act": True})[0] == 400  # needs the second confirm
    api(app, "chat_start", {"text": "do the thing", "act": True, "confirm_act": True})
    r = wait_chat(app, "approval")
    assert r["approval"]["act"] is True
    assert api(app, "chat_decide", {"approve": True})[0] == 400  # a real run needs the confirm again
    api(app, "chat_decide", {"approve": True, "confirm_act": True})
    r = wait_chat(app)
    assert r["result"]["task"]["uncertain"] and r["result"]["task"]["uncertain_note"] == "completion unknown; nothing was retried"
    marker = get(app, "/api/read/status")[1]["marker"]
    assert marker and not marker["running_now"]  # left in place: the effect was never seen
    assert api(app, "marker_clear", {})[0] == 400
    assert api(app, "marker_clear", {"confirm": True})[0] == 200
    assert get(app, "/api/read/status")[1]["marker"] is None
    assert api(app, "chat_stop")[1] == {"stopped": True}


def test_files_refuse_without_a_matching_plan_hash(make):
    app = make()
    assert api(app, "files_plan", {"root": "x", "intent": "type"})[0] == 400  # off by default
    ch = {"features": {"files": True}}  # switched on through the panel itself
    api(app, "write", {"changes": ch, "expect": api(app, "preview", {"changes": ch})[1]["expect"], "confirm": True})
    w = app.world
    w.make("a.png", "b.txt")
    code, plan = api(app, "files_plan", {"root": str(w.root), "intent": "type"})
    assert code == 200 and plan["count"] == 2
    ok = {"plan_hash": plan["plan_hash"], "confirm": True}
    assert api(app, "files_execute", {**ok, "approve": "0" * 32})[0] == 400  # wrong hash
    assert api(app, "files_execute", {**ok, "approve": plan["plan_hash"], "confirm": False})[0] == 400
    assert api(app, "files_execute", {"plan_hash": "nope", "approve": "nope", "confirm": True})[0] == 400
    assert w.listing() == ["a.png", "b.txt"]  # nothing moved
    code, done = api(app, "files_execute", {**ok, "approve": plan["plan_hash"]})
    assert code == 200 and done["status"] == "ok" and w.listing() != ["a.png", "b.txt"]
    code, _ = api(app, "files_undo", {"manifest": done["manifest"], "confirm": True})
    assert code == 200 and w.listing() == ["a.png", "b.txt"]
    assert api(app, "files_undo", {"manifest": "../../etc/passwd", "confirm": True})[0] == 400

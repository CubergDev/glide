"""The control panel's server: `glide.setup.server`'s loopback listener (one-time URL token, then a session header,
Host and Origin checks, POST-only changes, strict CSP, size limit, idle exit), with the panel's calls on top.

Keys: a pasted key lives in `self.keys` only (process memory). Every response is swept for those values before it is sent,
and nothing is logged. Providers are tested only on an explicit click (it spends tokens), and a real computer run only on
a second confirm. Page text, provider text and model replies are shown as text, never as markup.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from pathlib import Path

from glide.files import Approval, Plan, Refused, engine, planner
from glide.providers import doctor
from glide.providers.config import ALL_ROLES, GlideConfig, load_config
from glide.setup import model as setup_model
from glide.setup.server import SetupServer

from . import model
from .chat import ChatSession
from .model import PanelError
from .page import render_panel
from .settings import PanelSettings

RUN_LIST = 10
MAX_PLANS = 20
MAX_MOVES_SHOWN = 300


def real_probe(config: GlideConfig, role: str, info) -> doctor.Row:
    """One tiny real request to one slot. It spends tokens; tests replace this seam."""
    return doctor._probe(config, role, info, 20.0, time.monotonic)


class PanelServer(SetupServer):
    def __init__(
        self,
        config_path: Path,
        *,
        scope: str = "",
        env: Mapping[str, str] | None = None,
        idle_s: float = 30 * 60,
        runs_dir: Path = Path("runs"),
        state_dir: Path | None = None,
        file_runs_dir: Path | None = None,
        files_home: Path | None = None,
        prober: Callable[[GlideConfig, str, object], doctor.Row] | None = None,
        assistant_factory: Callable | None = None,
    ) -> None:
        super().__init__(config_path, env=env, idle_s=idle_s, prober=prober or real_probe)
        self.scope = scope
        self.runs_dir = runs_dir
        self.state_dir = state_dir or Path.home() / ".glide" / "panel"
        self.file_runs_dir = file_runs_dir or Path.home() / ".glide" / "file-runs"
        self.files_home = files_home
        self.plans: dict[str, Plan] = {}
        self.chat = ChatSession(
            load_config=self._load,
            settings=self.settings,
            runs_dir=runs_dir,
            state_dir=self.state_dir,
            assistant_factory=assistant_factory,
        )

    # -- helpers -----------------------------------------------------------------------------------------------

    def _load(self) -> GlideConfig:
        if self.config_path.is_file():
            return load_config(self.config_path, env=self.child_env())
        return GlideConfig.from_toml("", env=self.child_env(), source="built-in defaults")

    def settings(self) -> PanelSettings:
        _, doc = model.read(self.config_path)
        try:
            return PanelSettings.from_mapping(doc.get("panel"))
        except ValueError:
            return PanelSettings()  # a broken [panel] table never switches anything on

    def render(self) -> str:
        return render_panel(self.session, self.nonce)

    def _require(self, switch: str) -> None:
        if not getattr(self.settings(), switch):
            raise PanelError(f"{switch} is off: switch it on under Features and safety first")

    # -- reads -------------------------------------------------------------------------------------------------

    def state(self) -> dict:
        try:
            text, doc = model.read(self.config_path)
            problem = ""
        except PanelError as error:
            text, doc, problem = "", {}, str(error)
        providers = model.providers_view(doc) if not problem else []
        for row in providers:
            row["status"] = model.key_status(row["api_key_env"], self.keys, self.env)
        return {
            "file": {
                "path": str(self.config_path), "scope": self.scope, "exists": self.config_path.exists(),
                "expect": model.digest(text), "problem": problem,
            },
            "providers": providers,
            "kinds": list(model.KINDS),
            "roles": model.roles_view(doc) if not problem else {},
            "role_order": list(ALL_ROLES),
            "features": model.features_view(doc) if not problem else {},
            "engines": ["legacy", "structured"],
        }  # fmt: skip

    def read(self, name: str, query: dict) -> dict:
        if name == "chat":
            since = (query.get("since") or ["0"])[0]
            return self.chat.poll(int(since) if since.isdigit() else 0)
        if name == "status":
            return self._status()
        if name == "file_runs":
            self._require("files")
            return {"manifests": self._manifests()}
        raise PanelError("unknown call")

    def _status(self) -> dict:
        from glide import features

        out: dict = {"marker": self.chat.marker(), "runs": self._runs(), "engine": "", "rows": [], "features": [], "warnings": []}
        try:
            config = self._load()
        except ValueError as error:
            out["error"] = self.chat._clean(str(error))
            return out
        try:
            try:
                out["engine"] = features.engine_for(config)
            except ValueError as error:
                out["engine"] = f"error: {self.chat._clean(str(error))}"
            out["warnings"] = [config.scrub(w) for w in config.warnings]
            out["defaulted"] = list(config.defaulted)
            out["rows"] = [
                {"role": r.role, "slot": r.slot, "status": r.status, "detail": config.scrub(r.detail)}
                for r in doctor.doctor(config, live=False)
            ]
            out["features"] = [
                {"name": n, "ok": ok, "line": config.scrub(line)} for n, ok, line in features.feature_report(config)
            ]
        finally:
            config.close()
        return out

    def _runs(self) -> list[dict]:
        try:
            dirs = [p for p in self.runs_dir.iterdir() if p.is_dir()]
        except OSError:
            return []
        dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return [
            {"name": p.name, "when": time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime))} for p in dirs[:RUN_LIST]
        ]

    # -- calls -------------------------------------------------------------------------------------------------

    def api(self, name: str, body: dict) -> dict:
        match name:
            case "preview":
                return model.preview(self.config_path, body.get("changes"))
            case "write":
                done = model.write(self.config_path, body.get("changes"), body.get("expect"), body.get("confirm"))
                self.chat.reset()
                return done
            case "key":
                return self._key(body)
            case "test":
                return self._probe(body)
            case "chat_start":
                return self.chat.start(
                    body.get("text"), act=body.get("act") is True, engine=body.get("engine"), confirm_act=body.get("confirm_act")
                )
            case "chat_decide":
                return self.chat.decide(body.get("approve"), body.get("confirm_act"))
            case "chat_stop":
                return self.chat.stop()
            case "marker_clear":
                return self.chat.clear_marker(body.get("confirm"))
            case "files_plan":
                return self._files_plan(body)
            case "files_execute":
                return self._files_execute(body)
            case "files_undo":
                return self._files_undo(body)
        raise PanelError("unknown call")

    def _key(self, body: dict) -> dict:
        var, value = body.get("env_var"), body.get("value", "")
        _, doc = model.read(self.config_path)
        named = {r["api_key_env"] for r in model.providers_view(doc)} - {""}
        if not isinstance(var, str) or var not in named:
            raise PanelError("that variable is not named by any provider in this file")
        if value == "":
            self.keys.pop(var, None)
        elif isinstance(value, str) and setup_model.KEY_TEXT.fullmatch(value):
            self.keys[var] = value
        else:
            raise PanelError("the key must be 4 to 512 printable characters with no spaces")
        self.chat.reset()  # a running assistant was built with the old keys
        return {"status": model.key_status(var, self.keys, self.env)}  # presence only

    def _probe(self, body: dict) -> dict:
        if body.get("confirm") is not True:
            raise PanelError("a test runs only after an explicit click")
        provider = body.get("provider")
        _, doc = model.read(self.config_path)
        doc = model.apply(doc, body["changes"]) if body.get("changes") else doc
        text = model.validate(doc)
        config = GlideConfig.from_toml(text, env=self.child_env(), source="glide panel test")
        try:
            wanted = body.get("role")
            infos, role = [], ""
            for role in [wanted] if wanted else ALL_ROLES:
                try:
                    infos = [i for i in config.slots(role) if i.provider == provider]
                except ValueError:
                    infos = []
                if infos:
                    break
            if not infos:
                raise PanelError("that provider is in no role's chain: add it to a chain first")
            if not self.test_lock.acquire(blocking=False):
                raise PanelError("a test is already running")
            try:
                info = infos[0]
                if info.state != "ready":
                    return {
                        "role": role,
                        "status": f"skipped({info.short})",
                        "detail": config.scrub(info.reason),
                        "latency_s": None,
                    }
                row = self.prober(config, role, info)
                return {
                    "role": role,
                    "status": row.status,
                    "detail": self._scrub(config.scrub(row.detail)),
                    "latency_s": row.latency_s,
                }
            finally:
                self.test_lock.release()
        finally:
            config.close()

    # -- files -------------------------------------------------------------------------------------------------

    def _files_plan(self, body: dict) -> dict:
        self._require("files")
        intents = {"type": planner.ORGANIZE_BY_TYPE, "named": planner.NAMED_FOLDERS}
        if body.get("intent") not in intents or not isinstance(body.get("root"), str):
            raise PanelError("a plan needs a folder and an intent (type or named)")
        try:
            made = planner.plan(
                body["root"], intents[body["intent"]], categories=body.get("categories") or None,
                folders=body.get("folders") or None, home=self.files_home,
            )  # fmt: skip
        except Refused as error:
            raise PanelError(str(error)) from None
        if len(self.plans) >= MAX_PLANS:
            self.plans.pop(next(iter(self.plans)))
        self.plans[made.plan_hash] = made
        return {
            "plan_hash": made.plan_hash, "root": made.root, "count": len(made.moves), "preview": planner.preview(made),
            "moves": [{"source": m.source, "destination": m.destination} for m in made.moves[:MAX_MOVES_SHOWN]],
        }  # fmt: skip

    def _files_execute(self, body: dict) -> dict:
        self._require("files")
        plan = self.plans.get(str(body.get("plan_hash")))
        if plan is None:
            raise PanelError("there is no such plan: make one first")
        if body.get("confirm") is not True:
            raise PanelError("running a plan needs an explicit confirm")
        try:
            report = engine.execute(
                plan, Approval(str(body.get("approve"))), manifest_dir=self.file_runs_dir, home=self.files_home
            )
        except Refused as error:
            raise PanelError(str(error)) from None
        self.plans.pop(plan.plan_hash, None)
        return _report(report)

    def _files_undo(self, body: dict) -> dict:
        self._require("files")
        if body.get("confirm") is not True:
            raise PanelError("undo needs an explicit confirm")
        name = str(body.get("manifest") or "")
        if name not in self._manifests():
            raise PanelError("that is not one of this panel's run manifests")
        try:
            return _report(engine.undo(self.file_runs_dir / name, home=self.files_home))
        except Refused as error:
            raise PanelError(str(error)) from None

    def _manifests(self) -> list[str]:
        try:
            return sorted((p.name for p in self.file_runs_dir.glob("*.json")), reverse=True)[:30]
        except OSError:
            return []


def _report(report) -> dict:
    return {
        "status": report.status,
        "actions": [
            {"status": a.status, "source": a.source, "destination": a.destination, "reason": a.reason} for a in report.actions
        ][:MAX_MOVES_SHOWN],
        "manifest": Path(report.manifest).name if report.manifest else "",
    }

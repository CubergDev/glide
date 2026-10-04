"""What the panel edits, as plain data: glide.toml read into a dict, changed by a patch from the page, written back.

Nothing here touches the network, a key value or the machine. Every result is checked by the real loaders (the provider
config, `[panel]`, `[computer]`, `[research]`, `[memory]`, `[routing]`), and their one-line error is what the page shows.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import os
import re
import tomllib
from collections.abc import Mapping
from pathlib import Path

from glide.providers.config import (
    ALL_ROLES,
    KINDS,
    PRESETS,
    ROLE_KINDS,
    ROLES,
    GlideConfig,
    _default_roles,
    _providers,
)
from glide.setup import model as setup_model

from . import tomlio
from .settings import PanelSettings

PanelError = setup_model.SetupError
NAME = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,39}")
TEXT = re.compile(r"[^\x00-\x1f\x7f]{0,300}")
HEADER = "# Written by `glide panel`. It holds no keys: each provider reads its key from an environment variable."
MAX_CHAIN = 8
POLICY_KEYS = ("order", "fail_threshold", "cooldown_s", "auth_cooldown_s", "hedge_after_s", "latency_alpha", "deadline_s")

# feature name -> (table, key). `webhooks` is special: on means `[webhooks] config = "webhooks.json"`.
FEATURES: dict[str, tuple[str, str]] = {
    "voice": ("panel", "voice"),
    "computer": ("panel", "computer"),
    "files": ("panel", "files"),
    "point_ask": ("panel", "point_ask"),
    "record_content": ("panel", "record_content"),
    "retention_days": ("panel", "retention_days"),
    "confirm_acting": ("routing", "confirm_acting"),
    "confirm_tasks": ("speech", "confirm_tasks"),
    "engine": ("computer", "engine"),
    "research_calls": ("research", "calls"),
    "memory": ("memory", "enabled"),
    "memory_auto": ("memory", "auto_capture"),
}
FEATURE_DEFAULTS = {
    "voice": False, "computer": False, "files": False, "point_ask": False, "record_content": False, "retention_days": 30,
    "confirm_acting": False, "confirm_tasks": True, "engine": "legacy", "research_calls": 24, "memory": False,
    "memory_auto": False, "webhooks": False,
}  # fmt: skip


def locate(explicit: str | None, *, cwd: Path, home: Path) -> tuple[Path, str]:
    """The glide.toml the panel edits and which kind it is: explicit, project-local, user-level, or neither yet."""
    if explicit:
        return Path(explicit).expanduser(), "chosen with --config"
    here = cwd / "glide.toml"
    user = home / ".config" / "glide" / "glide.toml"
    if here.is_file():
        return here, "project-local (this folder's glide.toml, read before the user-level one)"
    if user.is_file():
        return user, "user-level (~/.config/glide/glide.toml)"
    return here, "none yet: writing creates the project-local ./glide.toml"


def read(path: Path) -> tuple[str, dict]:
    if not path.exists():
        return "", {}
    try:
        text = path.read_text(encoding="utf-8")
        return text, tomllib.loads(text)
    except tomllib.TOMLDecodeError as error:
        raise PanelError(f"{path.name} is not valid TOML: {error}") from None
    except (OSError, UnicodeDecodeError):
        raise PanelError(f"cannot read {path}") from None


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# -- reading the file into what the page shows ----------------------------------------------------------------


def slot_view(entry: object) -> dict:
    if isinstance(entry, str):
        if entry in ROLES:
            return {"provider": entry, "model": "", "options": {}, "name": ""}
        provider, _, model = entry.partition(":")
        return {"provider": provider, "model": model, "options": {}, "name": ""}
    e = entry if isinstance(entry, Mapping) else {}
    return {
        "provider": str(e.get("provider", "")),
        "model": str(e.get("model", "")),
        "options": dict(e.get("options") or {}),
        "name": str(e.get("name", "")),
    }


def role_table(doc: Mapping, role: str) -> Mapping | None:
    found = (doc.get("llm") or {}).get(role.split(".", 1)[1]) if role.startswith("llm.") else doc.get(role)
    return found if isinstance(found, Mapping) else None


def roles_view(doc: Mapping) -> dict:
    defaults = _default_roles()
    out = {}
    for role in ALL_ROLES:
        table = role_table(doc, role)
        d = defaults.get(role)
        out[role] = {
            "defined": table is not None,
            "chain": [slot_view(x) for x in (table or {}).get("chain", [])],
            "policy": {k: table[k] for k in POLICY_KEYS if table and k in table},
            "default_chain": [
                {"provider": s.provider or s.uses, "model": s.model, "options": dict(s.options), "name": ""} for s in d.slots
            ]
            if d
            else [],
            "stands_on": "llm.smart" if role in ("llm.planner", "llm.research") else "",
            "kinds": list(ROLE_KINDS[role.split(".")[0]]),
        }
    return out


def providers_view(doc: Mapping) -> list[dict]:
    specs = _providers(doc.get("providers") or {})
    mine = doc.get("providers") or {}
    rows = []
    for name, spec in specs.items():
        rows.append(
            {
                "name": name,
                "kind": spec.kind,
                "base_url": spec.base_url,
                "api_key_env": spec.api_key_env,
                "options": dict(spec.options),
                "builtin": name in PRESETS,
                "in_file": name in mine,
                "export": f"export {spec.api_key_env}='<paste your key here>'" if spec.api_key_env else "",
            }
        )
    return rows


def features_view(doc: Mapping) -> dict:
    out = dict(FEATURE_DEFAULTS)
    for name, (table, key) in FEATURES.items():
        t = doc.get(table)
        if isinstance(t, Mapping) and key in t:
            out[name] = t[key]
    out["webhooks"] = isinstance(doc.get("webhooks"), Mapping)
    return out


def key_status(var: str, pasted: Mapping[str, str], env: Mapping[str, str]) -> str:
    """'none needed', 'set' (in the environment), 'pasted' (this process only) or 'missing'. Never a value."""
    if not var:
        return "none needed"
    if env.get(var):
        return "set"
    return "pasted" if pasted.get(var) else "missing"


# -- applying a patch -------------------------------------------------------------------------------------------


def _scalar(value: object, where: str) -> object:
    if isinstance(value, (bool, int, float)) or (isinstance(value, str) and TEXT.fullmatch(value)):
        return value
    raise PanelError(f"{where} must be text, a number or true/false")


def _options(raw: object, where: str) -> dict:
    if raw in (None, ""):
        return {}
    if not isinstance(raw, Mapping) or len(raw) > 20:
        raise PanelError(f"{where} must be a list of name = value pairs")
    out = {}
    for k, v in raw.items():
        if not isinstance(k, str) or not NAME.fullmatch(k.replace(".", "_")):
            raise PanelError(f"{where} has a name that is not a plain word")
        out[k] = (
            {str(a): _scalar(b, where) for a, b in v.items()} if isinstance(v, Mapping) and len(v) <= 20 else _scalar(v, where)
        )
    return out


def _provider_table(name: str, raw: object) -> dict:
    where = f"provider {name}"
    if not isinstance(raw, Mapping):
        raise PanelError(f"{where} must be an object")
    out: dict = {}
    for key in ("kind", "base_url", "api_key_env"):
        v = str(raw.get(key) or "").strip()
        if v and not TEXT.fullmatch(v):
            raise PanelError(f"{where}: {key} has characters it cannot hold")
        if v:
            out[key] = v
    if out.get("kind") and out["kind"] not in KINDS:
        raise PanelError(f"{where}: kind must be one of {', '.join(KINDS)}")
    opts = _options(raw.get("options"), f"{where} options")
    if opts:
        out["options"] = opts
    return out


def _slot_entry(raw: object, role: str) -> object:
    if not isinstance(raw, Mapping):
        raise PanelError(f"{role}: a chain entry must be an object")
    provider = str(raw.get("provider") or "").strip()
    model = str(raw.get("model") or "").strip()
    name = str(raw.get("name") or "").strip()
    if not provider or not TEXT.fullmatch(provider) or not TEXT.fullmatch(model) or not TEXT.fullmatch(name):
        raise PanelError(f"{role}: every entry needs a provider, and its model id and name must be plain text")
    options = _options(raw.get("options"), f"{role} options")
    if provider in ROLES:  # "llm.fast" as a classifier slot
        return provider
    if not options and not name:
        return f"{provider}:{model}" if model else provider
    entry: dict = {"provider": provider}
    if model:
        entry["model"] = model
    if options:
        entry["options"] = options
    if name:
        entry["name"] = name
    return entry


def _role_table(role: str, raw: object) -> dict:
    if not isinstance(raw, Mapping):
        raise PanelError(f"{role} must be an object")
    chain = raw.get("chain")
    if not isinstance(chain, list) or not 0 < len(chain) <= MAX_CHAIN:
        raise PanelError(f"{role} needs a chain of 1 to {MAX_CHAIN} entries")
    table: dict = {"chain": [_slot_entry(x, role) for x in chain]}
    policy = raw.get("policy") or {}
    if not isinstance(policy, Mapping):
        raise PanelError(f"{role} policy must be an object")
    for key, value in policy.items():
        if key not in POLICY_KEYS or (key == "deadline_s" and not role.startswith("llm.")):
            raise PanelError(f"{role}: {str(key)[:30]!r} is not a policy setting")
        if value in ("", None):
            continue
        if key == "order":
            table[key] = str(value)
        elif isinstance(value, bool) or not isinstance(value, (int, float)):
            raise PanelError(f"{role} {key} must be a number")
        else:
            table[key] = value
    return table


def apply(doc: Mapping, changes: object) -> dict:
    """A copy of `doc` with the patch applied: {providers: {name: table | None}, roles: {role: table | None},
    features: {name: value}}. Unknown parts are errors."""
    if not isinstance(changes, Mapping) or set(changes) - {"providers", "roles", "features"}:
        raise PanelError("the change must hold providers, roles or features")
    new = copy.deepcopy(dict(doc))
    for name, raw in (changes.get("providers") or {}).items():
        if not isinstance(name, str) or not NAME.fullmatch(name) or name in ROLES:
            raise PanelError("a provider name is letters, digits, - and _, starting with a letter")
        table = new.setdefault("providers", {})
        if raw is None:
            table.pop(name, None)
        else:
            table[name] = _provider_table(name, raw)
        if not table:
            new.pop("providers")
    for role, raw in (changes.get("roles") or {}).items():
        if role not in ALL_ROLES:
            raise PanelError("that is not a role")
        parent = new.setdefault("llm", {}) if role.startswith("llm.") else new
        key = role.split(".", 1)[-1]
        if raw is None:
            parent.pop(key, None)
        else:
            parent[key] = _role_table(role, raw)
        if role.startswith("llm.") and not parent:
            new.pop("llm")
    for name, value in (changes.get("features") or {}).items():
        _feature(new, name, value)
    return new


def _feature(doc: dict, name: str, value: object) -> None:
    if name == "webhooks":
        if not isinstance(value, bool):
            raise PanelError("webhooks must be on or off")
        if value:
            doc.setdefault("webhooks", {}).setdefault("config", "webhooks.json")
        else:
            doc.pop("webhooks", None)
        return
    if name not in FEATURES:
        raise PanelError("that is not a feature switch")
    table, key = FEATURES[name]
    default = FEATURE_DEFAULTS[name]
    ok = (
        value in ("legacy", "structured")
        if name == "engine"
        else type(value) is int
        if isinstance(default, int) and not isinstance(default, bool)
        else isinstance(value, bool)
    )
    if not ok:
        raise PanelError(f"{name} has a value it cannot take")
    doc.setdefault(table, {})[key] = value


def validate(doc: Mapping) -> str:
    """The file text for `doc`, after every loader that reads it has accepted it. PanelError names the first problem."""
    from glide.computer import config as computer_config
    from glide.memory.settings import MemorySettings
    from glide.routing.settings import RoutingSettings

    try:
        text = tomlio.dumps(doc, HEADER)
        GlideConfig.from_toml(text, env={}, source="glide.toml").close()
        PanelSettings.from_mapping(doc.get("panel"))
        computer_config.engine(doc.get("computer") or {}, {}, None)
        computer_config.research_budget(doc.get("research"), {})
        MemorySettings.from_mapping(doc.get("memory"), {})
        RoutingSettings.from_table(doc.get("routing") or {})
    except (ValueError, TypeError) as error:
        raise PanelError(str(error).splitlines()[0][:300] if str(error) else type(error).__name__) from None
    return text


def preview(path: Path, changes: object) -> dict:
    old, doc = read(path)
    text = validate(apply(doc, changes))
    return {
        "toml": text,
        "diff": tomlio.diff(old, text),
        "expect": digest(old),
        "comments_note": "comments in the file are not kept",
    }


def write(path: Path, changes: object, expect: object, confirm: object) -> dict:
    if confirm is not True:
        raise PanelError("writing needs an explicit confirm")
    old, doc = read(path)
    if expect != digest(old):
        raise PanelError("the file changed since you looked: reload and review the diff again")
    text = validate(apply(doc, changes))
    if text == old:
        return {"written": str(path), "backup": "", "unchanged": True}
    backup = setup_model.write_config(path, text)
    with_mode(path)
    return {"written": str(path), "backup": str(backup) if backup else "", "unchanged": False}


def with_mode(path: Path) -> None:
    with contextlib.suppress(OSError):
        os.chmod(path, 0o600)

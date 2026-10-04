"""What the wizard decides, as plain data, and how it becomes a glide.toml. No network, no server, no key values."""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import sys
import time
import tomllib
from collections.abc import Mapping
from pathlib import Path

from glide.providers.config import PRESETS, ROLE_KINDS, ROLES, GlideConfig, _default_roles

KEY_LINKS = tomllib.loads((Path(__file__).parent / "keys.toml").read_text(encoding="utf-8"))
SAFE_TEXT = re.compile(r"[A-Za-z0-9._:/@+\-]{1,100}")
KEY_TEXT = re.compile(r"[\x21-\x7e]{4,512}")  # printable, no whitespace
ENV_NAME = re.compile(r"[A-Z_][A-Z0-9_]*")
FEATURES = ("computer", "webhooks", "memory")
DEFAULT_FEATURES = {"computer": False, "webhooks": False, "memory": False}
MODES = ("ask", "chat", "voice", "ui")
CLASSIFIER_OVER_LLM = "llm.fast"


class SetupError(ValueError):
    """The wizard's input is wrong. The message names the field, never a value."""


def candidates(role: str) -> list[str]:
    """The providers that can serve `role`, from the registry."""
    kinds = ROLE_KINDS[role.split(".")[0]]
    names = [n for n, spec in PRESETS.items() if spec.kind in kinds]
    return names + ([CLASSIFIER_OVER_LLM] if role == "classifier" else [])


def default_slot(role: str, provider: str) -> dict:
    """The built-in model (and options) for this provider in this role, or {} when Glide has none for it."""
    for slot in _default_roles()[role].slots:
        if slot.provider == provider:
            return {"model": slot.model, "options": dict(slot.options)}
    return {}


def provider_rows() -> list[dict]:
    rows = []
    for name, spec in PRESETS.items():
        link = KEY_LINKS.get(name, {}).get("url", "")
        rows.append(
            {
                "name": name,
                "env": spec.api_key_env,
                "link": link if link.startswith("https://") else "",
                "roles": [r for r in ROLES if name in candidates(r)],
            }
        )
    return rows


def preset(provider: str) -> dict[str, list[dict]]:
    """'One key is enough': every role this provider can serve uses it, TTS keeps the local voice as the fallback."""
    if provider not in PRESETS or not PRESETS[provider].api_key_env:
        raise SetupError("preset: choose a provider that takes a key")
    roles: dict[str, list[dict]] = {}
    for role in ("llm.fast", "llm.smart", "stt"):
        slot = default_slot(role, provider)
        if slot:
            roles[role] = [{"provider": provider, **slot}]
    roles["tts"] = [{"provider": "macos_say"}]
    roles["classifier"] = [{"provider": CLASSIFIER_OVER_LLM}]
    return roles


def _scalar(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    return json.dumps(str(value))  # JSON escapes are valid TOML basic-string escapes


def _slot_text(slot: dict) -> str:
    provider, model, options = slot["provider"], slot.get("model", ""), slot.get("options") or {}
    if provider == CLASSIFIER_OVER_LLM:
        return json.dumps(provider)
    if not options:
        return json.dumps(f"{provider}:{model}" if model else provider)
    inner = ", ".join(f"{k} = {_scalar(v)}" for k, v in options.items())
    return f"{{ provider = {json.dumps(provider)}, model = {json.dumps(model)}, options = {{ {inner} }} }}"


def clean_state(raw: object) -> dict:
    """Validate what the page or the prompts sent. Unknown keys are errors; nothing is trusted."""
    if not isinstance(raw, Mapping):
        raise SetupError("state must be an object")
    extra = set(raw) - {"features", "roles"}
    if extra:
        raise SetupError("unknown field in state")
    features = dict(DEFAULT_FEATURES)
    for k, v in (raw.get("features") or {}).items():
        if k not in FEATURES or not isinstance(v, bool):
            raise SetupError(f"features.{k} is not a known on/off switch")
        features[k] = v
    roles: dict[str, list[dict]] = {}
    for role, chain in (raw.get("roles") or {}).items():
        if role not in ROLES or not isinstance(chain, list) or not 0 < len(chain) <= 6:
            raise SetupError(f"roles.{role} is not a chain of 1 to 6 providers")
        out = []
        for slot in chain:
            if not isinstance(slot, Mapping):
                raise SetupError(f"roles.{role} has an entry that is not an object")
            provider = slot.get("provider")
            if provider not in candidates(role):
                raise SetupError(f"roles.{role}: {str(provider)[:30]!r} cannot serve this role")
            model = str(slot.get("model") or "").strip()
            if model and not SAFE_TEXT.fullmatch(model):
                raise SetupError(f"roles.{role}: the model id has characters a model id does not use")
            kind = PRESETS[provider].kind if provider in PRESETS else ""
            if provider != CLASSIFIER_OVER_LLM and not model and kind not in ("macos_say", "typesafe"):
                raise SetupError(f"roles.{role}: {provider} needs a model id")
            options = (
                default_slot(role, provider).get("options", {}) if model == default_slot(role, provider).get("model") else {}
            )
            entry = {"provider": provider}
            if model:
                entry |= {"model": model, "options": options}
            out.append(entry)
        roles[role] = out
    return {"features": features, "roles": roles}


def render_toml(state: dict) -> str:
    """The exact glide.toml. It names variables through the providers' own defaults and never contains a key."""
    s = clean_state(state)
    lines = [
        "# Written by `glide setup`. It holds no keys: each provider reads its key from an environment variable.",
        "# Model ids are unverified until `glide doctor --live` has run with real keys.",
        "",
    ]
    for role in ROLES:
        chain = s["roles"].get(role)
        if not chain:
            continue
        lines += [f"[{role}]", "chain = [", *[f"  {_slot_text(slot)}," for slot in chain], "]", ""]
    if s["features"]["memory"]:
        lines += ["[memory]", "enabled = true", ""]
    if s["features"]["webhooks"]:
        lines += ["[webhooks]", 'config = "webhooks.json"  # nothing runs until that file says "enabled": true', ""]
    text = "\n".join(lines).rstrip() + "\n"
    check(text)
    return text


def check(text: str) -> None:
    """The text must load with the real loader. Built with no environment, so nothing here can hold a key."""
    try:
        GlideConfig.from_toml(text, env={}, source="glide setup preview").close()
    except ValueError as e:
        raise SetupError(f"the generated glide.toml does not load: {e}") from None


def key_status(provider: str, pasted: Mapping[str, str], env: Mapping[str, str] | None = None) -> str:
    """'set' (in the environment), 'pasted' (held in this process only), 'missing' or 'none needed'. Never the value."""
    var = PRESETS[provider].api_key_env
    if not var:
        return "none needed"
    if (env if env is not None else os.environ).get(var):
        return "set"
    return "pasted" if pasted.get(var) else "missing"


def export_line(provider: str) -> str:
    var = PRESETS[provider].api_key_env
    return f"export {var}='<paste your key here>'" if var else ""


def write_config(path: Path, text: str) -> Path | None:
    """Back up an existing file next to it (never over an older backup), then replace it atomically. Returns the backup."""
    check(text)
    backup = None
    if path.exists():
        stamp = time.strftime("%Y%m%d-%H%M%S")
        backup = path.with_name(f"{path.name}.bak-{stamp}")
        n = 1
        while backup.exists():
            backup = path.with_name(f"{path.name}.bak-{stamp}-{n}")
            n += 1
        shutil.copy2(path, backup)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    return backup


def launch_argv(mode: str, config_path: Path, *, act: bool, text: str = "") -> list[str]:
    """The Glide command for `mode`, as an argument list (never a shell string)."""
    if mode not in MODES:
        raise SetupError("mode must be ask, chat, voice or ui")
    base = [sys.executable, "-m", "glide.cli", "--config", str(config_path)]
    if mode == "ui":
        return [*base, "app-server"]
    argv = [*base, {"ask": "ask", "chat": "chat", "voice": "voice"}[mode]]
    if act:
        argv.append("--act")
    if mode == "ask":
        if not text.strip() or len(text) > 2000:
            raise SetupError("ask needs a request of up to 2000 characters")
        argv += ["--", text]
    return argv


def equivalent_command(mode: str, config_path: Path, *, act: bool) -> str:
    argv = launch_argv(mode, config_path, act=act, text="<your request>")
    return shlex.join(["glide", *argv[3:]])

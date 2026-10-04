"""What the optional features are set to, and which optional packages they need. Standard library only, and it
reads nothing but glide.toml, the webhook JSON file's `enabled` flag and (for a voice-activity model) that one file.

`glide doctor` prints `feature_report`. `glide webhooks` takes its default file from `webhooks_file`. Nothing here
imports an extra, starts a server, opens a device, creates a directory, connects anywhere or downloads a model.
"""

from __future__ import annotations

import importlib.util
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

# The import names the webhooks extra provides (pyproject.toml's `webhooks` list); the voice stack's are named where used.
WEBHOOKS_MODULES = ("fastapi", "uvicorn", "pydantic", "jwt", "cryptography")
DEFAULT_WEBHOOKS_FILE = "webhooks.json"
WEBHOOKS_ENV = "GLIDE_WEBHOOK_CONFIG"
_WEBHOOKS_KEYS = ("config",)


def missing(modules: tuple[str, ...]) -> list[str]:
    """The names in `modules` that cannot be imported. Looks them up without importing any."""
    return [name for name in modules if importlib.util.find_spec(name) is None]


def extra_message(command: str, extra: str, lacking: list[str]) -> str:
    """One line for a command that cannot run without an extra: what is missing and how to install it."""
    return f"{command} needs the {extra} extra (missing: {', '.join(lacking)}): uv sync --extra {extra}"


def webhooks_file(environ: Mapping[str, str] | None, toml: Path | None, *, foreign: bool = False) -> Path:
    """The webhook service's JSON file: $GLIDE_WEBHOOK_CONFIG, else `config = ...` under [webhooks] in `toml` (the
    glide.toml in use, or None), else webhooks.json. A `foreign` file (a project's own glide.toml) may not name it."""
    from .memory.settings import SettingsError, read_table

    env = os.environ if environ is None else environ
    if named := (env.get(WEBHOOKS_ENV) or "").strip():
        return Path(named).expanduser()
    table = read_table("webhooks", toml, foreign=foreign)
    for key in table:
        if key not in _WEBHOOKS_KEYS:
            raise SettingsError(f"[webhooks] has an unknown key {key!r} (known: {', '.join(_WEBHOOKS_KEYS)})")
    path = table.get("config", DEFAULT_WEBHOOKS_FILE)
    if not isinstance(path, str) or not path.strip():
        raise SettingsError("[webhooks] config must be a path")
    return Path(path).expanduser()


def feature_report(glide_config, environ: Mapping[str, str] | None = None) -> list[tuple[str, bool, str]]:
    """(feature, ok, line) for voice, memory, webhooks and mcp. `ok` is False only for a setting that is wrong; a feature
    that is switched off, or an extra that is not installed, is a state to report, not a failure."""
    env = os.environ if environ is None else environ
    rows = []
    for name, report in (("voice", _voice), ("memory", _memory), ("webhooks", _webhooks), ("mcp", _mcp)):
        try:
            rows.append((name, True, report(glide_config, env)))
        except ValueError as error:  # SettingsError and ConfigError are both ValueErrors, and name the setting, never a value
            rows.append((name, False, f"error: {error}"))
    return rows


def _voice(glide_config, env) -> str:
    settings = glide_config.voice
    lacking = missing(("sounddevice",))  # the device is what every voice session needs; numpy and onnxruntime only Silero
    state = "speech extra installed" if not lacking else f"speech extra missing ({', '.join(lacking)}): uv sync --extra speech"
    if settings.vad == "energy" or (settings.vad == "auto" and not settings.silero_configured):
        detector = "loudness (no model configured)" if settings.vad == "auto" else "loudness"
        return f"vad {settings.vad}: {detector}; {state}"
    model = _vad_model(settings)
    if settings.vad == "silero" and not model.startswith("model ok"):
        raise ValueError(f"vad is silero but the {model}")
    fallback = "" if model.startswith("model ok") else "; auto falls back to loudness"
    packages = missing(("numpy", "onnxruntime"))
    needs = f"; silero needs {', '.join(packages)} (speech extra)" if packages else ""
    return f"vad {settings.vad}: {model}{fallback}{needs}; {state}"


def _vad_model(settings) -> str:
    """The configured voice-activity model's state, from the file on disk and its digest. Never fetches."""
    from .speech.vad import VadError, check_sha256, file_sha256

    path = Path(settings.vad_model_path).expanduser()
    try:
        expected = check_sha256(settings.vad_model_sha256)
    except VadError as error:
        return str(error)
    if not path.is_file():
        hint = "; fetch it with glide.speech.vad.install_model" if settings.vad_model_url else ""
        return f"model file {path} is missing{hint}"
    try:
        actual = file_sha256(path)
    except OSError:
        return f"model file {path} cannot be read"
    return f"model ok ({path})" if actual == expected else f"model file {path} does not match the configured checksum"


def _memory(glide_config, env) -> str:
    from .memory.settings import MemorySettings

    settings = MemorySettings.from_mapping(table(glide_config, "memory"), env)
    if not settings.enabled:
        return "off (set enabled = true under [memory], or GLIDE_MEMORY=1, to turn it on)"
    capture = "auto_capture on" if settings.auto_capture else "auto_capture off"
    return f"on, {capture}; database {settings.database_path(env)}"


def _webhooks(glide_config, env) -> str:
    path = webhooks_file(env, _source(glide_config), foreign=_foreign(glide_config))
    if not path.is_file():
        return f"off (no {path})"
    try:
        enabled = json.loads(path.read_text(encoding="utf-8")).get("enabled") is True
    except (OSError, ValueError, AttributeError):
        raise ValueError(f"{path} cannot be read as the webhook settings (see glide/webhooks/README.md)") from None
    lacking = missing(WEBHOOKS_MODULES)
    needs = f"; the webhooks extra is missing ({', '.join(lacking)}): uv sync --extra webhooks" if lacking else ""
    return f"{'enabled' if enabled else 'off (enabled is not true)'} in {path}{needs}"


def _mcp(glide_config, env) -> str:
    from .mcp.config import McpSettings
    from .memory.settings import MemorySettings

    settings = McpSettings.from_mapping(table(glide_config, "mcp"))
    if settings.server_memory != "off" and not MemorySettings.from_mapping(table(glide_config, "memory"), env).enabled:
        # `glide mcp serve` refuses this combination (mcp/cli.py), so the doctor must not call it healthy
        raise ValueError('[mcp] server_memory needs memory on: set [memory] enabled = true, or server_memory = "off"')
    names = ", ".join(spec.name for spec in settings.servers) or "none"
    return f"client servers configured: {names} (none started); server_memory {settings.server_memory}"


def _source(glide_config) -> Path | None:
    """The glide.toml the configuration was read from, or None when it is the built-in defaults. It is never searched
    for again: the tables are read from the file the providers were."""
    source = Path(getattr(glide_config, "source", "") or ".")
    return source if source.is_file() else None


def _foreign(glide_config) -> bool:
    return bool(getattr(glide_config, "foreign", False))


def table(glide_config, name: str) -> Mapping[str, Any]:
    """One top-level table of the glide.toml the configuration was read from ({} when there is none, or no such table)."""
    from .memory.settings import read_table

    return read_table(name, _source(glide_config), foreign=_foreign(glide_config))

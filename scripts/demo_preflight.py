#!/usr/bin/env python3
"""Offline preflight for the recorded Glide demo (docs/DEMO_PREFLIGHT.md).

It only reads. It never connects to anything (not the browser's debugging port, not a provider), never starts a program,
never touches the screen, the microphone or any macOS permission, and never prints the value of an environment variable.
It prints the commands YOU run to start the browser and to check it. Those commands take over this machine.

    uv run python scripts/demo_preflight.py [--config PATH] [--env-file PATH] [--port 9222]

Exit status: 0 when nothing blocks the demo (warnings are allowed), 1 when something does, 2 on bad arguments.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import platform
import re
import shlex
import shutil
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

OK, WARN, FAIL, INFO = "ok", "warn", "FAIL", "info"

# The import names each extra provides (pyproject.toml). Looked up with find_spec: nothing is imported.
EXTRAS: dict[str, tuple[str, ...]] = {
    "speech": ("numpy", "onnxruntime", "sounddevice"),
    "aec": ("livekit",),
    "ui": ("PySide6",),
}
TOOLS = ("uv", "swift", "ffmpeg", "node")  # `which` only; nothing is run
BROWSER_APPS = {
    "chrome": "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "brave": "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
}
PROFILE_DIR = "$HOME/glide-demo-profile"  # outside the repository, never the everyday profile
_ENV_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$")
ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Line:
    status: str
    topic: str
    text: str


# -- pure parts (tested) -------------------------------------------------------------------------------------


def dotenv_names(text: str) -> dict[str, bool]:
    """The variable names of a .env file mapped to whether they have a non-empty value. Values are dropped here."""
    found: dict[str, bool] = {}
    for raw in text.splitlines():
        if raw.lstrip().startswith("#"):
            continue
        match = _ENV_LINE.match(raw)
        if match:
            value = match.group(2).split(" #", 1)[0].strip().strip("'\"")
            found[match.group(1)] = bool(value)
    return found


def effective_env(environ: Mapping[str, str], dotenv_text: str | None) -> dict[str, str]:
    """The environment as Glide will see it: the process environment, then .env for names not already set (as `glide` does)."""
    merged = dict(environ)
    if dotenv_text:
        for raw in dotenv_text.splitlines():
            if raw.lstrip().startswith("#"):
                continue
            match = _ENV_LINE.match(raw)
            if match:
                merged.setdefault(match.group(1), match.group(2).split(" #", 1)[0].strip().strip("'\""))
    return merged


def names_set(names: list[str], env: Mapping[str, str]) -> dict[str, bool]:
    """Whether each variable NAME has a non-blank value. Never returns a value."""
    return {name: bool((env.get(name) or "").strip()) for name in names}


def extras_report(find: Callable[[str], object | None] = importlib.util.find_spec) -> dict[str, list[str]]:
    """For each extra, the import names that are missing (empty list: installed)."""
    lacking: dict[str, list[str]] = {}
    for extra, modules in EXTRAS.items():
        lacking[extra] = [m for m in modules if find(m) is None]
    return lacking


def chrome_command(port: int, app: str = "chrome") -> str:
    """The command that starts a throwaway-profile browser with remote debugging on loopback only. YOU run it."""
    binary = BROWSER_APPS[app]
    return (
        f"{shlex.quote(binary)} --remote-debugging-port={port} --remote-debugging-address=127.0.0.1 "
        f'--user-data-dir="{PROFILE_DIR}" --no-first-run about:blank'
    )


def verify_command(port: int) -> str:
    """The loopback check YOU run once the browser is up: it answers with the browser's version as JSON."""
    return f"curl -s http://127.0.0.1:{port}/json/version"


def port_of(endpoint: str) -> int | None:
    from glide.computer import browser_settings

    try:
        return browser_settings.loopback_origin(endpoint)[1]
    except ValueError:
        return None


# -- the checks (read only) ----------------------------------------------------------------------------------


def check_python() -> list[Line]:
    version = sys.version_info
    status = OK if version >= (3, 12) else FAIL
    return [Line(status, "python", f"{platform.python_version()} on {platform.system()} {platform.machine()} (3.12 or later)")]


def check_extras(find: Callable[[str], object | None] = importlib.util.find_spec) -> list[Line]:
    lacking = extras_report(find)
    lines = []
    purpose = {
        "speech": ("needed for scenes a, b (glide voice)", FAIL, "uv sync --extra speech"),
        "aec": (
            "optional: talking over Glide through speakers; without it speaker mode uses the numpy filter or none",
            WARN,
            "uv sync --extra aec",
        ),
        "ui": ("optional: the pet window (scene f fallback)", WARN, "uv sync --extra ui"),
    }
    for extra, missing in lacking.items():
        why, bad, fix = purpose[extra]
        if missing:
            lines.append(Line(bad, f"extra:{extra}", f"missing {', '.join(missing)} ({why}). Install: {fix}"))
        else:
            lines.append(Line(OK, f"extra:{extra}", "installed"))
    lines.append(
        Line(
            INFO, "extra:note", "run every demo command as `uv run --all-extras ...` or sync once and use `uv run --no-sync ...`"
        )
    )
    return lines


def check_tools(which: Callable[[str], str | None] = shutil.which) -> list[Line]:
    out = []
    for tool in TOOLS:
        found = which(tool)
        out.append(Line(OK if found else INFO, f"tool:{tool}", found or "not found on PATH"))
    return out


def load_configuration(path: str | None, env: Mapping[str, str]):
    """(config, lines). The loader is Glide's own, so a file it refuses is refused here with the same message."""
    from glide.providers.config import ConfigError, load_config

    try:
        config = load_config(path, env=env)
    except (ConfigError, ValueError, OSError) as error:
        return None, [Line(FAIL, "glide.toml", f"{type(error).__name__}: {' '.join(str(error).split())[:300]}")]
    lines = [Line(OK, "glide.toml", f"loaded {config.source}")]
    if config.defaulted:
        lines.append(Line(INFO, "glide.toml", f"built-in chains in use for: {', '.join(config.defaulted)}"))
    return config, lines


def check_chains(config) -> list[Line]:
    """Each slot: ready, or skipped with the variable NAME to set. Building a slot makes no request."""
    from glide.providers.config import ALL_ROLES, ConfigError

    needed = {"llm.fast": FAIL, "llm.smart": FAIL, "classifier": FAIL, "stt": WARN, "tts": WARN}
    lines = []
    for role in ALL_ROLES:
        try:
            slots = config.slots(role)
        except ConfigError as error:
            lines.append(Line(FAIL, role, f"cannot be set up: {config.scrub(' '.join(str(error).split()))[:200]}"))
            continue
        ready = [s for s in slots if s.state == "ready"]
        if role in ("llm.planner", "llm.research") and role not in config.roles:
            lines.append(
                Line(
                    INFO, role, "no chain of its own: it uses llm.smart (fine for a demo; research quality is the smart model's)"
                )
            )
            continue
        if ready:
            names = ", ".join(s.name for s in ready)
            first = slots[0]
            note = (
                "" if first.state == "ready" else f"; first slot {first.name} is skipped, so the demo starts on {ready[0].name}"
            )
            lines.append(Line(OK, role, f"{len(ready)} of {len(slots)} slots ready ({names}){note}"))
        else:
            missing = sorted({v for s in slots for v in s.missing})
            lines.append(Line(needed.get(role, WARN), role, f"no usable slot; set: {', '.join(missing) or 'see glide doctor'}"))
    return lines


def check_key_names(config, env: Mapping[str, str]) -> list[Line]:
    """Which key variables the configured providers name, and whether each is set. Names only."""
    names = sorted({spec.api_key_env for spec in config.providers.values() if spec.api_key_env})
    state = names_set(names, env)
    set_names = [n for n in names if state[n]]
    unset = [n for n in names if not state[n]]
    return [
        Line(OK if set_names else FAIL, "keys", f"set: {', '.join(set_names) or 'none'}"),
        Line(INFO, "keys", f"not set: {', '.join(unset) or 'none'} (their slots are skipped)"),
        Line(
            INFO,
            "keys",
            "scene a forced fallback: run `OPENROUTER_API_KEY=invalid-demo-key` (or the first llm.fast slot's variable) on the one "
            "demo command only; an unset key only skips a slot at load, a wrong one makes a visible `fallback:` line",
        ),
    ]


def check_speech(config, extras_missing: Mapping[str, list[str]]) -> list[Line]:
    s = config.voice  # the [speech] table (glide.speech.settings), as `glide doctor` reads it
    lines = [
        Line(INFO, "speech", f"headset = {s.headset}, echo_canceller = {s.echo_canceller!r}, vad = {s.vad!r}"),
        Line(
            INFO,
            "speech",
            f"voice task confirmation: confirm_tasks = {s.confirm_tasks}, phrase = {s.confirm_phrase!r}, window {s.confirm_timeout_s:g} s",
        ),
    ]
    if s.headset:
        lines.append(Line(OK, "speech", "headset mode: talk over Glide with a headset on (no echo reaches the microphone)"))
    elif s.echo_canceller == "none":
        lines.append(
            Line(WARN, "speech", "speaker mode with echo_canceller = none is half duplex: talking over Glide does nothing")
        )
    elif extras_missing.get("aec"):
        lines.append(
            Line(
                WARN,
                "speech",
                "speaker mode without the aec extra: `auto` falls back to the numpy filter or half duplex, and says which",
            )
        )
    else:
        lines.append(Line(OK, "speech", "speaker mode with WebRTC echo cancellation available"))
    return lines


def check_browser(config, env: Mapping[str, str], port_default: int) -> list[Line]:
    """The [browser] settings as Glide resolves them. No connection is made."""
    from glide.computer import browser_settings
    from glide.features import table

    try:
        settings = browser_settings.resolve(table(config, "browser"), env)
    except ValueError as error:
        return [Line(FAIL, "browser", f"settings refused: {' '.join(str(error).split())[:300]}")]
    lines = [
        Line(
            INFO,
            "browser",
            f"provider {settings.provider} (from {settings.source}), target {settings.target or 'the single open tab'}",
        )
    ]
    if settings.provider in browser_settings.ENDPOINT_PROVIDERS:
        endpoint = settings.endpoints.get(settings.provider, "")
        port = port_of(endpoint) or port_default
        lines.append(Line(OK, "browser", f"endpoint is loopback, port {port} (not contacted by this script)"))
    else:
        port = port_default
        lines.append(
            Line(
                WARN,
                "browser",
                "scenes c and d need provider = cdp (the only one ever qualified live): set [browser] in glide.toml",
            )
        )
    if not settings.search_url:
        lines.append(Line(WARN, "browser", "no search_url: a research task must name its sites, or set [browser] search_url"))
    lines.append(
        Line(
            INFO, "browser", f"calls budget for research: {env.get('GLIDE_RESEARCH_CALLS') or 'default (24) or [research] calls'}"
        )
    )
    app_found = [name for name, path in BROWSER_APPS.items() if Path(path).exists()]
    lines.append(
        Line(
            OK if app_found else WARN,
            "browser",
            f"installed: {', '.join(app_found) or 'neither Chrome nor Brave in /Applications'}",
        )
    )
    app = app_found[0] if app_found else "chrome"
    lines.append(Line(INFO, "browser", f"TAKES OVER THE MACHINE, run it yourself: {chrome_command(port, app)}"))
    lines.append(Line(INFO, "browser", f"then check, yourself: {verify_command(port)}   (expect JSON with a Browser field)"))
    return lines


def check_engine() -> list[Line]:
    return [
        Line(
            INFO,
            "engine",
            "there is no engine setting in glide.toml, ask, chat, voice or the app on this branch: the structured engine is "
            "only `glide computer GOAL --engine structured`; ask/chat/listen/voice tasks use the legacy loop",
        )
    ]


def check_files(root: Path = ROOT) -> list[Line]:
    lines = []
    for rel in ("glide.toml", ".env"):
        path = root / rel
        lines.append(
            Line(
                OK if path.is_file() else INFO,
                f"file:{rel}",
                "present" if path.is_file() else "not present in the repository folder",
            )
        )
    swift_app = root / "app"
    lines.append(
        Line(
            INFO,
            "file:app/",
            "present"
            if swift_app.is_dir()
            else "the SwiftUI app is not on this branch (it is on consolidation/app-swiftui): scene f falls back",
        )
    )
    return lines


def run(
    config_path: str | None,
    env_file: Path,
    port: int,
    environ: Mapping[str, str] | None = None,
    root: Path = ROOT,
) -> list[Line]:
    environ = os.environ if environ is None else environ
    dotenv_text = env_file.read_text(encoding="utf-8", errors="replace") if env_file.is_file() else None
    env = effective_env(environ, dotenv_text)
    lines = check_python() + check_files(root)
    lacking = extras_report()
    lines += check_extras() + check_tools()
    config, loaded = load_configuration(config_path, env)
    lines += loaded
    if config is not None:
        try:
            lines += check_key_names(config, env) + check_chains(config) + check_speech(config, lacking)
            lines += check_browser(config, env, port)
        finally:
            config.close()
    lines += check_engine()
    return lines


def render(lines: list[Line]) -> str:
    return "\n".join(f"[{line.status:>4}] {line.topic:<14} {line.text}" for line in lines)


def blocking(lines: list[Line]) -> bool:
    return any(line.status == FAIL for line in lines)


def main(argv: list[str] | None = None, environ: Mapping[str, str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Offline preflight for the Glide demo. Reads only; connects to nothing.")
    parser.add_argument("--config", default=None, help="the glide.toml to check (default: as `glide doctor` finds it)")
    parser.add_argument(
        "--env-file", type=Path, default=Path(".env"), help="a .env whose variable NAMES are read (default: ./.env)"
    )
    parser.add_argument("--port", type=int, default=9222, help="loopback port to print the browser command for (default 9222)")
    args = parser.parse_args(argv)
    logging.disable(logging.WARNING)  # the loader logs each skipped slot; this report says it once, with the variable name
    if not 1024 <= args.port <= 65535:
        parser.error("--port must be between 1024 and 65535")
    lines = run(args.config, args.env_file, args.port, environ)
    print(render(lines))
    print()
    print(
        "Nothing was started or contacted. The manual steps (permissions, headset, notifications) are in docs/DEMO_PREFLIGHT.md."
    )
    print(
        "BLOCKED: fix the [FAIL] lines first."
        if blocking(lines)
        else "No blocking problem found offline. This is not a live check."
    )
    return 1 if blocking(lines) else 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    sys.exit(main())

"""`glide setup --text`: the same decisions as plain prompts. Keys typed here are read without echo, held in this
process only, and never written."""

from __future__ import annotations

import getpass
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

from glide.providers.config import PRESETS, GlideConfig

from . import model


def _yes(ask: Callable[[str], str], question: str, default: bool = False) -> bool:
    answer = ask(f"{question} [{'Y/n' if default else 'y/N'}] ").strip().lower()
    return default if not answer else answer.startswith("y")


def run_text(
    path: Path,
    *,
    ask: Callable[[str], str] = input,
    secret: Callable[[str], str] = getpass.getpass,
    out: Callable[[str], None] = print,
    runner: Callable[..., object] = subprocess.call,
    env: dict | None = None,
) -> int:
    import os

    environ = os.environ if env is None else env
    pasted: dict[str, str] = {}
    out("Glide setup. Nothing is written until the end, and no key is ever written.")
    features = {
        "computer": _yes(ask, "Let Glide control the computer (click and type, with confirmation)?"),
        "webhooks": _yes(ask, "Turn on webhooks?"),
        "memory": _yes(ask, "Turn on memory (saves what you say or type, on this machine)?"),
    }
    names = [n for n, s in PRESETS.items() if s.api_key_env]
    out("One key is enough. Providers: " + ", ".join(names))
    choice = ask("Provider to use for everything it can do (blank to keep the built-in chains): ").strip()
    roles: dict = {}
    if choice:
        try:
            roles = model.preset(choice)
        except model.SetupError as e:
            out(str(e))
            return 2
    state = {"features": features, "roles": roles}
    used = sorted({s["provider"] for chain in roles.values() for s in chain if s["provider"] in PRESETS} or {choice} - {""})
    for name in used:
        spec = PRESETS[name]
        if not spec.api_key_env:
            continue
        link = model.KEY_LINKS.get(name, {}).get("url", "")
        out(f"{name}: Glide reads {spec.api_key_env} (status: {model.key_status(name, pasted, environ)})")
        if link:
            out(f"  where to get a key (check the provider's site): {link}")
        out(f"  {model.export_line(name)}")
        value = secret("  paste a key to hold in memory for this session, or press Enter to skip: ")
        if value:
            if model.KEY_TEXT.fullmatch(value):
                pasted[spec.api_key_env] = value
            else:
                out("  that does not look like a key; skipped")
    try:
        text = model.render_toml(state)
    except model.SetupError as e:
        out(str(e))
        return 2
    out("\nThis is the glide.toml that will be written:\n" + text)
    if not _yes(ask, f"Write {path}" + (" (the old file is backed up first)?" if path.exists() else "?"), True):
        out("Nothing written.")
        return 0
    backup = model.write_config(path, text)
    out(f"Wrote {path}" + (f"; previous file saved as {backup}" if backup else ""))
    child = {**environ, **pasted}
    if roles and pasted and _yes(ask, "Test the first provider now? It sends one tiny request and spends a few tokens."):
        from glide.providers import doctor

        config = GlideConfig.from_toml(text, env=child, source=str(path))
        try:
            role, slot = next((r, c[0]) for r, c in roles.items() if c[0]["provider"] in PRESETS)
            info = next(i for i in config.slots(role) if i.provider == slot["provider"])
            row = doctor._probe(config, role, info, 20.0, time.monotonic)
            out(f"{row.status}: {config.scrub(row.detail)}")
        finally:
            config.close()
    mode = ask("Start Glide now? ask / chat / voice / ui (blank to stop): ").strip().lower()
    if mode in model.MODES:
        act = features["computer"] and _yes(ask, "Computer control is on. Let Glide click and type?")
        request = ask("Your request: ") if mode == "ask" else ""
        argv = model.launch_argv(mode, path, act=act, text=request)
        out("Equivalent command: " + model.equivalent_command(mode, path, act=act))
        runner(argv, env=child)
    return 0

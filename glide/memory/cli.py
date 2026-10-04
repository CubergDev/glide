"""Explicit local memory administration (`glide memory ...`). Never invokes a model, browser or device.

Memory is opt-in. `status` always works and reports whether it is on; every other command refuses until
the user turns memory on ([memory] enabled = true in glide.toml, or GLIDE_MEMORY=1).
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TextIO

from .catalog import Catalog
from .contracts import Model, Scope
from .harness import Harness
from .service import MemoryService
from .settings import MemorySettings, SettingsError

OFF = "memory is off: set `enabled = true` under [memory] in glide.toml, or GLIDE_MEMORY=1, to turn it on"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="glide memory", description=__doc__)
    parser.add_argument("--config", type=Path, help="glide.toml to read ([memory] table); default: the usual search")
    parser.add_argument("--user", default="local", help="user id (an application boundary, not authentication)")
    parser.add_argument("--project", default="default")
    parser.add_argument("--session", default="cli")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status", help="show whether memory is on and where it lives; creates nothing")
    commands.add_parser("init")
    remember = commands.add_parser("remember")
    remember.add_argument("key")
    remember.add_argument("text")
    remember.add_argument("--level", choices=("user", "project", "session"), default="project")
    remember.add_argument("--ttl", type=float)
    commands.add_parser("recall")
    commands.add_parser("events")
    forget = commands.add_parser("forget")
    forget.add_argument("id")
    proposal = commands.add_parser("propose")
    proposal.add_argument("target", help="prompt:NAME or skill:ID")
    proposal.add_argument("text")
    proposal.add_argument("--evidence", nargs="+", required=True)
    commands.add_parser("proposals")
    for name in ("apply", "rollback"):
        operation = commands.add_parser(name)
        operation.add_argument("id")
        operation.add_argument("--revision", type=int, required=True)
    preview = commands.add_parser("plan")
    preview.add_argument("goal")
    preview.add_argument("--catalog", type=Path, required=True)
    preview.add_argument("--plugin", action="append", default=[])
    preview.add_argument("--model", required=True, help="inventory label only; no provider is contacted")
    preview.add_argument("--window", type=int, required=True)
    preview.add_argument("--stage", choices=("write", "answer", "handoff", "refine"), default="handoff")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    home: Path | None = None,
    out: TextIO | None = None,
    err: TextIO | None = None,
) -> int:
    """Run one command; returns the exit status (0 ok, 2 refused or invalid). Argument errors exit via argparse."""
    out, err = out or sys.stdout, err or sys.stderr
    args = _parser().parse_args(argv)
    try:
        settings = MemorySettings.load(environ, config=args.config, home=home)
        if args.command == "status":
            database = settings.database_path(environ, home)
            result = {
                "enabled": settings.enabled,
                "auto_capture": settings.auto_capture,
                "database": str(database),
                "exists": database.exists(),
            }
        else:
            scope = Scope(args.user, args.project, args.session)
            service = MemoryService.open(settings, environ=environ, home=home)
            if service is None:
                err.write(f"glide memory: {OFF}\n")
                return 2
            with service:
                result = _run(service, scope, args)
        out.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        return 0
    except (ValueError, OSError, sqlite3.Error) as error:  # SettingsError is a ValueError
        err.write(f"glide memory: {error}\n")
        return 2


def _plan(store, scope: Scope, args: argparse.Namespace) -> dict:
    catalog = Catalog(args.catalog, enabled_plugins=frozenset(args.plugin))
    harness = Harness(store, catalog, models=(Model(args.model, frozenset({args.stage}), args.window),))
    bundle = harness.prepare(scope, args.goal, stage=args.stage)
    return {"context": bundle.context, **(harness.command(scope, "/context") or {})}


def _run(service: MemoryService, scope: Scope, args: argparse.Namespace):
    store = service.store
    commands = {
        "init": lambda: {"database": str(service.database), "revision": store.revision(scope)},
        "remember": lambda: store.remember(scope, args.key, args.text, level=args.level, ttl_seconds=args.ttl),
        "recall": lambda: store.memories(scope),
        "events": lambda: store.events(scope),
        "forget": lambda: {"forgotten": store.forget(scope, args.id)},
        "propose": lambda: store.propose(scope, args.target, args.text, args.evidence),
        "proposals": lambda: store.proposals(scope),
        "apply": lambda: store.apply(scope, args.id, expected_revision=args.revision),
        "rollback": lambda: store.rollback(scope, args.id, expected_revision=args.revision),
        "plan": lambda: _plan(store, scope, args),
    }
    return commands[args.command]()


__all__ = ["OFF", "SettingsError", "main"]

"""python -m glide.files plan | preview | apply --approve <plan_hash> | undo."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import engine, planner
from .model import Approval, Plan
from .safety import Refused

INTENTS = {"type": planner.ORGANIZE_BY_TYPE, "named": planner.NAMED_FOLDERS}


def _json_file(path: str) -> dict:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise Refused(f"cannot read {path} as JSON") from error
    if not isinstance(data, dict):
        raise Refused(f"{path} must hold a JSON object")
    return data


def _load_plan(path: str) -> Plan:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as error:
        raise Refused(f"cannot read {path}") from error
    return Plan.from_json(text)


def _report(report) -> int:
    print(report.status)
    for action in report.actions:
        print(f"  {action.status}: {action.source} -> {action.destination}" + (f" ({action.reason})" if action.reason else ""))
    if report.manifest:
        print(f"manifest: {report.manifest}")
    return 0 if report.status == "ok" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m glide.files")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan", help="decide what would move; moves nothing")
    p.add_argument("root")
    p.add_argument("--intent", choices=sorted(INTENTS), required=True)
    p.add_argument("--categories", help="JSON file: folder -> extensions")
    p.add_argument("--folders", help="JSON file: folder -> keywords (for --intent named)")
    p.add_argument("--out", help="write the plan here instead of stdout")
    v = sub.add_parser("preview", help="dry run: describe a plan")
    v.add_argument("plan_file")
    a = sub.add_parser("apply", help="run a plan; needs the hash preview printed")
    a.add_argument("plan_file")
    a.add_argument("--approve", required=True, metavar="PLAN_HASH")
    a.add_argument("--manifest-dir", default=str(Path.home() / ".glide" / "file-runs"))
    u = sub.add_parser("undo", help="reverse a run from its manifest")
    u.add_argument("manifest")
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            made = planner.plan(
                args.root,
                INTENTS[args.intent],
                categories=_json_file(args.categories) if args.categories else None,
                folders=_json_file(args.folders) if args.folders else None,
            )
            if args.out:
                Path(args.out).write_text(made.to_json(), encoding="utf-8")
                print(f"plan {made.plan_hash}: {len(made.moves)} move(s) written to {args.out}")
            else:
                print(made.to_json())
            return 0
        if args.command == "preview":
            print(planner.preview(_load_plan(args.plan_file)))
            return 0
        if args.command == "apply":
            return _report(engine.execute(_load_plan(args.plan_file), Approval(args.approve), manifest_dir=args.manifest_dir))
        return _report(engine.undo(args.manifest))
    except (Refused, OSError) as error:
        print(f"refused: {error}", file=sys.stderr)
        return 1

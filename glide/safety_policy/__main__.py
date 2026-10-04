"""`python -m glide.safety_policy --kind click --role button --label "Place order"` prints the verdict as JSON.

Only the decision, the reason class and the list phrase are printed, never the label, value or text given.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path

from . import Action, decide, load_lexicon


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m glide.safety_policy", description=__doc__)
    parser.add_argument("--kind", required=True)
    for name in ("label", "role", "hint", "value", "field-name", "field-type", "text", "key", "task-text"):
        parser.add_argument(f"--{name}", default="")
    parser.add_argument("--supplied", action="append", default=[], help="a configured value that counts as supplied")
    parser.add_argument("--secret", action="store_true")
    parser.add_argument("--send-intent", action="store_true")
    parser.add_argument("--lexicon", type=Path, help="a TOML file of extra words, in the shape of default.toml")
    args = parser.parse_args(argv)
    extra = tomllib.loads(args.lexicon.read_text(encoding="utf-8")) if args.lexicon else None
    action = Action(
        args.kind,
        label=args.label,
        role=args.role,
        hint=args.hint,
        value=args.value,
        field_name=args.field_name,
        field_type=args.field_type,
        text=args.text,
        key=args.key,
        task_text=args.task_text,
        supplied=tuple(args.supplied),
        secret=args.secret,
        send_intent=args.send_intent,
    )
    json.dump(decide(action, load_lexicon(extra)).as_dict(), sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

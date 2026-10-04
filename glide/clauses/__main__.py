"""`python -m glide.clauses "open notes and write hello then close it"` prints the plan as JSON. Reads nothing else."""

from __future__ import annotations

import dataclasses
import json
import sys

from glide.clauses.split import split_clauses


def main(argv: list[str]) -> int:
    if not argv:
        print('usage: python -m glide.clauses "<utterance>"', file=sys.stderr)
        return 2
    plan = split_clauses(" ".join(argv))
    print(json.dumps([dataclasses.asdict(s) for s in plan.steps], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

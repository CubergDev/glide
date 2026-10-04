"""`python -m glide.direct [--config glide.toml] "request"`: print what a request compiles to. Opens nothing."""

from __future__ import annotations

import argparse
import sys

from . import DirectConfigError, load_settings, load_settings_file, resolve


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m glide.direct", description=__doc__)
    parser.add_argument("text", help="the request, for example 'what is 15% of 240' or 'open youtube'")
    parser.add_argument("--config", help="a glide.toml whose [direct] table overrides the defaults")
    args = parser.parse_args(argv)
    try:
        settings = load_settings_file(args.config) if args.config else load_settings()
    except DirectConfigError as error:
        print(f"config error: {error}", file=sys.stderr)
        return 2
    got = resolve(args.text, settings)
    if got is None:
        print("not a direct request", file=sys.stderr)
        return 1
    print(f"answer: {got.answer}" if got.kind == "answer" else f"{got.kind}: {got.url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

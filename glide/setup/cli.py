"""`glide setup [--text] [--no-open] [--config PATH]`."""

from __future__ import annotations

import argparse
import contextlib
import threading
import webbrowser
from collections.abc import Callable, Sequence
from pathlib import Path

from .server import SetupServer


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="glide setup", description="First-run wizard: writes glide.toml, never a key.")
    p.add_argument("--text", action="store_true", help="plain prompts in this terminal instead of a local page")
    p.add_argument("--no-open", action="store_true", help="print the page address instead of opening a browser")
    p.add_argument("--config", help="where to write glide.toml (default: ./glide.toml)")
    return p


def main(argv: Sequence[str] | None = None, *, opener: Callable[[str], object] = webbrowser.open) -> int:
    args = build_parser().parse_args(argv)
    path = Path(args.config or "glide.toml").expanduser()
    if args.text:
        from .text import run_text

        return run_text(path)
    server = SetupServer(path)
    print(f"Glide setup is at {server.url}")
    print("It listens on this computer only and the link works once. Press Ctrl+C to stop.")
    if not args.no_open:
        threading.Thread(target=opener, args=(server.url,), daemon=True).start()
    with contextlib.suppress(KeyboardInterrupt):
        server.serve()
    if path.is_file():
        print(f"Next time use the control panel, which edits {path} and more: glide panel")
    return 0

"""`glide panel [--no-open] [--config PATH] [--idle-minutes N] [--runs DIR]`."""

from __future__ import annotations

import argparse
import contextlib
import os
import threading
import webbrowser
from collections.abc import Callable, Sequence
from pathlib import Path

from . import model
from .server import PanelServer


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="glide panel", description="A local control panel for glide.toml, chat, files and status.")
    p.add_argument("--no-open", action="store_true", help="print the page address instead of opening a browser")
    p.add_argument("--config", help="the glide.toml to edit (default: ./glide.toml, else ~/.config/glide/glide.toml)")
    p.add_argument("--idle-minutes", type=float, default=30.0, help="stop after this long with no request (default 30)")
    p.add_argument("--runs", default="runs", help="where computer tasks write their run folders (default ./runs)")
    return p


def main(argv: Sequence[str] | None = None, *, opener: Callable[[str], object] = webbrowser.open) -> int:
    args = build_parser().parse_args(argv)
    if not 1 <= args.idle_minutes <= 24 * 60:
        print("glide panel: --idle-minutes must be from 1 to 1440")
        return 2
    path, scope = model.locate(args.config, cwd=Path.cwd(), home=Path(os.path.expanduser("~")))
    server = PanelServer(path, scope=scope, idle_s=args.idle_minutes * 60, runs_dir=Path(args.runs).expanduser())
    print(f"Glide panel is at {server.url}")
    print("It listens on this computer only and the link works once. Press Ctrl+C to stop.")
    print(f"Editing {path} ({scope}).")
    if not args.no_open:
        threading.Thread(target=opener, args=(server.url,), daemon=True).start()
    with contextlib.suppress(KeyboardInterrupt):
        server.serve()
    with contextlib.suppress(Exception):
        server.chat.stop()
    return 0

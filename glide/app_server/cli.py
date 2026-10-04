"""`glide app-server`: serve the SwiftUI app (and anything else that speaks app/PROTOCOL.md) over a local socket.

    glide app-server [--socket PATH] [--record-content]

The socket is a Unix domain socket in a private directory: `--socket`, else `$GLIDE_SOCKET`, else
`~/Library/Application Support/Glide/glide.sock`. Nothing listens on a network port. The server starts nothing that touches the
screen or the microphone: a computer task starts only when the app approves it, and the microphone opens only when the app
turns hands-free on. Settings changed from the app last until this process ends and never rewrite glide.toml.

`--record-content` opts in to detailed recording for this run (utterances, typed text and what was read off the screen may then
be stored and sent to the app). Without it, none is: transcripts carry only a length.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import signal
import sys
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .bridge import AppBridge, core_version
from .runtime import real_voice
from .server import AppServer
from .transport import Listener, SocketRefused, resolve_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="glide app-server",
        description="Serve the Glide app over a local Unix socket (never a network port).",
    )
    parser.add_argument(
        "--socket", help="the socket path (default: $GLIDE_SOCKET, else ~/Library/Application Support/Glide/glide.sock)"
    )
    parser.add_argument(
        "--record-content",
        action="store_true",
        help="opt in to detailed recording for this run: utterances, typed text and screen content may be stored and sent to the app",
    )
    return parser


def main(argv: Sequence[str], config: Any) -> int:
    args = build_parser().parse_args(list(argv))
    path = resolve_path(args.socket, os.environ)
    try:
        return serve(config, path, record_content=args.record_content)
    except (SocketRefused, OSError) as exc:
        print(f"glide app-server: {exc}", file=sys.stderr)
        return 2


def serve(config: Any, path: Path, *, record_content: bool = False) -> int:
    """Run until interrupted. This binds the socket and builds the real assistant, so tests replace it."""
    bridge = AppBridge(config, record_content=record_content, voice_factory=real_voice)
    server = AppServer(bridge, core_version=core_version(), listener=Listener(path))
    bridge.bind(server)
    done = threading.Event()
    previous: dict[int, Any] = {}
    if threading.current_thread() is threading.main_thread():
        for number in (signal.SIGINT, signal.SIGTERM):
            previous[number] = signal.signal(number, lambda *_: done.set())
    try:
        server.start()
        print(f"glide app-server: listening on {path} (Ctrl-C to stop)", file=sys.stderr, flush=True)
        if record_content:
            print("glide app-server: detailed recording of content is ON for this run", file=sys.stderr, flush=True)
        done.wait()
    finally:
        server.stop()
        bridge.close()
        for number, handler in previous.items():
            with contextlib.suppress(Exception):
                signal.signal(number, handler)
    return 0

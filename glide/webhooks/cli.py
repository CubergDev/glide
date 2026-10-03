"""Explicit server launch. Nothing starts on import; loopback bind by default; no desktop worker inside HTTP."""

from __future__ import annotations

import argparse
import ipaddress
import os
from pathlib import Path

from .settings import load_settings


def serve(settings, host: str, port: int) -> None:
    import uvicorn

    from .app import create_app

    uvicorn.run(
        create_app(settings), host=host, port=port, workers=1, proxy_headers=False, limit_concurrency=32, timeout_keep_alive=5
    )


def check(settings) -> None:
    """Raise ValueError/ImportError if the configuration cannot be served. Starts nothing, contacts nothing."""
    from .app import create_app

    create_app(settings, check_only=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="glide webhooks serve", description="Receive authenticated webhooks and queue agent requests"
    )
    parser.add_argument("--config", type=Path, default=Path(os.environ.get("GLIDE_WEBHOOK_CONFIG", "webhooks.json")))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--behind-proxy", action="store_true", help="allow a public bind behind a TLS-terminating reverse proxy")
    parser.add_argument("--check-config", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    try:
        local = args.host == "localhost" or ipaddress.ip_address(args.host).is_loopback
    except ValueError:
        local = False
    if not local and not args.behind_proxy:
        parser.error("public binds require --behind-proxy and a TLS reverse proxy")
    try:
        settings = load_settings(args.config)
        check(settings)
        if args.check_config:
            print("Webhook configuration is valid. No server or agent started.")
            return 0
        serve(settings, args.host, args.port)
    except (ValueError, ImportError, OSError):
        parser.exit(2, "Could not configure the webhook service. See glide/webhooks/README.md.\n")
    return 0

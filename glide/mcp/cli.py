"""`glide mcp ...`: serve Glide over MCP (stdio) and show the MCP settings. Never drives the machine.

`serve` speaks newline-delimited JSON-RPC on stdin/stdout, so nothing else may print to stdout. With the
default settings it offers no tools at all. Memory tools appear only when the user set `server_memory` under
[mcp] AND turned memory on (see glide.memory.settings).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TextIO

from glide.memory.contracts import Scope
from glide.memory.service import MemoryService
from glide.memory.settings import MemorySettings

from .config import McpSettings
from .server import GlideMCPServer, memory_tools, serve_stream, text_stream_writer


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="glide mcp", description=__doc__)
    parser.add_argument("--config", type=Path, help="glide.toml to read; default: the usual search")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status", help="show the MCP settings (server names only; nothing is started)")
    serve = commands.add_parser("serve", help="serve Glide over stdio until stdin closes")
    serve.add_argument("--user", default="local")
    serve.add_argument("--project", default="default")
    serve.add_argument("--session", default="mcp")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    home: Path | None = None,
    stdin: TextIO | None = None,
    out: TextIO | None = None,
    err: TextIO | None = None,
) -> int:
    stdin, out, err = stdin or sys.stdin, out or sys.stdout, err or sys.stderr
    args = _parser().parse_args(argv)
    try:
        settings = McpSettings.load(environ, config=args.config, home=home)
        if args.command == "status":
            result = {
                "server_memory": settings.server_memory,
                "client_servers": [spec.name for spec in settings.servers],
            }
            out.write(json.dumps(result, indent=2) + "\n")
            return 0
        service = None
        tools = []
        if settings.server_memory != "off":
            service = MemoryService.open(MemorySettings.load(environ, config=args.config, home=home), environ=environ, home=home)
            if service is None:
                err.write('glide mcp: server_memory is set but memory is off; turn memory on or set server_memory = "off"\n')
                return 2
            tools = memory_tools(
                service.store, Scope(args.user, args.project, args.session), write=settings.server_memory == "write"
            )
        server = GlideMCPServer(tools)

        async def readline() -> str:
            return await asyncio.to_thread(stdin.readline)

        try:
            asyncio.run(serve_stream(server, readline, text_stream_writer(out)))
        finally:
            if service is not None:
                service.close()
        return 0
    except (ValueError, OSError) as error:
        err.write(f"glide mcp: {error}\n")
        return 2

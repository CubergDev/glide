"""Open a real MCP session over stdio with the optional SDK (`pip install 'glide[mcp]'`).

This is the only code in the package that starts a process. It is never exercised by the offline tests
(tests/conftest.py refuses `subprocess.Popen`; tests/guards_memory-mcp.py also refuses the SDK's transports), so
it is a thin shim around the SDK and needs a live check by the user. Starting a server launches a local command,
so the caller must pass `approve`, a callback that shows the user that exact command and returns True only after
a yes (AGENTS.md: ask first, for that exact command). Without a yes nothing is started.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import timedelta

from glide.providers.errors import ProviderError

from .client import SessionClient, to_provider_error
from .config import McpServerSpec
from .session import session_request


@asynccontextmanager
async def open_stdio_client(
    spec: McpServerSpec,
    *,
    approve: Callable[[McpServerSpec], bool],
    environ: Mapping[str, str] | None = None,
) -> AsyncIterator[SessionClient]:
    provider = f"mcp:{spec.name}"
    if approve(spec) is not True:
        raise ProviderError(f"{provider}: the user did not approve starting this server", kind="unsupported", provider=provider)
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError:
        raise ProviderError(
            f"{provider}: the MCP SDK is not installed (install glide's mcp extra)", kind="unsupported", provider=provider
        ) from None
    source = os.environ if environ is None else environ
    env = {name: source[name] for name in spec.env_names if name in source}  # values go to the child only
    parameters = StdioServerParameters(command=spec.command, args=list(spec.args), env=env or None)
    body_failed = False
    try:
        async with AsyncExitStack() as stack:
            try:
                async with asyncio.timeout(spec.timeout_s):  # a server that never answers initialize must not hang setup
                    read, write = await stack.enter_async_context(stdio_client(parameters))
                    session = await stack.enter_async_context(ClientSession(read, write))
                    await session.initialize()
            except Exception as error:
                raise to_provider_error(error, server=spec.name, action="connect") from None
            try:
                yield SessionClient(
                    spec.name,
                    session_request(session, read_timeout_seconds=timedelta(seconds=spec.timeout_s)),
                    timeout_s=spec.timeout_s,
                )
            except BaseException:
                body_failed = True
                raise
    except ProviderError:
        raise
    except Exception as error:
        if body_failed:
            raise  # raised by the caller's own block: not ours to relabel
        raise to_provider_error(error, server=spec.name, action="close") from None  # SDK teardown after a clean body

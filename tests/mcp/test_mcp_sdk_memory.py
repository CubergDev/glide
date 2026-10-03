"""Real SDK client and an SDK server joined by in-memory streams: no socket, no subprocess, no stdio."""

import importlib.util

import pytest
from mcp_guard_loader import no_real_memory_or_mcp, sync  # noqa: F401  (autouse guard, async runner)

from glide.mcp import MCPBridge, SessionClient, session_request

pytestmark = [pytest.mark.anyio, pytest.mark.skipif(importlib.util.find_spec("mcp") is None, reason="needs the optional MCP SDK")]


def make_server():
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("in-process")

    @server.tool()
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    return server


@sync
async def test_sdk_session_through_session_request_and_client():
    try:
        from mcp.shared.memory import create_connected_server_and_client_session as connect
    except ImportError:
        pytest.skip("this SDK version has no in-memory session helper")
    async with connect(make_server()._mcp_server) as session:
        client = SessionClient("sdk", session_request(session), timeout_s=10)
        tools = await client.list_tools()
        assert [t.name for t in tools] == ["add"]
        result = await client.call_tool("add", {"a": 2, "b": 3})
        assert result["isError"] is False and result["content"][0]["text"] == "5"

        bridge = MCPBridge(
            "sdk",
            session_request(session),
            asynchronous=True,
            capabilities={"tools": {}},
            permissions={"add": frozenset()},
            keywords={"add": ("add",)},
        )
        bound = await bridge.arefresh()
        assert (await bound[0].invoke({"a": 1, "b": 1}))["content"][0]["text"] == "2"

"""SessionClient: discovery and calls over a fake `request`, ProviderError only, no replay of a call."""

import asyncio

import httpx
import pytest
from mcp_guard_loader import no_real_memory_or_mcp, sync  # noqa: F401  (autouse guard, async runner)

from glide.mcp import MCPCallError, SessionClient
from glide.mcp.client import to_provider_error
from glide.providers.errors import ProviderError

TOOL = {"name": "lookup", "description": "d", "inputSchema": {"type": "object"}}


class RpcError(Exception):
    def __init__(self, code):
        self.error = {"code": code, "message": "SECRET-SERVER-TEXT"}
        super().__init__("SECRET-SERVER-TEXT")


def client(request, **options):
    return SessionClient("files", request, **options)


@sync
async def test_lists_tools_across_pages():
    pages = {None: {"tools": [TOOL], "nextCursor": "p2"}, "p2": {"tools": [{**TOOL, "name": "other"}]}}
    seen = []

    async def request(method, params):
        seen.append((method, dict(params)))
        return pages[params.get("cursor")]

    tools = await client(request).list_tools()
    assert [t.name for t in tools] == ["lookup", "other"]
    assert seen == [("tools/list", {}), ("tools/list", {"cursor": "p2"})]


@pytest.mark.parametrize(
    "pages",
    [
        {"tools": [TOOL, TOOL]},  # duplicate name
        {"tools": "nope"},
        {"tools": [{"name": "x", "inputSchema": {"type": "string"}}]},
        {"tools": [], "nextCursor": ""},
    ],
)
@sync
async def test_unusable_inventory_is_a_content_error(pages):
    async def request(method, params):
        return pages

    with pytest.raises(ProviderError) as caught:
        await client(request).list_tools()
    assert caught.value.kind == "content" and caught.value.provider == "mcp:files"


@sync
async def test_repeated_cursor_is_refused():
    async def request(method, params):
        return {"tools": [], "nextCursor": "same"}

    with pytest.raises(ProviderError, match="unusable"):
        await client(request).list_tools()


@sync
async def test_call_returns_normalized_result_and_keeps_tool_errors_as_feedback():
    async def request(method, params):
        assert method == "tools/call" and params == {"name": "lookup", "arguments": {"q": 1}}
        return {"content": [{"type": "text", "text": "no such file"}], "isError": True}

    result = await client(request).call_tool("lookup", {"q": 1})
    assert result["isError"] is True and result["content"][0]["text"] == "no such file"


@sync
async def test_call_is_sent_exactly_once_even_when_it_fails():
    calls = []

    async def request(method, params):
        calls.append(params)
        raise ConnectionResetError("boom")

    with pytest.raises(MCPCallError) as caught:
        await client(request).call_tool("lookup", {})
    assert len(calls) == 1
    assert caught.value.kind == "transport" and caught.value.outcome_unknown is True


def http_status(code):
    return httpx.HTTPStatusError("x", request=httpx.Request("GET", "http://h"), response=httpx.Response(code))


# outcome_unknown is False only for an explicit rejection that is made before a server runs anything: a JSON-RPC
# request/params/method/parse error, or an HTTP 400/401/403/404/405/413/415/422. Every other failure of a call that
# was sent (any 5xx, 408, 429, -32603, -32000..-32099, a timeout, a dropped connection, an unknown exception) may
# have come after the tool ran.
@pytest.mark.parametrize(
    ("error", "kind", "unknown"),
    [
        (RpcError(-32602), "bad_request", False),
        (RpcError(-32601), "unsupported", False),
        (RpcError(-32600), "bad_request", False),
        (RpcError(-32700), "bad_request", False),
        (RpcError(-32000), "server", True),
        (RpcError(-32603), "server", True),
        (RpcError(-32099), "server", True),
        (RpcError(408), "timeout", True),
        (RpcError(7), "server", True),
        (TimeoutError(), "timeout", True),
        (httpx.ConnectError("SECRET-URL"), "transport", True),
        (httpx.ReadTimeout("SECRET-URL"), "timeout", True),
        (RuntimeError("SECRET"), "transport", True),
        *[(http_status(code), kind, False) for code, kind in ((400, "bad_request"), (401, "auth"), (403, "auth"), (404, None))],
        *[(http_status(code), None, True) for code in (408, 429, 500, 502, 503, 504)],
    ],
)
@sync
async def test_errors_map_to_provider_error_kinds_without_server_text(error, kind, unknown):
    calls = []

    async def request(method, params):
        calls.append(params)
        raise error

    with pytest.raises(MCPCallError) as caught:
        await client(request).call_tool("lookup", {})
    assert kind is None or caught.value.kind == kind
    assert caught.value.outcome_unknown is unknown
    assert len(calls) == 1  # never retried, whatever the outcome
    assert "SECRET" not in str(caught.value)
    assert ("may or may not have happened" in str(caught.value)) is unknown


@sync
async def test_a_provider_error_raised_while_calling_still_says_whether_the_outcome_is_known():
    async def request(method, params):
        raise ProviderError("mcp:files dropped", kind="stream", provider="mcp:files")

    with pytest.raises(MCPCallError) as caught:
        await client(request).call_tool("lookup", {})
    assert caught.value.kind == "stream" and caught.value.outcome_unknown is True
    assert str(caught.value).startswith("mcp:files dropped")


@sync
async def test_an_unusable_listing_is_not_a_call_and_carries_no_outcome_flag():
    async def request(method, params):
        raise http_status(500)

    with pytest.raises(ProviderError) as caught:
        await client(request).list_tools()
    assert not isinstance(caught.value, MCPCallError) and "may or may not" not in str(caught.value)


@sync
async def test_an_unusable_call_reply_leaves_the_outcome_unknown():
    async def request(method, params):
        return {"content": "not a list"}

    with pytest.raises(MCPCallError) as caught:
        await client(request).call_tool("lookup", {})
    assert caught.value.kind == "content" and caught.value.outcome_unknown is True


@sync
async def test_listing_errors_are_plain_provider_errors():
    async def request(method, params):
        raise RpcError(-32000)

    with pytest.raises(ProviderError) as caught:
        await client(request).list_tools()
    assert not isinstance(caught.value, MCPCallError) and caught.value.kind == "server"


@sync
async def test_timeout_bounds_a_hung_request_and_is_unknown_for_calls():
    async def request(method, params):
        await asyncio.sleep(10)

    with pytest.raises(MCPCallError) as caught:
        await client(request, timeout_s=0.01).call_tool("lookup", {})
    assert caught.value.kind == "timeout" and caught.value.outcome_unknown is True


@sync
async def test_cancellation_propagates_untouched():
    started = asyncio.Event()

    async def request(method, params):
        started.set()
        await asyncio.sleep(10)

    task = asyncio.create_task(client(request).call_tool("lookup", {}))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


@sync
async def test_bad_arguments_never_reach_the_server():
    async def request(method, params):
        raise AssertionError("must not be sent")

    for name, arguments in (
        ("", {}),
        ("x" * 129, {}),
        ("lookup", [1]),
        ("lookup", {"v": float("nan")}),
        ("lookup", {"v": "x" * 70000}),
    ):
        with pytest.raises(ProviderError) as caught:
            await client(request).call_tool(name, arguments)
        assert caught.value.kind == "bad_request"


@sync
async def test_close_runs_once_and_maps_errors():
    closed = []

    async def close():
        closed.append(1)

    c = client(None, close=close)
    await c.aclose()
    await c.aclose()
    assert closed == [1]

    async def broken():
        raise OSError("SECRET")

    with pytest.raises(ProviderError) as caught:
        await client(None, close=broken).aclose()
    assert "SECRET" not in str(caught.value)


def test_a_provider_error_passes_through_unchanged():
    original = ProviderError("x", kind="auth", provider="p")
    assert to_provider_error(original, server="s", action="a") is original

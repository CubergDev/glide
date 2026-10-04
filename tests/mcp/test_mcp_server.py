"""GlideMCPServer in-process (no SDK, no sockets, no stdio): protocol behaviour, approval and memory tools."""

import io
import json

import pytest
from mcp_guard_loader import no_real_memory_or_mcp, sync  # noqa: F401  (autouse guard, async runner)

from glide.mcp import GlideMCPServer, ServerTool, SessionClient, memory_tools, serve_stream
from glide.memory import Scope, Store

SCOPE = Scope("alice", "project", "mcp")


def rpc(method, params=None, id=1):
    message = {"jsonrpc": "2.0", "method": method, "id": id}
    if params is not None:
        message["params"] = params
    return message


async def ready(server):
    reply = await server.handle(rpc("initialize", {"protocolVersion": "2025-11-25"}))
    assert reply["result"]["serverInfo"]["name"] == "glide"
    assert await server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    return reply["result"]


def echo(approval=False):
    return ServerTool(
        "echo",
        "Echo",
        {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"], "additionalProperties": False},
        lambda args: {"echo": args["text"]},
        needs_approval=approval,
    )


@sync
async def test_default_server_offers_nothing():
    server = GlideMCPServer()
    result = await ready(server)
    assert result["capabilities"] == {"tools": {"listChanged": False}}
    assert (await server.handle(rpc("tools/list")))["result"]["tools"] == []


@sync
async def test_requires_initialize_and_negotiates_version():
    server = GlideMCPServer()
    assert (await server.handle(rpc("tools/list")))["error"]["code"] == -32600
    assert (await server.handle(rpc("initialize", {})))["error"]["code"] == -32602
    result = (await server.handle(rpc("initialize", {"protocolVersion": "1999-01-01"})))["result"]
    assert result["protocolVersion"] == "2025-11-25"
    assert (await server.handle(rpc("ping")))["result"] == {}


@pytest.mark.parametrize(
    "message", [[], "x", {"method": "ping"}, {"jsonrpc": "2.0"}, {"jsonrpc": "2.0", "method": "ping", "id": True}]
)
@sync
async def test_malformed_requests(message):
    reply = await GlideMCPServer().handle(message)
    assert reply["error"]["code"] == -32600


@sync
async def test_unknown_method_and_non_object_params():
    server = GlideMCPServer()
    await ready(server)
    assert (await server.handle(rpc("resources/list")))["error"]["code"] == -32601
    assert (await server.handle({"jsonrpc": "2.0", "method": "ping", "id": 2, "params": []}))["error"]["code"] == -32602


@sync
async def test_tool_call_validates_arguments():
    server = GlideMCPServer([echo()])
    await ready(server)
    ok = (await server.handle(rpc("tools/call", {"name": "echo", "arguments": {"text": "hi"}})))["result"]
    assert ok["isError"] is False and ok["structuredContent"] == {"echo": "hi"}
    for arguments in ({}, {"text": 5}, {"text": "x", "extra": 1}, []):
        reply = await server.handle(rpc("tools/call", {"name": "echo", "arguments": arguments}))
        assert reply["error"]["code"] == -32602
    assert (await server.handle(rpc("tools/call", {"name": "nope"})))["error"]["code"] == -32602


@sync
async def test_machine_touching_tools_need_an_exact_true_from_the_approver():
    calls = []
    tool = ServerTool(
        "act", "Act", {"type": "object", "properties": {}}, lambda a: calls.append(a) or {"done": True}, needs_approval=True
    )
    none = GlideMCPServer([tool])
    await ready(none)
    reply = await none.handle(rpc("tools/call", {"name": "act", "arguments": {}}))
    assert reply["result"]["isError"] is True and "no approver" in reply["result"]["content"][0]["text"]
    for verdict in (False, None, "yes", 1):
        server = GlideMCPServer([tool], approve=lambda t, a, v=verdict: v)
        await ready(server)
        reply = await server.handle(rpc("tools/call", {"name": "act", "arguments": {}}))
        assert reply["result"]["isError"] is True

    async def yes(t, a):
        return True

    server = GlideMCPServer([tool], approve=yes)
    await ready(server)
    assert (await server.handle(rpc("tools/call", {"name": "act", "arguments": {}})))["result"]["isError"] is False
    assert calls == [{}]


@sync
async def test_handler_failures_do_not_leak_internals():
    def boom(args):
        raise RuntimeError("/Users/secret/path")

    server = GlideMCPServer([ServerTool("boom", "b", {"type": "object", "properties": {}}, boom)])
    await ready(server)
    reply = (await server.handle(rpc("tools/call", {"name": "boom", "arguments": {}})))["result"]
    assert reply["isError"] is True and "secret" not in json.dumps(reply) and "RuntimeError" in json.dumps(reply)


def test_duplicate_or_non_object_tools_are_refused():
    with pytest.raises(ValueError, match="duplicate"):
        GlideMCPServer([echo(), echo()])
    with pytest.raises(ValueError, match="object"):
        GlideMCPServer([ServerTool("x", "x", {"type": "string"}, lambda a: 1)])


@sync
async def test_memory_tools_are_read_only_by_default_and_scoped_by_the_host(tmp_path):
    with Store(tmp_path / "s.sqlite") as store:
        store.remember(SCOPE, "style", "concise")
        store.remember(Scope("bob", "project", "mcp"), "style", "BOB-PRIVATE")
        server = GlideMCPServer(memory_tools(store, SCOPE))
        await ready(server)
        assert server.tool_names == ("glide.memory.recall",)
        reply = (await server.handle(rpc("tools/call", {"name": "glide.memory.recall", "arguments": {}})))["result"]
        assert "concise" in json.dumps(reply) and "BOB-PRIVATE" not in json.dumps(reply)
        # a client cannot choose another scope
        reply = await server.handle(rpc("tools/call", {"name": "glide.memory.recall", "arguments": {"user": "bob"}}))
        assert reply["error"]["code"] == -32602


@sync
async def test_memory_write_tools_when_allowed_refuse_credentials(tmp_path):
    with Store(tmp_path / "s.sqlite") as store:
        server = GlideMCPServer(memory_tools(store, SCOPE, write=True))
        await ready(server)
        saved = (
            await server.handle(
                rpc("tools/call", {"name": "glide.memory.remember", "arguments": {"key": "k", "text": "likes tea"}})
            )
        )["result"]
        identifier = saved["structuredContent"]["id"]
        assert store.memories(SCOPE)[0]["source"] == "mcp"
        secret = "sk-abcdefghijklmnopqrstuvwxyz0123456789"
        refused = (
            await server.handle(rpc("tools/call", {"name": "glide.memory.remember", "arguments": {"key": "k2", "text": secret}}))
        )["result"]
        assert refused["isError"] is True and secret not in json.dumps(refused)
        gone = (await server.handle(rpc("tools/call", {"name": "glide.memory.forget", "arguments": {"id": identifier}})))[
            "result"
        ]
        assert gone["structuredContent"] == {"forgotten": True} and store.memories(SCOPE) == []


@sync
async def test_stream_round_trip_and_bad_lines(tmp_path):
    server = GlideMCPServer([echo()])
    lines = [
        json.dumps(rpc("initialize", {"protocolVersion": "2025-11-25"})),
        json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        "",
        "{not json",
        json.dumps(rpc("tools/call", {"name": "echo", "arguments": {"text": "hi"}}, id=2)),
        "x" * 1048577,
    ]
    out = io.StringIO()

    pending = list(lines)

    async def readline():
        return (pending.pop(0) + "\n") if pending else ""

    await serve_stream(server, readline, out.write)
    replies = [json.loads(line) for line in out.getvalue().splitlines()]
    assert [r.get("id") for r in replies] == [1, None, 2, None]
    assert replies[1]["error"]["code"] == -32700 and replies[3]["error"]["message"] == "message too large"
    assert replies[2]["result"]["structuredContent"] == {"echo": "hi"}


@sync
async def test_server_and_client_talk_in_process():
    """The client interface against the server, wired by a function call: no transport at all."""
    server = GlideMCPServer([echo()])
    await server.handle(rpc("initialize", {"protocolVersion": "2025-11-25"}))

    async def request(method, params):
        reply = await server.handle(rpc(method, params))
        if "error" in reply:
            raise type("Rpc", (Exception,), {"error": reply["error"]})()
        return reply["result"]

    client = SessionClient("glide", request)
    assert [t.name for t in await client.list_tools()] == ["echo"]
    result = await client.call_tool("echo", {"text": "hi"})
    assert result["structuredContent"] == {"echo": "hi"}

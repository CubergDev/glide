"""Wire and handler hardening of GlideMCPServer: what an untrusted client or a careless handler can make it do.

PR5-4175258143 (NaN), PR5-4175597271 (nesting), PR5-4175632621 (lone surrogate), PR5-4175597277 (unsupported
schema types), PR5-4175258130 (handler ValueError text), PR5-4175597262 (argument names), PR5-4175614370 (level),
PR5-4175258140 (bounded stdio read).
"""

import io
import json

import pytest
from mcp_guard_loader import no_real_memory_or_mcp, sync  # noqa: F401  (autouse guard, async runner)

from glide.mcp import GlideMCPServer, ServerTool, memory_tools, serve_stream
from glide.mcp.server import MAX_MESSAGE_BYTES, read_bounded_line
from glide.memory import Scope, Store

SCOPE = Scope("alice", "project", "mcp")
INIT = json.dumps({"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": {"protocolVersion": "2025-11-25"}})
PING = json.dumps({"jsonrpc": "2.0", "id": 99, "method": "ping"})


def call(name, arguments, id=1):
    return json.dumps({"jsonrpc": "2.0", "id": id, "method": "tools/call", "params": {"name": name, "arguments": arguments}})


async def serve(server, *lines):
    pending = [line + "\n" for line in lines]
    out = []

    async def readline():
        return pending.pop(0) if pending else ""

    await serve_stream(server, readline, out.append)
    return [json.loads(line) for line in out]


def number_tool(seen):
    schema = {"type": "object", "properties": {"x": {"type": "number"}}, "additionalProperties": False}
    return ServerTool("num", "n", schema, lambda arguments: seen.append(arguments) or {"ok": True})


@sync
@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
async def test_non_finite_numbers_never_reach_a_handler(constant):
    seen = []
    replies = await serve(
        GlideMCPServer([number_tool(seen)]),
        INIT,
        f'{{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{{"name":"num","arguments":{{"x":{constant}}}}}}}',
        PING,
    )
    assert seen == [] and replies[1]["error"]["code"] == -32700 and replies[2]["id"] == 99


@sync
async def test_a_nesting_bomb_is_a_parse_error_and_the_server_keeps_serving():
    bomb = "[" * 200_000 + "]" * 200_000
    replies = await serve(GlideMCPServer(), INIT, bomb, PING)
    assert [r.get("id") for r in replies] == [0, None, 99] and replies[1]["error"]["code"] == -32700


@sync
async def test_a_lone_surrogate_in_a_request_id_gets_an_ascii_safe_reply_and_the_stream_continues():
    ping = '{"jsonrpc":"2.0","id":"\\ud800","method":"ping"}'
    out = []
    pending = [INIT + "\n", ping + "\n", PING + "\n"]

    async def readline():
        return pending.pop(0) if pending else ""

    await serve_stream(GlideMCPServer(), readline, out.append)
    for line in out:
        line.encode("utf-8")  # the stdio writer encodes exactly this; it must not raise
    assert [json.loads(line)["id"] for line in out] == [0, "\ud800", 99]


@pytest.mark.parametrize("wanted", [["string", "null"], "null", "date", {"anyOf": []}])
def test_a_tool_whose_schema_the_server_cannot_enforce_is_refused_at_registration(wanted):
    schema = {"type": "object", "properties": {"n": {"type": wanted}}}
    with pytest.raises(ValueError, match="plain JSON type"):
        GlideMCPServer([ServerTool("t", "d", schema, lambda arguments: {})])


def test_untyped_and_typed_properties_still_register():
    schema = {"type": "object", "properties": {"any": {}, "n": {"type": "integer"}}}
    assert GlideMCPServer([ServerTool("t", "d", schema, lambda arguments: {})]).tool_names == ("t",)


@sync
async def test_a_handlers_plain_value_error_is_reported_by_type_not_text():
    def handler(arguments):
        raise ValueError("bad value sk-abcdefghijklmnopqrstuvwxyz0123456789")

    def careful(arguments):
        from glide.mcp.server import ToolInputError

        raise ToolInputError("the note is too long")

    tools = [
        ServerTool("plain", "p", {"type": "object", "properties": {}}, handler),
        ServerTool("careful", "c", {"type": "object", "properties": {}}, careful),
    ]
    replies = await serve(GlideMCPServer(tools), INIT, call("plain", {}), call("careful", {}))
    assert replies[1]["result"]["content"][0]["text"] == "the tool failed (ValueError)"
    assert replies[2]["result"]["content"][0]["text"] == "the note is too long"


@sync
async def test_an_unknown_argument_name_is_never_echoed():
    replies = await serve(GlideMCPServer([number_tool([])]), INIT, call("num", {"sk_live_SECRET123": 1}))
    assert replies[1]["error"]["message"] == "unknown argument" and "SECRET" not in json.dumps(replies[1])


@sync
async def test_a_remote_client_cannot_choose_the_memory_level(tmp_path):
    with Store(tmp_path / "s.sqlite") as store:
        tools = {tool.name: tool for tool in memory_tools(store, SCOPE, write=True)}
        assert "level" not in tools["glide.memory.remember"].input_schema["properties"]
        replies = await serve(
            GlideMCPServer(list(tools.values())),
            INIT,
            call("glide.memory.remember", {"key": "k", "text": "everywhere", "level": "user"}),
            call("glide.memory.remember", {"key": "k", "text": "here"}, id=2),
        )
        assert replies[1]["error"]["code"] == -32602 and replies[2]["result"]["isError"] is False
        [row] = store.memories(SCOPE)
        assert (row["kind"], row["source"], row["text"]) == ("note", "mcp", "here")
        assert store.memories(Scope("alice", "another-project", "mcp")) == []  # project level: not every project


class RecordingStream:
    """A text stream that records the size every readline asked for (the real stdin allocates what it is asked)."""

    def __init__(self, text):
        self._text, self.sizes = io.StringIO(text), []

    def readline(self, size=-1):
        self.sizes.append(size)
        return self._text.readline(size)


def test_the_stdio_read_is_bounded_and_the_rest_of_an_oversize_line_is_drained():
    stream = RecordingStream("x" * (3 * MAX_MESSAGE_BYTES) + "\n" + PING + "\n")
    first = read_bounded_line(stream)
    assert len(first) == MAX_MESSAGE_BYTES + 1 and not first.endswith("\n")
    assert read_bounded_line(stream) == PING + "\n"  # the drained line did not leak into the next message
    assert all(0 < size <= MAX_MESSAGE_BYTES + 1 for size in stream.sizes)


def test_a_normal_line_and_eof_pass_through():
    stream = RecordingStream(PING + "\n")
    assert read_bounded_line(stream) == PING + "\n" and read_bounded_line(stream) == ""
    unterminated = RecordingStream("y" * (MAX_MESSAGE_BYTES + 5))  # oversize, then EOF without a newline
    assert len(read_bounded_line(unterminated)) == MAX_MESSAGE_BYTES + 1 and read_bounded_line(unterminated) == ""

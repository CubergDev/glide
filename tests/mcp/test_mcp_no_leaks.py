"""A hostile server, hostile tool arguments and private memory text never come back out in an error or a log line.

Every marker below stands for something that must stay out of messages and stored events: text a remote server chose,
arguments a model or client passed, and what the user saved. The test drives every path of glide.memory and glide.mcp
that raises or reports, collects the message of everything raised, and looks for the markers. Nothing is logged:
the two packages have no logging and print nothing.
"""

import ast
import io
import json
from functools import partial
from pathlib import Path

import httpx
from mcp_guard_loader import no_real_memory_or_mcp, sync  # noqa: F401  (autouse guard, async runner)

import glide.mcp
import glide.memory
from glide.mcp import GlideMCPServer, MCPBridge, ServerTool, SessionClient, bind_mcp, memory_tools
from glide.mcp.bridge import normalize_mcp_result
from glide.memory import Catalog, Harness, Model, Policy, Reply, Scope, Store, ToolCall
from glide.memory.cli import main as memory_cli
from glide.memory.events import SQLiteEventSink
from glide.memory.store import validate_text

SCOPE = Scope("alice", "project", "session")
SERVER_TEXT = "SECRET-SERVER-TEXT"
ARGUMENT = "SECRET-ARGUMENT"
MEMORY = "SECRET-MEMORY"
MARKERS = (SERVER_TEXT, ARGUMENT, MEMORY, "SECRET-URL", "SECRET-BODY", "SECRET-KEY")


class RpcError(Exception):
    def __init__(self, code):
        self.error = {"code": code, "message": SERVER_TEXT}
        super().__init__(SERVER_TEXT)


def hostile_errors():
    request = httpx.Request("GET", "https://user:SECRET-URL@host/path?token=SECRET-URL")
    return [
        RpcError(-32602),
        RpcError(-32000),
        RpcError(408),
        httpx.ConnectError("https://user:SECRET-URL@host"),
        httpx.HTTPStatusError(SERVER_TEXT, request=request, response=httpx.Response(500, text="SECRET-BODY", request=request)),
        httpx.HTTPStatusError(SERVER_TEXT, request=request, response=httpx.Response(401, text="SECRET-BODY", request=request)),
        TimeoutError(SERVER_TEXT),
        RuntimeError(SERVER_TEXT),
    ]


def hostile_replies():
    secret = {"SECRET-KEY": SERVER_TEXT}
    return [
        None,
        SERVER_TEXT,
        {"content": [SERVER_TEXT]},
        {"content": [{"type": "resource", "resource": SERVER_TEXT}]},
        {"content": SERVER_TEXT},
        {"content": [{"type": SERVER_TEXT}]},
        {"content": [{"type": "text", "text": 5, "extra": SERVER_TEXT}]},
        {"content": [{"type": "image", "data": SERVER_TEXT}]},
        {"content": [{"type": "resource", "resource": {"uri": SERVER_TEXT, "blob": 3}}]},
        {"content": [{"type": "text", "text": "x"}], "isError": SERVER_TEXT},
        {"content": [{"type": "text", "text": "x"}], "structuredContent": SERVER_TEXT},
        {"error": {"code": 1, "message": SERVER_TEXT}},
        {"content": [{"type": "text", "text": SERVER_TEXT * 20000}]},
        {"content": [{"type": "text", "text": "x"}], "structuredContent": {"v": float("nan"), **secret}},
    ]


def message_of(error: BaseException) -> str:
    return f"{error!s} {error!r} {error.args!r}"


def collect(call) -> list[str]:
    try:
        call()
    except BaseException as error:
        return [message_of(error)]
    return []


async def acollect(call) -> list[str]:
    try:
        await call()
    except BaseException as error:
        return [message_of(error)]
    return []


def assert_clean(messages):
    assert messages, "the scenario produced no error at all"
    for message in messages:
        for marker in MARKERS:
            assert marker not in message, f"{marker} leaked into: {message[:200]}"


@sync
async def test_the_client_interface_never_repeats_what_a_server_sent_or_what_was_asked():
    messages = []
    for error in hostile_errors():

        async def failing(method, params, error=error):
            raise error

        client = SessionClient("srv", failing)
        messages += await acollect(client.list_tools)
        messages += await acollect(partial(client.call_tool, "lookup", {"q": ARGUMENT}))
    for reply in hostile_replies():

        async def answering(method, params, reply=reply):
            return reply if method == "tools/call" else {"tools": [reply]}

        client = SessionClient("srv", answering)
        messages += await acollect(partial(client.call_tool, "lookup", {"q": ARGUMENT}))
        messages += await acollect(client.list_tools)

    async def refusing(method, params):
        raise RpcError(-32000)

    for name, arguments in (("x" * 200, {}), ("lookup", {"q": "x" * 70000}), ("lookup", {"q": float("nan"), "k": ARGUMENT})):
        messages += await acollect(partial(SessionClient("srv", refusing).call_tool, name, arguments))
    assert len(messages) > 30
    assert_clean(messages)


def test_the_bridge_names_no_content_in_its_own_errors():
    messages = []
    for reply in hostile_replies():
        messages += collect(lambda reply=reply: normalize_mcp_result(reply))
        tool = bind_mcp(
            "srv",
            [{"name": "lookup", "inputSchema": {"type": "object"}, "outputSchema": {"type": "object"}}],
            lambda name, arguments, reply=reply: reply,
            permissions={"lookup": frozenset()},
            keywords={"lookup": ("lookup",)},
        )[0]
        messages += collect(lambda tool=tool: tool.invoke({"q": ARGUMENT}))
    for descriptor in (
        {"name": SERVER_TEXT * 10, "inputSchema": {"type": "object"}},
        {"name": "ok", "description": 5, "title": SERVER_TEXT, "inputSchema": {"type": "object"}},
        {"name": "ok", "annotations": SERVER_TEXT, "inputSchema": {"type": "object"}},
        {"name": "ok", "execution": {"taskSupport": SERVER_TEXT}, "inputSchema": {"type": "object"}},
        {"name": "ok", "inputSchema": {"type": SERVER_TEXT}},
        {"name": "ok", "inputSchema": {"type": "object"}, "outputSchema": SERVER_TEXT},
    ):
        messages += collect(
            lambda descriptor=descriptor: bind_mcp(
                "srv", [descriptor], lambda n, a: None, permissions={"ok": frozenset()}, keywords={"ok": ("ok",)}
            )
        )
    assert len(messages) > 25
    assert_clean(messages)


def test_stored_events_never_hold_arguments_results_or_server_text(tmp_path):
    with Store(tmp_path / "s.sqlite") as store:

        def request(method, params):
            if method == "tools/list":
                return {"tools": [{"name": "lookup", "inputSchema": {"type": "object"}}]}
            if params["arguments"].get("fail"):
                raise RpcError(-32000)
            return {"content": [{"type": "text", "text": SERVER_TEXT}], "structuredContent": {"v": SERVER_TEXT}}

        bridge = MCPBridge(
            "srv",
            request,
            permissions={"lookup": frozenset()},
            keywords={"lookup": ("lookup",)},
            capabilities={"tools": {}, "logging": {}},
        )
        bridge.event_bus.subscribe(SQLiteEventSink(store, SCOPE))
        tool = bridge.refresh()[0]
        tool.invoke({"q": ARGUMENT})
        collect(lambda: tool.invoke({"q": ARGUMENT, "fail": True}))
        bridge.on_notification("notifications/message", {"level": "info", "data": SERVER_TEXT, "SECRET-KEY": ARGUMENT})
        stored = json.dumps(store.events(SCOPE, 1000))
    assert "call_finished" in stored
    for marker in MARKERS:
        assert marker not in stored


def test_memory_and_harness_errors_never_echo_what_was_saved_or_passed(tmp_path):
    (tmp_path / "catalog").mkdir()
    credential = f"{MEMORY} sk-{'a' * 30}"
    with Store(tmp_path / "s.sqlite") as store:
        harness = Harness(
            store,
            Catalog(tmp_path / "catalog"),
            models=(Model("m", frozenset({"handoff"}), 100000, supports_tools=True),),
            policy=Policy(auto_memory=True),
        )
        messages = []
        messages += collect(lambda: store.remember(SCOPE, "k", credential))
        messages += collect(lambda: store.remember(SCOPE, "k", f"password: {MEMORY}"))
        messages += collect(lambda: store.remember(SCOPE, "k" + MEMORY * 100, "text"))
        messages += collect(lambda: store.remember(SCOPE, MEMORY, "x" * 9000, level=MEMORY))
        messages += collect(lambda: store.event(SCOPE, "kind", {"v": credential}))
        messages += collect(lambda: store.record_outcome(SCOPE, "run", True, f"visited https://{MEMORY}.example/{ARGUMENT}"))
        messages += collect(lambda: store.record_outcome(SCOPE, "run", True, credential))
        messages += collect(lambda: store.propose(SCOPE, f"prompt:{MEMORY}!", "text", [MEMORY]))
        messages += collect(lambda: store.propose(SCOPE, "prompt:x", credential, [MEMORY]))
        messages += collect(lambda: store.forget(SCOPE, MEMORY * 100 + "\n"))
        messages += collect(lambda: harness.observe_user(SCOPE, f"remember that {credential}"))
        messages += collect(lambda: harness.command(SCOPE, f"/remember k = {credential}"))
        messages += collect(lambda: harness.command(SCOPE, f"/{MEMORY} {ARGUMENT}"))
        messages += collect(lambda: harness.command(SCOPE, f"/rollback {MEMORY}"))
        plan = harness.prepare(SCOPE, "goal")
        messages += collect(
            lambda: harness.invoke(SCOPE, plan.id, MEMORY, {"v": ARGUMENT}, grants=frozenset(), authorize=lambda *_: True)
        )
        messages += collect(
            lambda: harness.invoke(SCOPE, MEMORY, "t", {"v": ARGUMENT}, grants=frozenset(), authorize=lambda *_: True)
        )
        messages += collect(
            lambda: harness.invoke(
                SCOPE, plan.id, "t", {"v": float("nan"), "k": ARGUMENT}, grants=frozenset(), authorize=lambda *_: True
            )
        )
        messages += collect(
            lambda: harness.dispatch(
                SCOPE,
                "goal",
                lambda request: Reply(calls=(ToolCall(MEMORY, ARGUMENT, {"v": ARGUMENT}),)),
                authorize=lambda *_: True,
            )
        )
        stored = json.dumps(store.events(SCOPE, 1000))
    assert len(messages) > 15
    assert_clean(messages)
    for marker in MARKERS:
        assert marker not in stored


def test_the_memory_command_line_reports_errors_without_the_text(tmp_path):
    environ = {"GLIDE_MEMORY": "1", "GLIDE_DATA_DIR": str(tmp_path / "data")}
    for words in (
        ["remember", "k", f"{MEMORY} sk-{'a' * 30}"],
        ["remember", "k", f"password: {MEMORY}"],
        ["propose", "prompt:x", f"{MEMORY} sk-{'a' * 30}", "--evidence", MEMORY],
        ["apply", MEMORY, "--revision", "0"],
        ["forget", MEMORY * 100 + "\n"],
    ):
        err = io.StringIO()
        code = memory_cli(words, environ=environ, home=tmp_path, out=io.StringIO(), err=err)
        assert code == 2 and err.getvalue()
        assert MEMORY not in err.getvalue()


@sync
async def test_the_server_reports_a_failing_tool_by_type_and_never_echoes_client_values(tmp_path):
    with Store(tmp_path / "s.sqlite") as store:

        def exploding(arguments):
            raise RuntimeError(f"{SERVER_TEXT} {arguments}")

        tools = [*memory_tools(store, SCOPE, write=True), ServerTool("explode", "x", {"type": "object"}, exploding)]
        server = GlideMCPServer(tools)
        await server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-11-25"}})
        texts = []
        for name, arguments in (
            ("glide.memory.remember", {"key": "k", "text": f"{MEMORY} sk-{'a' * 30}"}),
            ("glide.memory.remember", {"key": "k", "text": "x", "level": MEMORY}),
            ("glide.memory.remember", {"key": "k", "text": 5}),
            ("glide.memory.remember", {"key": "k", "text": "x", "SECRET-KEY": ARGUMENT}),
            ("glide.memory.remember", {"key": "k" * 500, "text": MEMORY}),
            ("glide.memory.forget", {"id": 5}),
            ("glide.memory.forget", {"id": MEMORY * 100 + "\n"}),
            (MEMORY, {}),
            ("explode", {"v": ARGUMENT}),
        ):
            reply = await server.handle(
                {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": name, "arguments": arguments}}
            )
            texts.append(json.dumps(reply))
    assert len(texts) == 9
    assert any("the tool failed (RuntimeError)" in text for text in texts)
    for text in texts:
        for marker in MARKERS:
            assert marker not in text, text[:200]


def test_neither_package_logs_or_prints():
    for package in (glide.mcp, glide.memory):
        for path in Path(package.__file__).parent.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    assert all(alias.name != "logging" for alias in node.names), path
                if isinstance(node, ast.ImportFrom):
                    assert node.module != "logging", path
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    assert node.func.id != "print", path


def test_the_leak_markers_are_not_credentials_by_accident():
    # a marker the credential scanner refuses would make "no leak" true for the wrong reason
    for marker in MARKERS:
        assert validate_text(marker) == marker

"""Whole-output characterisation of the MCP bridge: the ordered events of one session and every refusal message.

The expected values are in tests/mcp/golden/bridge.json and were produced by the bridge as it stood before it was
restructured. A difference is a behaviour change, not a stale fixture: decide it, do not regenerate it.
"""

import asyncio
import json
from pathlib import Path

import pytest

from glide.mcp.bridge import MCPBridge, MCPProtocolError, bind_mcp, normalize_mcp_result, read_tools_page
from glide.memory.events import EventBus

GOLDEN = Path(__file__).parent / "golden"


def expected(name):
    return json.loads((GOLDEN / f"{name}.json").read_text(encoding="utf-8"))


def jsonable(value):
    return json.loads(json.dumps(value))


def refusal(call) -> str:
    try:
        call()
    except BaseException as error:
        return f"{type(error).__name__}: {error}"
    return "no error"


def descriptor(name, **extra):
    return {"name": name, "description": f"Tool {name}", "inputSchema": {"type": "object"}, **extra}


TEXT = {"content": [{"type": "text", "text": "ok"}]}
PAGES = {
    None: {
        "tools": [
            descriptor("echo", outputSchema={"type": "object"}),
            descriptor("slow"),
            descriptor("boom"),
            descriptor("garbled"),
            descriptor("needs-task", execution={"taskSupport": "required"}),
            descriptor("unlisted"),
        ],
        "nextCursor": "page-2",
    },
    "page-2": {"tools": [descriptor("other")]},
}
NAMES = ("echo", "slow", "boom", "garbled", "needs-task", "other")
PERMISSIONS = {name: frozenset({"net"}) for name in NAMES}
KEYWORDS = {name: (name,) for name in NAMES}
CAPABILITIES = {"tools": {"listChanged": True}, "resources": {"subscribe": True, "listChanged": False}, "logging": {}}


class Session:
    """A bridge, its fake server and a log of the events it published, with tokens and clocks normalised."""

    def __init__(self, *, asynchronous=False, pages=PAGES, capabilities=CAPABILITIES):
        self.log, self.tokens, self.pages, self.progress_errors = [], {}, pages, []
        self.bus = EventBus()
        self.bus.subscribe(self.record)
        self.bridge = MCPBridge(
            "srv",
            self.request,
            permissions=PERMISSIONS,
            keywords=KEYWORDS,
            capabilities=capabilities,
            event_bus=self.bus,
            asynchronous=asynchronous,
        )

    def token(self, value):
        return self.tokens.setdefault(value, f"<token-{len(self.tokens) + 1}>")

    def record(self, event):
        data = dict(event.data)
        if "progressToken" in data:
            data["progressToken"] = self.token(data["progressToken"])
        if "duration_ms" in data:
            assert isinstance(data.pop("duration_ms"), float)
            data["duration_ms"] = "<float>"
        correlation = event.correlation_id
        self.log.append(
            {
                "kind": event.kind,
                "source": event.source,
                "data": data,
                "scope": event.scope,
                "correlation": self.tokens.get(correlation, correlation),
            }
        )

    def request(self, method, params):
        if method == "tools/list":
            return self.pages[params.get("cursor")]
        name = params["name"]
        token = params["_meta"]["progressToken"]
        if name == "slow":
            for progress, total in ((0.3, 1), (0.3, 1), (0.6, None), (0.9, 0.5)):
                try:
                    self.bridge.on_notification(
                        "notifications/progress", {"progressToken": token, "progress": progress, "total": total}
                    )
                except MCPProtocolError as error:
                    self.progress_errors.append(str(error))
            return TEXT
        if name == "boom":
            raise RuntimeError("server exploded")
        if name == "garbled":
            return {"content": "not a list"}
        if name == "echo" and params["arguments"].get("fail"):
            return {"content": [{"type": "text", "text": "tool says no"}], "isError": True}
        if name == "echo":
            return {**TEXT, "structuredContent": {"echoed": params["arguments"]}}
        return TEXT


def tool_by(bridge, name):
    return next(tool for tool in bridge.tools() if tool.id == f"mcp:srv/{name}")


def test_one_session_publishes_exactly_these_events_in_this_order():
    session = Session()
    bridge, outcome = session.bridge, {}
    outcome["tools_before_refresh"] = refusal(bridge.tools)
    outcome["refresh_ids"] = [tool.id for tool in bridge.refresh()]
    outcome["echo"] = tool_by(bridge, "echo").invoke({"text": "hi"})
    outcome["tool_error_is_feedback"] = tool_by(bridge, "echo").invoke({"fail": True})
    outcome["slow"] = tool_by(bridge, "slow").invoke({})
    outcome["progress_refusals"] = session.progress_errors
    outcome["boom"] = refusal(lambda: tool_by(bridge, "boom").invoke({}))
    outcome["garbled"] = refusal(lambda: tool_by(bridge, "garbled").invoke({}))
    outcome["arguments_not_json"] = refusal(lambda: tool_by(bridge, "echo").invoke({"x": float("inf")}))
    outcome["arguments_not_an_object"] = refusal(lambda: tool_by(bridge, "echo").invoke([1]))
    stale = tool_by(bridge, "echo")
    for method, params in (
        ("notifications/resources/list_changed", {}),
        ("notifications/resources/updated", {"uri": "memo://x"}),
        ("notifications/message", {"level": "info", "data": "hello"}),
        ("notifications/prompts/list_changed", {}),
        ("notifications/unknown", {}),
        ("notifications/tools/list_changed", None),
    ):
        bridge.on_notification(method, params)
    outcome["tools_after_list_changed"] = refusal(bridge.tools)
    outcome["stale_closure_refused"] = refusal(lambda: stale.invoke({}))
    outcome["refresh_after_change"] = [tool.id for tool in bridge.refresh()]
    bridge.set_connected(False)
    bridge.set_connected(False)
    outcome["refresh_while_disconnected"] = refusal(bridge.refresh)
    outcome["tools_while_disconnected"] = refusal(bridge.tools)
    bridge.set_connected(True)
    outcome["refresh_after_reconnect"] = len(bridge.refresh())
    session.pages = {None: {"tools": "nope"}}
    outcome["bad_page"] = refusal(bridge.refresh)
    session.pages = {None: {"tools": [], "nextCursor": "again"}, "again": {"tools": [], "nextCursor": "again"}}
    outcome["repeated_cursor"] = refusal(bridge.refresh)
    session.pages = {None: {"tools": [descriptor("echo"), descriptor("echo")]}}
    outcome["duplicate_names"] = refusal(bridge.refresh)
    outcome["sync_bridge_refuses_arefresh"] = refusal(lambda: asyncio.run(bridge.arefresh()))
    assert jsonable({"outcome": outcome, "events": session.log}) == expected("bridge")["one_session"]


def test_inventory_observers_get_the_current_state_not_the_event():
    session = Session()
    seen = []
    unsubscribe = session.bridge.attach_inventory(lambda tools: seen.append([tool.id for tool in tools]))
    session.bridge.refresh()
    session.bridge.on_notification("notifications/tools/list_changed")
    session.bridge.set_connected(False)
    unsubscribe()
    session.bridge.set_connected(True)
    failing = refusal(lambda: session.bridge.attach_inventory(lambda tools: 1 / 0))
    not_callable = refusal(lambda: session.bridge.attach_inventory("no"))
    assert jsonable({"seen": seen, "failing": failing, "not_callable": not_callable}) == expected("bridge")["observers"]


def test_a_cancelled_refresh_and_a_cancelled_call_are_reported_as_cancelled():
    async def scenario():
        session = Session(asynchronous=True)
        started = asyncio.Event()

        async def hang(method, params):
            started.set()
            await asyncio.sleep(60)

        calls = {"tools/list": session.request}

        async def request(method, params):
            if method == "tools/call" and params["name"] == "slow":
                return await hang(method, params)
            return calls.get(method, session.request)(method, params)

        session.bridge.request = request
        await session.bridge.arefresh()
        task = asyncio.ensure_future(tool_by(session.bridge, "slow").invoke({}))
        await started.wait()
        task.cancel()
        outcome = {"call": refusal_async(await settle(task))}
        started.clear()
        session.pages = {None: {"tools": []}}

        async def hanging_list(method, params):
            started.set()
            await asyncio.sleep(60)

        session.bridge.request = hanging_list
        refresh = asyncio.ensure_future(session.bridge.arefresh())
        await started.wait()
        refresh.cancel()
        outcome["refresh"] = refusal_async(await settle(refresh))
        outcome["sync_refresh_on_async_bridge"] = refusal(session.bridge.refresh)
        return {"outcome": outcome, "events": session.log}

    assert jsonable(asyncio.run(scenario())) == expected("bridge")["cancellation"]


async def settle(task):
    try:
        await task
    except BaseException as error:
        return error
    return None


def refusal_async(error) -> str:
    return "no error" if error is None else f"{type(error).__name__}: {error}"


def test_a_validator_that_rejects_or_breaks_is_a_protocol_error():
    def bound(validate):
        tool = bind_mcp(
            "srv",
            [descriptor("echo", outputSchema={"type": "object"})],
            lambda name, arguments: {**TEXT, "structuredContent": {"a": 1}},
            permissions=PERMISSIONS,
            keywords=KEYWORDS,
            validate_output=validate,
        )[0]
        return refusal(lambda: tool.invoke({}))

    def boom(schema, value):
        raise RuntimeError("validator exploded")

    results = {
        "accepts": bound(lambda schema, value: True),
        "rejects": bound(lambda schema, value: False),
        "breaks": bound(boom),
        "returns_none_is_accepted": bound(lambda schema, value: None),
    }
    missing = bind_mcp(
        "srv",
        [descriptor("echo", outputSchema={"type": "object"})],
        lambda name, arguments: TEXT,
        permissions=PERMISSIONS,
        keywords=KEYWORDS,
    )[0]
    results["structured_content_missing"] = refusal(lambda: missing.invoke({}))
    assert jsonable(results) == expected("bridge")["validators"]


def test_every_malformed_result_and_page_has_its_own_message():
    results = {}
    cases = {
        "not_a_dict": None,
        "error_member": {"error": {"code": 1}},
        "content_not_list": {"content": 1},
        "is_error_not_bool": {**TEXT, "isError": 1},
        "structured_not_object": {**TEXT, "structuredContent": 1},
        "block_type_not_string": {"content": [{"type": 1}]},
        "text_without_string": {"content": [{"type": "text"}]},
        "media_without_mime": {"content": [{"type": "image", "data": "x"}]},
        "resource_without_uri": {"content": [{"type": "resource", "resource": {"text": "t"}}]},
        "resource_blob_not_string": {"content": [{"type": "resource", "resource": {"uri": "u", "blob": 1}}]},
        "resource_without_body": {"content": [{"type": "resource", "resource": {"uri": "u"}}]},
        "resource_not_a_dict": {"content": [{"type": "resource", "resource": 1}]},
        "link_without_name": {"content": [{"type": "resource_link", "uri": "u"}]},
        "unsupported_type": {"content": [{"type": "video"}]},
        "oversized": {"content": [{"type": "text", "text": "x" * 70000}]},
        "not_json": {"content": [{"type": "text", "text": "t"}], "structuredContent": {"a": float("nan")}},
    }
    for name, value in cases.items():
        results[name] = refusal(lambda value=value: normalize_mcp_result(value))
    results["media_is_projected"] = normalize_mcp_result(
        {
            "content": [
                {"type": "audio", "data": "AAAA", "mimeType": "audio/wav", "_meta": {"x": 1}},
                {"type": "resource", "resource": {"uri": "u", "blob": "BB", "_meta": 1}},
            ]
        }
    )
    inventory, seen = [], set()
    pages = {
        "tools_not_list": {"tools": 1},
        "cursor_not_string": {"tools": [], "nextCursor": 5},
        "cursor_empty": {"tools": [], "nextCursor": ""},
        "cursor_too_long": {"tools": [], "nextCursor": "c" * 4097},
        "page_not_a_dict": [1],
        "page_has_error": {"error": 1},
    }
    for name, value in pages.items():
        results[name] = refusal(lambda value=value: read_tools_page(value, inventory, seen))
    results["last_page_has_no_cursor"] = read_tools_page({"tools": [descriptor("a")]}, inventory, seen)
    results["inventory_too_large"] = refusal(lambda: read_tools_page({"tools": [descriptor("t")] * 4097}, [], set()))
    assert jsonable(results) == expected("bridge")["malformed"]


@pytest.mark.parametrize(
    "arguments",
    [
        ({"server": 1}, "constructor_server_not_text"),
        ({"server": "bad name!"}, "constructor_server_not_an_identifier"),
        ({"request": "no"}, "constructor_request_not_callable"),
        ({"asynchronous": 1}, "constructor_asynchronous_not_bool"),
        ({"validate_output": 1}, "constructor_validator_not_callable"),
        ({"permissions": []}, "constructor_permissions_not_a_mapping"),
        ({"keywords": []}, "constructor_keywords_not_a_mapping"),
        ({"capabilities": {"tools": 1}}, "constructor_capability_not_an_object"),
        ({"capabilities": []}, "constructor_capabilities_not_a_mapping"),
    ],
    ids=lambda case: case[1] if isinstance(case, tuple) else "",
)
def test_constructor_refusals_are_pinned(arguments):
    changes, name = arguments
    options = {"server": "srv", "request": lambda m, p: TEXT, "permissions": {}, "keywords": {}, "capabilities": {"tools": {}}}
    options.update(changes)
    server, request = options.pop("server"), options.pop("request")
    assert refusal(lambda: MCPBridge(server, request, **options)) == expected("bridge")["constructor"][name]


def test_notification_refusals_and_a_bridge_without_a_tools_capability():
    session = Session()
    session.bridge.refresh()
    results = {
        "progress_token_not_valid": refusal(
            lambda: session.bridge.on_notification("notifications/progress", {"progressToken": True})
        ),
        "progress_for_an_unknown_call_is_ignored": refusal(
            lambda: session.bridge.on_notification("notifications/progress", {"progressToken": "none", "progress": 1})
        ),
        "params_not_a_mapping": refusal(lambda: session.bridge.on_notification("notifications/message", 5)),
        "params_with_an_error": refusal(lambda: session.bridge.on_notification("notifications/message", {"error": 1})),
    }
    bare = Session(capabilities={})
    results["no_tools_capability_means_no_tools"] = list(bare.bridge.tools())
    results["no_tools_capability_refuses_refresh"] = refusal(bare.bridge.refresh)
    results["capabilities_are_copies"] = (
        session.bridge.capabilities == CAPABILITIES and session.bridge.capabilities is not session.bridge.capabilities
    )
    results["not_connected_flag"] = refusal(lambda: session.bridge.set_connected("yes"))
    assert jsonable(results) == expected("bridge")["notifications"]

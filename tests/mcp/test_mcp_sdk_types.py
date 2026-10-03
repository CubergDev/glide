"""Optional compatibility checks with real SDK types and in-process session doubles."""

import unittest

from glide.mcp import MCPBridge, MCPStaleCatalogError, normalize_mcp_result, session_request

try:
    from mcp import types
    from mcp.shared import exceptions as sdk_exceptions

    McpError = getattr(sdk_exceptions, "McpError", None)  # 2.x renamed it (MCPError) and changed these types
except ImportError:
    types = None
    McpError = None


@unittest.skipIf(
    types is None or McpError is None, "needs the MCP Python SDK 1.x (glide[mcp] pins mcp>=1.30,<2); 2.x changed these types"
)
class MCPTypesTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_discovery_result_capabilities_notifications_and_progress(self):
        class Session:
            async def list_tools(self, cursor=None):
                return types.ListToolsResult(tools=[types.Tool(name="lookup", inputSchema={"type": "object"})])

            async def call_tool(self, name, arguments, progress_callback=None, *, meta=None):
                await progress_callback(1, 1, "ephemeral update")
                return types.CallToolResult(
                    content=[types.TextContent(type="text", text="42")], structuredContent={"answer": 42}, isError=False
                )

        bridge = None
        adapted = session_request(Session(), on_notification=lambda method, data: bridge.on_notification(method, data))
        bridge = MCPBridge(
            "real-types",
            adapted,
            asynchronous=True,
            capabilities=types.ServerCapabilities(tools=types.ToolsCapability(listChanged=True)),
            permissions={"lookup": frozenset()},
            keywords={"lookup": ("lookup",)},
        )
        seen = []
        bridge.event_bus.subscribe(seen.append)
        tools = await bridge.arefresh()
        result = await tools[0].invoke({})
        self.assertEqual(result["structuredContent"], {"answer": 42})
        self.assertTrue(any(event.kind == "mcp.progress" for event in seen))
        failure = types.CallToolResult(content=[types.TextContent(type="text", text="feedback")], isError=True)
        self.assertTrue(normalize_mcp_result(failure)["isError"])
        notification = types.ServerNotification(root=types.ToolListChangedNotification(method="notifications/tools/list_changed"))
        wire = notification.model_dump(mode="json", by_alias=True, exclude_none=True)
        bridge.on_notification(wire["method"], wire.get("params", {}))
        with self.assertRaises(MCPStaleCatalogError):
            bridge.tools()

    async def test_real_sdk_protocol_exception_is_audited_separately(self):
        failure = McpError(types.ErrorData(code=-32602, message="invalid request"))

        async def request(method, params):
            if method == "tools/list":
                return types.ListToolsResult(tools=[types.Tool(name="lookup", inputSchema={"type": "object"})])
            raise failure

        bridge = MCPBridge(
            "real-types",
            request,
            asynchronous=True,
            capabilities={"tools": {}},
            permissions={"lookup": frozenset()},
            keywords={"lookup": ("lookup",)},
        )
        seen = []
        bridge.event_bus.subscribe(seen.append)
        tools = await bridge.arefresh()
        with self.assertRaises(McpError):
            await tools[0].invoke({})
        self.assertTrue(any(event.kind == "mcp.protocol_failure" for event in seen))
        self.assertFalse(any(event.kind == "mcp.execution_failure" for event in seen))


from mcp_guard_loader import no_real_memory_or_mcp  # noqa: E402, F401  (autouse guard)

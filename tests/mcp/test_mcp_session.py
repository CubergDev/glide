import asyncio
import sys
import unittest
from datetime import timedelta
from unittest.mock import patch

from glide.mcp import MCPBridge, session_request


class SDKAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_session_cursor_metadata_progress_and_timeout(self):
        class Session:
            def __init__(self):
                self.pages = []
                self.calls = []

            async def list_tools(self, cursor=None):
                self.pages.append(cursor)
                return (
                    {"tools": [{"name": "lookup", "inputSchema": {"type": "object"}}]}
                    if cursor
                    else {"tools": [], "nextCursor": "second"}
                )

            async def call_tool(self, name, arguments, read_timeout_seconds=None, progress_callback=None, *, meta=None):
                self.calls.append((name, arguments, meta, read_timeout_seconds))
                await progress_callback(1, 2, "ephemeral message")
                return {"content": [{"type": "text", "text": "success"}]}

        session = Session()
        bridge = None
        adapted = session_request(
            session,
            read_timeout_seconds=timedelta(seconds=5),
            on_notification=lambda method, data: bridge.on_notification(method, data),
        )
        bridge = MCPBridge(
            "mock",
            adapted,
            asynchronous=True,
            capabilities={"tools": {"listChanged": True}},
            permissions={"lookup": frozenset()},
            keywords={"lookup": ("lookup",)},
        )
        seen = []
        bridge.event_bus.subscribe(seen.append)
        tools = await bridge.arefresh()
        result = await tools[0].invoke({"value": 1})
        self.assertEqual(result["content"][0]["text"], "success")
        self.assertEqual(session.pages, [None, "second"])
        self.assertEqual(session.calls[0][3], timedelta(seconds=5))
        token = session.calls[0][2]["progressToken"]
        event = next(e for e in seen if e.kind == "mcp.progress")
        self.assertEqual(event.data["progressToken"], token)
        self.assertEqual(event.data["progress"], 1)

    async def test_params_only_sdk_requires_native_pagination_factory(self):
        class Session:
            async def list_tools(self, *, params=None):
                return params

            async def call_tool(self, name, arguments):
                return name

        session = Session()
        plain = session_request(session)
        with patch.dict(sys.modules, {"mcp": None, "mcp.types": None}), self.assertRaisesRegex(TypeError, "factory"):
            await plain("tools/list", {"cursor": "next"})
        adapted = session_request(session, page_params=lambda cursor: ("native", cursor))
        self.assertEqual(await adapted("tools/list", {"cursor": "next"}), ("native", "next"))

    async def test_other_methods_never_call_session(self):
        class Session:
            async def list_tools(self):
                raise AssertionError("must not be called")

            async def call_tool(self, name, arguments):
                raise AssertionError("must not be called")

        adapted = session_request(Session())
        with self.assertRaises(ValueError):
            await adapted("resources/read", {"uri": "file:///private"})

    async def test_async_progress_observer_is_awaited(self):
        seen = []

        class Session:
            async def list_tools(self):
                return {}

            async def call_tool(self, name, arguments, progress_callback=None, *, meta=None):
                await progress_callback(1)
                return {}

        async def observer(method, params):
            await asyncio.sleep(0)
            seen.append(params)

        adapted = session_request(Session(), on_notification=observer)
        await adapted("tools/call", {"name": "lookup", "_meta": {"progressToken": "token"}})
        self.assertEqual(seen, [{"progressToken": "token", "progress": 1}])

    async def test_missing_metadata_support_fails_before_call(self):
        class Session:
            async def list_tools(self):
                return {}

            async def call_tool(self, name, arguments):
                raise AssertionError("must not be called")

        adapted = session_request(Session())
        with self.assertRaisesRegex(TypeError, "metadata"):
            await adapted("tools/call", {"name": "lookup", "_meta": {"progressToken": "token"}})

    async def test_notification_fault_cannot_mask_successful_tool_result(self):
        class Session:
            async def list_tools(self):
                return {}

            async def call_tool(self, name, arguments, progress_callback=None, *, meta=None):
                await progress_callback(1)
                return {"content": [{"type": "text", "text": "effect completed"}]}

        def observer(*_):
            raise ValueError("observer failed")

        adapted = session_request(Session(), on_notification=observer)
        result = await adapted("tools/call", {"name": "lookup", "_meta": {"progressToken": "token"}})
        self.assertEqual(result["content"][0]["text"], "effect completed")
        self.assertEqual(adapted.notification_errors[-1]["error"], "ValueError")

    async def test_async_notification_cancellation_propagates(self):
        class Session:
            async def list_tools(self):
                return {}

            async def call_tool(self, name, arguments, progress_callback=None, *, meta=None):
                await progress_callback(1)
                return {}

        async def observer(*_):
            raise asyncio.CancelledError()

        adapted = session_request(Session(), on_notification=observer)
        with self.assertRaises(asyncio.CancelledError):
            await adapted("tools/call", {"name": "lookup", "_meta": {"progressToken": "token"}})
        self.assertFalse(adapted.notification_errors)


if __name__ == "__main__":
    unittest.main()


from mcp_guard_loader import no_real_memory_or_mcp  # noqa: E402, F401  (autouse guard)

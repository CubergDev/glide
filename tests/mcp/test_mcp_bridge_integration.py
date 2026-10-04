import contextvars
import json
import tempfile
import unittest
from pathlib import Path

from glide.mcp import MCPBridge, MCPStaleCatalogError
from glide.memory import Catalog, EventBus, Harness, Model, Reply, Scope, SQLiteEventSink, Store, ToolCall


class SDKResult:
    def __init__(self, value):
        self.value = value

    def model_dump(self, **options):
        if options != {"mode": "json", "by_alias": True, "exclude_none": True}:
            raise AssertionError("SDK normalization must preserve native aliases")
        return self.value


class MCPIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_host_model_mcp_progress_sqlite_and_inventory_refresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "catalog/plugins").mkdir(parents=True)
            metadata = {"id": "mcp-tools", "description": "Connected tool", "skills": [], "tools": ["mcp:mock/lookup"]}
            (root / "catalog/plugins/mcp.md").write_text("---\n" + json.dumps(metadata) + "\n---\n")
            scope = Scope("alice", "project", "session")
            with Store(root / "state.sqlite") as store:
                bus, ephemeral = EventBus(), []
                sink = SQLiteEventSink(store, default_scope=scope)
                bus.subscribe(sink)
                bus.subscribe(ephemeral.append)
                bridge = None
                calls = []
                listing = [
                    {
                        "name": "lookup",
                        "inputSchema": {"type": "object"},
                        "outputSchema": {"properties": {"answer": {"type": "integer"}}},
                    }
                ]

                async def request(method, params):
                    if method == "tools/list":
                        if "cursor" not in params:
                            return SDKResult({"tools": listing, "nextCursor": "second-page"})
                        self.assertEqual(params["cursor"], "second-page")
                        return SDKResult({"tools": []})
                    self.assertEqual(method, "tools/call")
                    calls.append(params)
                    token = params["_meta"]["progressToken"]
                    # Notification delivery can occur outside the caller's Python task context.
                    contextvars.Context().run(
                        bridge.on_notification,
                        "notifications/progress",
                        {"progressToken": token, "progress": 1, "total": 2, "message": "private user data"},
                    )
                    return SDKResult(
                        {
                            "content": [{"type": "text", "text": "answer 42"}],
                            "structuredContent": {"answer": 42},
                            "isError": False,
                        }
                    )

                bridge = MCPBridge(
                    "mock",
                    request,
                    asynchronous=True,
                    permissions={"lookup": frozenset({"lookup"})},
                    keywords={"lookup": ("lookup",)},
                    capabilities={"tools": {"listChanged": True}, "resources": {"subscribe": True}, "logging": {}},
                    event_bus=bus,
                    validate_output=lambda schema, value: value["answer"] == 42,
                )
                await bridge.arefresh()
                harness = Harness(
                    store,
                    Catalog(root / "catalog", enabled_plugins=frozenset({"mcp-tools"})),
                    tools=bridge.tools(),
                    models=(Model("mock-model", frozenset({"handoff"}), 32768, supports_tools=True),),
                )
                detach = bridge.attach_inventory(lambda tools: harness.replace_tool_namespace("mcp:mock/", tools))
                self.addCleanup(detach)
                requests = []

                async def model(value):
                    requests.append(value)
                    if len(requests) == 1:
                        self.assertEqual(value.plan.tools[0].id, "mcp:mock/lookup")
                        return Reply(calls=(ToolCall("model-call-1", "mcp:mock/lookup", {"question": "private arguments"}),))
                    self.assertEqual(value.trajectory[1]["result"]["structuredContent"]["answer"], 42)
                    return Reply("finished")

                reply = await harness.adispatch(
                    scope, "lookup answer", model, grants=frozenset({"lookup"}), authorize=lambda *_: True
                )
                self.assertEqual(reply.text, "finished")
                self.assertEqual(len(calls), 1)
                progress = next(event for event in ephemeral if event.kind == "mcp.progress")
                self.assertEqual(progress.scope, scope)
                self.assertEqual(progress.correlation_id, "model-call-1")
                saved = store.events(scope)
                self.assertTrue(any(e["kind"] == "mcp.progress" and e["payload"]["progress"] == 1 for e in saved))
                serialized = json.dumps(saved)
                self.assertNotIn("private user data", serialized)
                self.assertNotIn("private arguments", serialized)
                self.assertEqual(sink.errors, []) if isinstance(sink.errors, list) else self.assertFalse(sink.errors)
                old = bridge.tools()[0]
                old_plan = harness.prepare(scope, "lookup", grants=frozenset({"lookup"}))
                bridge.on_notification("notifications/tools/list_changed", {})
                self.assertFalse(harness.tools)
                with self.assertRaises(MCPStaleCatalogError):
                    await old.invoke({})
                listing.clear()
                await bridge.arefresh()
                self.assertFalse(harness.tools)
                with self.assertRaisesRegex(ValueError, "expired"):
                    await harness.ainvoke(
                        scope,
                        old_plan.id,
                        "mcp:mock/lookup",
                        {},
                        grants=frozenset({"lookup"}),
                        authorize=lambda *_: True,
                    )
                self.assertEqual(len(calls), 1)
                bridge.on_notification("notifications/resources/updated", {"uri": "file:///private-reference"})
                bridge.on_notification("notifications/message", {"level": "info", "data": "private log"})
                self.assertNotIn("private log", json.dumps(store.events(scope)))
                self.assertNotIn("private-reference", json.dumps(store.events(scope)))
                self.assertEqual(len(calls), 1)  # Notifications never fetch resources or invoke tools.


if __name__ == "__main__":
    unittest.main()


from mcp_guard_loader import no_real_memory_or_mcp  # noqa: E402, F401  (autouse guard)

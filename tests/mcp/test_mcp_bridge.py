"""Only local mocked callbacks and temporary extension state are exercised."""

import asyncio
import json
import tempfile
import threading
import unittest
from pathlib import Path

from glide.mcp.bridge import MCPBridge, MCPProtocolError, MCPStaleCatalogError, bind_mcp, normalize_mcp_result
from glide.memory.contracts import Scope
from glide.memory.events import EventBus, SQLiteEventSink, current_invocation, invocation
from glide.memory.store import Store


def descriptor(name="echo", **extra):
    return {"name": name, "description": "Echo a message", "inputSchema": {"type": "object"}, **extra}


def reply(text="done", **extra):
    return {"content": [{"type": "text", "text": text}], **extra}


class SDKLike:
    def __init__(self, value):
        self.value = value
        self.options = None

    def model_dump(self, **options):
        self.options = options
        return self.value


class NormalizationTests(unittest.TestCase):
    def test_protocol_metadata_is_kept_out_of_model_visible_content(self):
        source = reply(_meta={"private_widget_state": "client-only"})
        source["content"][0]["_meta"] = {"private_client_data": "client-only"}
        source["content"].append(
            {
                "type": "resource",
                "resource": {"uri": "memo://note", "text": "note", "_meta": {"private": "client-only"}},
            }
        )
        result = normalize_mcp_result(source)
        self.assertNotIn("client-only", json.dumps(result))
        self.assertEqual(source["_meta"]["private_widget_state"], "client-only")

    def test_sdk_aliases_errors_and_detached_structure(self):
        model = SDKLike(reply("try another input", isError=True, structuredContent={"retry": True}))
        value = normalize_mcp_result(model)
        self.assertEqual(model.options, {"mode": "json", "by_alias": True, "exclude_none": True})
        self.assertTrue(value["isError"])
        self.assertEqual(value["structuredContent"], {"retry": True})
        value["content"][0]["text"] = "changed"
        self.assertEqual(model.value["content"][0]["text"], "try another input")

    def test_media_projection_and_resource_references_do_not_call_anything(self):
        encoded = "BASE64_MEDIA_SENTINEL" * 50000
        source = {
            "content": [
                {"type": "image", "data": encoded, "mimeType": "image/png"},
                {"type": "audio", "data": encoded, "mimeType": "audio/wav"},
                {"type": "resource", "resource": {"uri": "memo://binary", "blob": encoded}},
                {"type": "resource", "resource": {"uri": "memo://text", "text": "note"}},
                {"type": "resource_link", "uri": "memo://later", "name": "later"},
            ]
        }
        result = normalize_mcp_result(source)
        self.assertNotIn("BASE64_MEDIA_SENTINEL", json.dumps(result))
        self.assertEqual(result["content"][0]["encoded_bytes"], len(encoded))
        self.assertEqual(result["content"][-1]["uri"], "memo://later")
        self.assertIn("data", source["content"][0])

    def test_invalid_and_oversized_results_fail_closed(self):
        values = [
            None,
            {},
            {"error": {"code": -32602, "message": "invalid"}},
            reply(isError="true"),
            reply(structuredContent=[]),
            {"content": [{"type": "image", "data": "AA=="}]},
            {"content": [{"type": "unknown"}]},
            reply("x" * 70000),
            {"content": [{"type": []}]},
            reply(structuredContent={"bad": float("nan")}),
        ]
        for value in values:
            with self.subTest(value_type=type(value).__name__), self.assertRaises(MCPProtocolError):
                normalize_mcp_result(value)

    def test_bind_default_normalizes_and_explicit_legacy_opt_out(self):
        permissions, keywords = {"echo": frozenset({"read"})}, {"echo": ("echo",)}
        tool = bind_mcp("mock", [descriptor()], lambda name, args: SDKLike(reply()), permissions=permissions, keywords=keywords)[
            0
        ]
        self.assertEqual(tool.invoke({}), reply())
        legacy = bind_mcp(
            "mock",
            [descriptor()],
            lambda name, args: "legacy",
            permissions=permissions,
            keywords=keywords,
            normalize_results=False,
        )[0]
        self.assertEqual(legacy.invoke({}), "legacy")

    def test_host_binding_types_and_remote_metadata_are_validated(self):
        valid = {"permissions": {"echo": frozenset()}, "keywords": {"echo": ("echo",)}}
        for changed in (
            {"permissions": {"echo": {"read"}}},
            {"permissions": {"echo": frozenset({1})}},
            {"keywords": {"echo": ["echo"]}},
            {"keywords": {"echo": ()}},
            {"keywords": {"echo": ("",)}},
            {"asynchronous": 1},
            {"normalize_results": "yes"},
        ):
            with self.subTest(changed=changed), self.assertRaises(TypeError):
                bind_mcp("mock", [descriptor()], lambda n, a: reply(), **(valid | changed))
        for extra in (
            {"description": 1},
            {"annotations": "hint"},
            {"execution": "required"},
            {"execution": {"taskSupport": "invented"}},
            {"execution": {"taskSupport": []}},
            {"outputSchema": []},
        ):
            with self.subTest(extra=extra), self.assertRaises(MCPProtocolError):
                bind_mcp("mock", [descriptor(**extra)], lambda n, a: reply(), **valid)
        with self.assertRaises(ValueError):
            bind_mcp("x" * 129, [], lambda n, a: reply(), **valid)
        with self.assertRaises(MCPProtocolError):
            bind_mcp("mock", [descriptor("x" * 129)], lambda n, a: reply(), **valid)

    def test_implicit_and_reference_output_schemas_are_private_snapshots(self):
        for schema in ({"properties": {"count": {"type": "integer"}}}, {"$ref": "#/$defs/result"}):
            validations = []
            inventory = [descriptor(outputSchema=schema)]
            tool = bind_mcp(
                "mock",
                inventory,
                lambda n, a: reply(structuredContent={"count": 1}),
                permissions={"echo": frozenset()},
                keywords={"echo": ("echo",)},
                validate_output=lambda output, data, validations=validations: validations.append(output),
            )[0]
            tool.output_schema["mutated"] = True
            inventory[0]["outputSchema"] = {"type": "array"}
            tool.invoke({})
            self.assertEqual(validations, [schema])


class BridgeTests(unittest.TestCase):
    def make_bridge(self, request, *, names=("echo",), capabilities=None, **options):
        return MCPBridge(
            "mock",
            request,
            permissions={name: frozenset({"read"}) for name in names},
            keywords={name: (name,) for name in names},
            capabilities={"tools": {"listChanged": True}} if capabilities is None else capabilities,
            **options,
        )

    def test_discovery_is_explicit_paginated_and_host_filtered(self):
        requests = []

        def request(method, params):
            requests.append((method, params))
            if not params:
                return SDKLike({"tools": [descriptor("echo")], "nextCursor": "two"})
            return {"tools": [descriptor("ungranted", annotations={"readOnlyHint": True})]}

        bridge = self.make_bridge(request)
        self.assertEqual(requests, [])
        with self.assertRaises(MCPStaleCatalogError):
            bridge.tools()
        self.assertEqual([tool.id for tool in bridge.refresh()], ["mcp:mock/echo"])
        self.assertEqual(requests, [("tools/list", {}), ("tools/list", {"cursor": "two"})])

    def test_attached_inventory_installs_withdraws_and_unsubscribes_without_requests(self):
        requests, installed, bus = [], [], EventBus()

        def request(method, params):
            requests.append(method)
            return {"tools": [descriptor()]}

        bridge = self.make_bridge(request, event_bus=bus)
        unsubscribe = bridge.attach_inventory(lambda tools: installed.append(tuple(tool.id for tool in tools)))
        self.assertEqual(installed, [()])
        self.assertEqual(requests, [])
        bridge.refresh()
        self.assertEqual(installed[-1], ("mcp:mock/echo",))
        bus.publish("mcp.tools_changed", "mcp:other", {"generation": 2})
        self.assertEqual(installed[-1], ("mcp:mock/echo",))
        bridge.on_notification("notifications/tools/list_changed", {})
        self.assertEqual(installed[-1], ())
        self.assertEqual(requests, ["tools/list"])
        bridge.refresh()
        self.assertEqual(installed[-1], ("mcp:mock/echo",))
        bridge.set_connected(False)
        self.assertEqual(installed[-1], ())
        bridge.set_connected(True)
        self.assertEqual(installed[-1], ())
        self.assertEqual(requests, ["tools/list", "tools/list"])
        unsubscribe()
        unsubscribe()
        count = len(installed)
        bridge.refresh()
        self.assertEqual(len(installed), count)

    def test_attach_clean_inventory_initially_and_isolate_subsequent_callback_faults(self):
        bus, updates, requests = EventBus(), [], []

        def request(method, params):
            requests.append(method)
            return {"tools": [descriptor()]} if method == "tools/list" else reply()

        bridge = self.make_bridge(request, event_bus=bus)
        bridge.refresh()

        def callback(tools):
            updates.append(tools)
            if len(updates) > 1:
                raise RuntimeError("host registry fault")

        bridge.attach_inventory(callback)
        self.assertEqual(updates[0][0].id, "mcp:mock/echo")
        tool = bridge.refresh()[0]
        self.assertEqual(len(bus.errors), 1)
        self.assertEqual(tool.invoke({}), reply())
        bridge.on_notification("notifications/tools/list_changed", {})
        self.assertEqual(len(bus.errors), 2)
        self.assertEqual(requests, ["tools/list", "tools/list", "tools/call"])

    def test_late_inventory_events_republish_current_authoritative_state(self):
        bus, installed = EventBus(), []
        bridge = self.make_bridge(lambda method, params: {"tools": [descriptor()]}, event_bus=bus)
        bridge.refresh()
        bridge.refresh()
        bridge.attach_inventory(lambda tools: installed.append(tuple(tool.id for tool in tools)))
        current = ("mcp:mock/echo",)
        self.assertEqual(installed[-1], current)
        for kind in ("mcp.inventory_refreshed", "mcp.tools_changed", "mcp.connection_changed"):
            with self.subTest(kind=kind):
                bus.publish(kind, "mcp:mock", {"generation": 1, "connected": False})
                self.assertEqual(installed[-1], current)
        bridge.on_notification("notifications/tools/list_changed", {})
        self.assertEqual(installed[-1], ())
        bus.publish("mcp.inventory_refreshed", "mcp:mock", {"generation": 1})
        self.assertEqual(installed[-1], ())
        bridge.refresh()
        self.assertEqual(installed[-1], current)

    def test_initial_attachment_fault_unsubscribes_before_any_requests(self):
        calls, bus = [], EventBus()
        bridge = self.make_bridge(lambda m, p: self.fail("attach must not make requests"), event_bus=bus)

        def broken(tools):
            calls.append(tools)
            raise RuntimeError("initial registry fault")

        with self.assertRaises(RuntimeError):
            bridge.attach_inventory(broken)
        bridge.on_notification("notifications/tools/list_changed", {})
        self.assertEqual(calls, [()])
        self.assertEqual(len(bus.errors), 0)

    def test_refresh_notification_race_cannot_publish_a_stale_inventory(self):
        bus, updates, failures = EventBus(), [], []
        event_ready, release = threading.Event(), threading.Event()

        def pause_refresh(event):
            if event.kind == "mcp.inventory_refreshed":
                event_ready.set()
                if not release.wait(2):
                    raise RuntimeError("test observer timed out")

        bus.subscribe(pause_refresh)
        bridge = self.make_bridge(lambda m, p: {"tools": [descriptor()]}, event_bus=bus)
        bridge.attach_inventory(lambda tools: updates.append(tuple(tool.id for tool in tools)))

        def refresh():
            try:
                bridge.refresh()
            except Exception as error:
                failures.append(error)

        worker = threading.Thread(target=refresh, daemon=True)
        worker.start()
        self.assertTrue(event_ready.wait(2))
        bridge.on_notification("notifications/tools/list_changed", {})
        release.set()
        worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertTrue(all(update == () for update in updates))
        self.assertEqual(len(failures), 1)
        self.assertIsInstance(failures[0], MCPStaleCatalogError)

    def test_repeated_cursor_and_invalid_refresh_are_atomic(self):
        responses = [{"tools": [descriptor()]}]

        def request(method, params):
            return responses.pop(0)

        bridge = self.make_bridge(request)
        original = bridge.refresh()[0]
        responses.extend([{"tools": [descriptor("new")], "nextCursor": "repeat"}, {"tools": [], "nextCursor": "repeat"}])
        with self.assertRaisesRegex(MCPProtocolError, "repeated"):
            bridge.refresh()
        self.assertEqual(bridge.tools()[0].id, original.id)
        responses.append({"tools": [descriptor(), descriptor()]})
        with self.assertRaises(MCPProtocolError):
            bridge.refresh()
        self.assertEqual(bridge.tools()[0].id, original.id)

    def test_disabled_malformed_descriptor_cannot_fail_after_publication(self):
        responses = [{"tools": [descriptor()]}]
        bridge = self.make_bridge(lambda m, p: responses.pop(0))
        old = bridge.refresh()[0]
        responses.append({"tools": [descriptor("echo"), descriptor("disabled", execution="invalid")]})
        with self.assertRaises(MCPProtocolError):
            bridge.refresh()
        self.assertEqual(bridge.tools()[0].id, old.id)
        responses.append(reply())
        self.assertEqual(old.invoke({}), reply())

    def test_pagination_and_inventory_bounds(self):
        count = 0

        def pages(method, params):
            nonlocal count
            count += 1
            return {"tools": [], "nextCursor": str(count)}

        with self.assertRaisesRegex(MCPProtocolError, "128 pages"):
            self.make_bridge(pages).refresh()
        self.assertEqual(count, 128)
        with self.assertRaisesRegex(MCPProtocolError, "4096 tools"):
            self.make_bridge(lambda m, p: {"tools": [descriptor(str(i)) for i in range(4097)]}).refresh()

    def test_notification_disconnect_and_refresh_invalidate_old_closures(self):
        requests = []

        def request(method, params):
            requests.append(method)
            return {"tools": [descriptor()]} if method == "tools/list" else reply()

        bridge = self.make_bridge(request)
        old = bridge.refresh()[0]
        bridge.on_notification("notifications/tools/list_changed", {})
        with self.assertRaises(MCPStaleCatalogError):
            bridge.tools()
        with self.assertRaises(MCPStaleCatalogError):
            old.invoke({})
        self.assertNotIn("tools/call", requests)
        current = bridge.refresh()[0]
        self.assertEqual(current.invoke({}), reply())
        with self.assertRaises(MCPStaleCatalogError):
            old.invoke({})
        bridge.set_connected(False)
        with self.assertRaises(MCPStaleCatalogError):
            current.invoke({})
        bridge.set_connected(True)
        with self.assertRaises(MCPStaleCatalogError):
            bridge.tools()
        bridge.refresh()

    def test_inventory_change_during_refresh_prevents_publication(self):
        bridge = None

        def request(method, params):
            bridge.on_notification("notifications/tools/list_changed")
            return {"tools": [descriptor()]}

        bridge = self.make_bridge(request)
        with self.assertRaises(MCPStaleCatalogError):
            bridge.refresh()
        with self.assertRaises(MCPStaleCatalogError):
            bridge.tools()

    def test_is_error_is_model_visible_and_protocol_error_is_distinct(self):
        bus, seen = EventBus(), []
        bus.subscribe(seen.append)
        response = reply("bad input", isError=True)

        def request(method, params):
            return {"tools": [descriptor()]} if method == "tools/list" else response

        bridge = self.make_bridge(request, event_bus=bus)
        tool = bridge.refresh()[0]
        self.assertEqual(tool.invoke({}), response)
        self.assertEqual(seen[-1].kind, "mcp.call_finished")
        self.assertFalse(seen[-1].data["success"])
        self.assertIn("mcp.execution_failure", [event.kind for event in seen])
        response = {"error": {"code": -32601, "message": "missing"}}
        with self.assertRaises(MCPProtocolError):
            tool.invoke({})
        self.assertIn("mcp.protocol_failure", [event.kind for event in seen])

    def test_negotiated_notifications_remain_ephemeral_and_task_required_is_skipped(self):
        requests, seen, bus = [], [], EventBus()
        bus.subscribe(seen.append)

        def request(method, params):
            requests.append(method)
            return {"tools": [descriptor(), descriptor("task", execution={"taskSupport": "required"})]}

        caps = {
            "tools": {"listChanged": True},
            "resources": {"listChanged": True, "subscribe": True},
            "prompts": {"listChanged": True},
            "logging": {},
        }
        bridge = self.make_bridge(request, names=("echo", "task"), capabilities=caps, event_bus=bus)
        self.assertEqual(len(bridge.refresh()), 1)
        self.assertIn("mcp.tool_skipped", [event.kind for event in seen])
        for method, data in (
            ("notifications/resources/list_changed", {}),
            ("notifications/resources/updated", {"uri": "memo://private"}),
            ("notifications/prompts/list_changed", {}),
            ("notifications/message", {"level": "info", "data": "private log"}),
        ):
            bridge.on_notification(method, data)
        self.assertEqual(requests, ["tools/list"])
        self.assertEqual(
            [event.kind for event in seen[-4:]],
            ["mcp.resources.list_changed", "mcp.resources.updated", "mcp.prompts.list_changed", "mcp.message"],
        )
        none = self.make_bridge(lambda m, p: self.fail("request must not run"), capabilities={}, event_bus=bus)
        self.assertEqual(none.tools(), ())
        before = len(seen)
        none.on_notification("notifications/message", {"data": "not negotiated"})
        self.assertEqual(len(seen), before)

    def test_output_schema_and_validator_are_host_owned(self):
        schema = {"type": "object", "required": ["count"]}
        validations = []
        response = reply(structuredContent={"count": 1})

        def validate(output, data):
            validations.append((output, data))
            return data.get("count") == 1

        bridge = self.make_bridge(
            lambda m, p: {"tools": [descriptor(outputSchema=schema)]} if m == "tools/list" else response,
            validate_output=validate,
        )
        tool = bridge.refresh()[0]
        self.assertEqual(tool.output_schema, schema)
        tool.invoke({})
        self.assertEqual(validations, [(schema, {"count": 1})])
        response = reply(structuredContent={"count": 2})
        with self.assertRaises(MCPProtocolError):
            tool.invoke({})
        response = reply("execution error", isError=True)
        self.assertTrue(tool.invoke({})["isError"])

    def test_schema_snapshots_and_sdk_raised_protocol_errors(self):
        validations, seen, bus = [], [], EventBus()
        bus.subscribe(seen.append)

        class SDKProtocolError(Exception):
            def __init__(self, message):
                super().__init__(message)
                self.error = {"code": -32602, "message": "invalid arguments"}

        failure = None

        def request(method, params):
            if method == "tools/list":
                return {"tools": [descriptor(outputSchema={"properties": {"count": {"type": "integer"}}})]}
            if failure:
                raise failure
            return reply(structuredContent={"count": 1})

        bridge = self.make_bridge(request, event_bus=bus, validate_output=lambda schema, data: validations.append(schema))
        tool = bridge.refresh()[0]
        tool.schema["mutated"] = True
        tool.output_schema["mutated"] = True
        tool.invoke({})
        self.assertNotIn("mutated", validations[0])
        self.assertNotIn("mutated", bridge.tools()[0].schema)
        self.assertNotIn("mutated", bridge.tools()[0].output_schema)
        failure = SDKProtocolError("do not echo this detail")
        with self.assertRaises(SDKProtocolError):
            tool.invoke({})
        self.assertEqual(seen[-1].data["status"], "protocol_failure")


class AsyncBridgeTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_declarations_require_awaitable_host_results(self):
        tool = bind_mcp(
            "mock",
            [descriptor()],
            lambda name, arguments: reply(),
            permissions={"echo": frozenset()},
            keywords={"echo": ("echo",)},
            asynchronous=True,
        )[0]
        with self.assertRaises(TypeError):
            await tool.invoke({})
        bridge = MCPBridge(
            "mock",
            lambda method, params: {"tools": [descriptor()]},
            permissions={"echo": frozenset()},
            keywords={"echo": ("echo",)},
            capabilities={"tools": {}},
            asynchronous=True,
        )
        with self.assertRaises(TypeError):
            await bridge.arefresh()

        async def discovery():
            return {"tools": [descriptor()]}

        bridge.request = lambda method, params: discovery() if method == "tools/list" else reply()
        tool = (await bridge.arefresh())[0]
        seen = []
        bridge.event_bus.subscribe(seen.append)
        with self.assertRaises(TypeError):
            await tool.invoke({})
        self.assertEqual(seen[-1].kind, "mcp.call_finished")
        self.assertFalse(seen[-1].data["success"])

    async def test_sync_bridge_rejects_and_cancels_pending_async_work(self):
        pending = asyncio.get_running_loop().create_future()
        bridge = MCPBridge(
            "mock",
            lambda method, params: {"tools": [descriptor()]} if method == "tools/list" else pending,
            permissions={"echo": frozenset()},
            keywords={"echo": ("echo",)},
            capabilities={"tools": {}},
        )
        tool = bridge.refresh()[0]
        with self.assertRaises(TypeError):
            tool.invoke({})
        self.assertTrue(pending.cancelled())

    async def test_async_sdk_binding_and_bridge_work_in_running_event_loop(self):
        async def call(name, args):
            return SDKLike(reply(args["message"]))

        tool = bind_mcp(
            "mock",
            [descriptor()],
            call,
            permissions={"echo": frozenset()},
            keywords={"echo": ("echo",)},
            asynchronous=True,
        )[0]
        self.assertTrue(tool.asynchronous)
        self.assertEqual((await tool.invoke({"message": "async"}))["content"][0]["text"], "async")

        async def request(method, params):
            return SDKLike({"tools": [descriptor()]}) if method == "tools/list" else await call("echo", params["arguments"])

        bridge = MCPBridge(
            "mock",
            request,
            permissions={"echo": frozenset()},
            keywords={"echo": ("echo",)},
            capabilities={"tools": {}},
            asynchronous=True,
        )
        tools = await bridge.arefresh()
        self.assertEqual((await tools[0].invoke({"message": "bridge"}))["content"][0]["text"], "bridge")
        with self.assertRaises(TypeError):
            bridge.refresh()

    async def test_progress_routes_captured_scope_across_notification_tasks(self):
        bus, seen = EventBus(), []
        bus.subscribe(seen.append)
        started, release = asyncio.Event(), asyncio.Event()
        params_by_message = {}

        async def request(method, params):
            if method == "tools/list":
                return {"tools": [descriptor()]}
            params_by_message[params["arguments"]["message"]] = params
            if len(params_by_message) == 2:
                started.set()
            await release.wait()
            return reply()

        bridge = MCPBridge(
            "mock",
            request,
            permissions={"echo": frozenset()},
            keywords={"echo": ("echo",)},
            capabilities={"tools": {}},
            asynchronous=True,
            event_bus=bus,
        )
        tool = (await bridge.arefresh())[0]
        scopes = [Scope("alice", "project", "a"), Scope("bob", "project", "b")]

        async def invoke(index):
            with invocation(scopes[index], "plan", f"call-{index}"):
                return await tool.invoke({"message": str(index)})

        tasks = [asyncio.create_task(invoke(i)) for i in range(2)]
        await started.wait()
        tokens = [params_by_message[str(i)]["_meta"]["progressToken"] for i in range(2)]
        self.assertEqual(len(set(tokens)), 2)
        self.assertIsNone(current_invocation())
        for i, token in enumerate(tokens):
            bridge.on_notification(
                "notifications/progress",
                {"progressToken": token, "progress": 1, "total": 2, "message": "ephemeral progress"},
            )
            event = seen[-1]
            self.assertEqual((event.kind, event.scope, event.correlation_id), ("mcp.progress", scopes[i], f"call-{i}"))
        with self.assertRaises(MCPProtocolError):
            bridge.on_notification("notifications/progress", {"progressToken": tokens[0], "progress": 1})
        release.set()
        await asyncio.gather(*tasks)
        count = len(seen)
        bridge.on_notification("notifications/progress", {"progressToken": tokens[0], "progress": 2})
        self.assertEqual(len(seen), count)

    async def test_cancellation_cleans_active_progress_and_preserves_cancelled_error(self):
        bus, seen = EventBus(), []
        bus.subscribe(seen.append)
        started = asyncio.Event()
        captured = {}

        async def request(method, params):
            if method == "tools/list":
                return {"tools": [descriptor()]}
            captured.update(params)
            started.set()
            await asyncio.Event().wait()

        bridge = MCPBridge(
            "mock",
            request,
            permissions={"echo": frozenset()},
            keywords={"echo": ("echo",)},
            capabilities={"tools": {}},
            asynchronous=True,
            event_bus=bus,
        )
        tool = (await bridge.arefresh())[0]
        task = asyncio.create_task(tool.invoke({}))
        await started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual([event.kind for event in seen[-2:]], ["mcp.cancelled", "mcp.call_finished"])
        self.assertEqual(seen[-1].data["status"], "cancelled")
        count = len(seen)
        bridge.on_notification("notifications/progress", {"progressToken": captured["_meta"]["progressToken"], "progress": 1})
        self.assertEqual(len(seen), count)


class EventTests(unittest.TestCase):
    def test_call_finish_and_progress_persist_tool_operation_and_generated_token(self):
        scope = Scope("alice", "project", "session")
        with tempfile.TemporaryDirectory() as directory, Store(Path(directory) / "events.sqlite") as store:
            bus = EventBus()
            bus.subscribe(SQLiteEventSink(store, scope))
            bridge = None

            def request(method, params):
                if method == "tools/list":
                    return {"tools": [descriptor()]}
                bridge.on_notification(
                    "notifications/progress",
                    {
                        "progressToken": params["_meta"]["progressToken"],
                        "progress": 1,
                        "total": 2,
                        "name": "forged",
                        "operation": "resources/read",
                        "message": "RAW_PROGRESS",
                    },
                )
                return reply("RAW_RESULT")

            bridge = MCPBridge(
                "mock",
                request,
                permissions={"echo": frozenset()},
                keywords={"echo": ("echo",)},
                capabilities={"tools": {}},
                event_bus=bus,
            )
            tool = bridge.refresh()[0]
            with invocation(scope, "plan", "call-1"):
                tool.invoke({"message": "RAW_ARGUMENTS"})
            rows = store.events(scope)
            payloads = {row["kind"]: row["payload"] for row in rows}
            for kind in ("mcp.call_started", "mcp.progress", "mcp.call_finished"):
                self.assertEqual(payloads[kind]["name"], "echo")
                self.assertEqual(payloads[kind]["operation"], "tools/call")
                self.assertEqual(payloads[kind]["correlation_id"], "call-1")
                self.assertRegex(payloads[kind]["progressToken"], r"^[0-9a-f]{32}$")
            self.assertEqual(payloads["mcp.progress"]["progressToken"], payloads["mcp.call_finished"]["progressToken"])
            self.assertEqual(payloads["mcp.inventory_refreshed"]["operation"], "tools/list")
            self.assertNotIn("RAW_", json.dumps(rows))
            bus.publish(
                "mcp.fake",
                "mcp:mock",
                {"name": "memo://RAW_URI", "operation": "resources/read", "progressToken": "RAW_TOKEN"},
                scope=scope,
            )
            saved = store.events(scope)[0]["payload"]
            self.assertNotIn("name", saved)
            self.assertNotIn("operation", saved)
            self.assertNotIn("progressToken", saved)

    def test_notifications_are_persisted_only_as_metadata_in_host_scope(self):
        scope, other = Scope("alice", "project", "s1"), Scope("mallory", "project", "s1")
        with tempfile.TemporaryDirectory() as directory, Store(Path(directory) / "events.sqlite") as store:
            bus, sink, seen = EventBus(), SQLiteEventSink(store, scope), []
            bus.subscribe(sink)
            bus.subscribe(seen.append)
            bridge = MCPBridge(
                "mock",
                lambda m, p: self.fail("notification must not make a request"),
                permissions={},
                keywords={},
                event_bus=bus,
                capabilities={
                    "resources": {"subscribe": True, "listChanged": True},
                    "logging": {},
                    "prompts": {"listChanged": True},
                },
            )
            bridge.on_notification("notifications/resources/updated", {"uri": "memo://RAW_URI", "scope": other.__dict__})
            bridge.on_notification(
                "notifications/message", {"level": "error", "data": "RAW_LOG", "_meta": {"scope": other.__dict__}}
            )
            bridge.on_notification("notifications/prompts/list_changed", {})
            self.assertEqual(len(store.events(scope)), 3)
            self.assertEqual(store.events(other), [])
            self.assertNotIn("RAW_", json.dumps(store.events(scope)))
            self.assertEqual(seen[1].data["data"], "RAW_LOG")
            self.assertEqual(len(sink.errors), 0)

    def test_observer_failure_is_bounded_isolated_and_unsubscribe_works(self):
        bus, seen = EventBus(), []

        def broken(event):
            raise RuntimeError("raw sensitive exception must not be stored")

        bus.subscribe(broken)
        unsubscribe = bus.subscribe(seen.append, kinds=frozenset({"wanted"}))
        for _ in range(140):
            bus.publish("wanted", "host", {"count": 1})
        self.assertEqual(len(bus.errors), 128)
        self.assertEqual(len(seen), 140)
        self.assertNotIn("sensitive", json.dumps(list(bus.errors)))
        unsubscribe()
        unsubscribe()
        bus.publish("wanted", "host", {})
        self.assertEqual(len(seen), 140)

    def test_invocation_context_and_sink_persist_metadata_only_exact_scope(self):
        scope, other = Scope("alice", "project", "s1"), Scope("alice", "project", "s2")
        with tempfile.TemporaryDirectory() as directory, Store(Path(directory) / "events.sqlite") as store:
            bus, sink = EventBus(), SQLiteEventSink(store, default_scope=other)
            bus.subscribe(sink)
            with invocation(scope, "plan-1", "call-1"):
                self.assertEqual(current_invocation().scope, scope)
                event = bus.publish(
                    "mcp.progress",
                    "mcp:mock",
                    {
                        "progress": 2,
                        "total": 3,
                        "duration_ms": 1,
                        "arguments": {"secret": "RAW_ARGUMENTS"},
                        "result": "RAW_RESULT",
                        "uri": "memo://RAW_URI",
                        "message": "RAW_PROGRESS",
                        "data": "RAW_LOG",
                        "scope": {"user": "forged"},
                    },
                )
            self.assertIsNone(current_invocation())
            self.assertEqual(event.scope, scope)
            saved = store.events(scope)[0]["payload"]
            self.assertEqual(saved["correlation_id"], "call-1")
            self.assertEqual(saved["progress"], 2)
            self.assertNotIn("RAW_", json.dumps(saved))
            self.assertEqual(store.events(other), [])
            store.close()
            bus.publish("sink_fault", "host", {}, scope=scope)
            self.assertEqual(sink.errors[-1]["error_type"], "RuntimeError")

    def test_subscriber_fault_cannot_mask_completed_tool_effects(self):
        bus, effects = EventBus(), []
        bus.subscribe(lambda event: (_ for _ in ()).throw(RuntimeError("observer fault")))

        def request(method, params):
            if method == "tools/list":
                return {"tools": [descriptor()]}
            effects.append(params["arguments"])
            return reply("completed")

        bridge = MCPBridge(
            "mock",
            request,
            permissions={"echo": frozenset()},
            keywords={"echo": ("echo",)},
            capabilities={"tools": {}},
            event_bus=bus,
        )
        result = bridge.refresh()[0].invoke({"operation": "mock write"})
        self.assertEqual(result, reply("completed"))
        self.assertEqual(effects, [{"operation": "mock write"}])
        self.assertGreater(len(bus.errors), 0)


if __name__ == "__main__":
    unittest.main()


from mcp_guard_loader import no_real_memory_or_mcp  # noqa: E402, F401  (autouse guard)

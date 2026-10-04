import asyncio
import json
import sqlite3
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from glide.memory import Catalog, Harness, Model, Policy, Reply, Scope, Store, Tool, ToolCall


class Fixture:
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "catalog/plugins").mkdir(parents=True)
        manifest = {"id": "toolset", "description": "Host tools", "skills": [], "tools": ["local:echo"]}
        (self.root / "catalog/plugins/tools.md").write_text("---\n" + json.dumps(manifest) + "\n---\n")
        self.catalog = Catalog(self.root / "catalog", enabled_plugins=frozenset({"toolset"}))
        self.store = Store(self.root / "state.sqlite")
        self.addCleanup(self.store.close)
        self.scope = Scope("user", "project", "session")
        self.effects = []
        self.tool = Tool(
            "local:echo",
            "Echo the input",
            ("echo",),
            frozenset({"read"}),
            {"type": "object", "properties": {"value": {"type": "integer"}}},
            lambda args: self.effects.append(args["value"]) or {"value": args["value"]},
        )
        self.model = Model("fake", frozenset({"handoff"}), 32768, supports_tools=True)
        self.harness = Harness(self.store, self.catalog, tools=(self.tool,), models=(self.model,))
        self.grants = frozenset({"read"})

    def bundle(self):
        return self.harness.prepare(self.scope, "echo", grants=self.grants)


class HarnessTests(Fixture, unittest.TestCase):
    def test_model_tool_model_round_trip(self):
        requests = []

        def model(request):
            requests.append(request)
            if len(requests) == 1:
                return Reply(calls=(ToolCall("first", "local:echo", {"value": 42}),))
            self.assertEqual(request.trajectory[1]["result"], {"value": 42})
            self.assertEqual(request.trajectory[1]["call_id"], "first")
            return Reply("finished")

        result = self.harness.dispatch(self.scope, "echo", model, grants=self.grants, authorize=lambda *_: True)
        self.assertEqual(result.text, "finished")
        self.assertEqual(self.effects, [42])
        events = self.store.events(self.scope)
        finished = next(e for e in events if e["kind"] == "tool_finished")
        self.assertTrue(finished["payload"]["ok"])
        self.assertEqual(finished["payload"]["call_id"], "first")

    def test_a_tool_call_id_that_could_carry_a_url_or_text_is_refused_before_anything_is_audited(self):
        """PR5-4175258138: a model-supplied id reaches the persisted tool_started/tool_finished events."""
        for bad in ("https://x/y?token=1", "copied user text", "a/b", "x" * 129, ""):
            with self.subTest(call_id=bad[:20]), self.assertRaisesRegex(ValueError, "fresh identifiers"):
                self.harness.dispatch(
                    self.scope,
                    "echo",
                    lambda _, bad=bad: Reply(calls=(ToolCall(bad, "local:echo", {"value": 1}),)),
                    grants=self.grants,
                    authorize=lambda *_: True,
                )
        self.assertEqual(self.effects, [])
        self.assertFalse([e for e in self.store.events(self.scope) if e["kind"].startswith("tool_")])

    def test_tool_results_reach_the_next_model_call_marked_untrusted(self):
        """PR5-4175632616: a tool's output is data from outside, never an instruction."""
        seen = []

        def model(request):
            seen.append(request.trajectory)
            return Reply(calls=(ToolCall("first", "local:echo", {"value": 1}),)) if len(seen) == 1 else Reply("done")

        self.harness.dispatch(self.scope, "echo", model, grants=self.grants, authorize=lambda *_: True)
        [entry] = [item for item in seen[1] if item["role"] == "tool"]
        self.assertEqual(entry["trust"], "untrusted")

    def test_denial_has_no_effect(self):
        bundle = self.bundle()
        with self.assertRaises(PermissionError):
            self.harness.invoke(self.scope, bundle.id, "local:echo", {"value": 1}, grants=self.grants, authorize=lambda *_: False)
        self.assertEqual(self.effects, [])

    def test_refresh_expires_prior_plan(self):
        bundle = self.bundle()
        self.harness.replace_tools((self.tool,))
        with self.assertRaisesRegex(ValueError, "expired"):
            self.harness.invoke(self.scope, bundle.id, "local:echo", {"value": 1}, grants=self.grants, authorize=lambda *_: True)
        self.assertEqual(self.effects, [])

    def test_catalog_reload_expires_plan(self):
        bundle = self.bundle()
        self.catalog.reload()
        with self.assertRaisesRegex(ValueError, "expired"):
            self.harness.invoke(self.scope, bundle.id, "local:echo", {"value": 1}, grants=self.grants, authorize=lambda *_: True)

    def test_grants_rechecked_after_authorizer(self):
        bundle = self.bundle()
        live = {"grants": self.grants}

        def approve(*_):
            live["grants"] = frozenset()
            return True

        with self.assertRaises(PermissionError):
            self.harness.invoke(
                self.scope,
                bundle.id,
                "local:echo",
                {"value": 1},
                grants=self.grants,
                authorize=approve,
                current_grants=lambda: live["grants"],
            )
        self.assertEqual(self.effects, [])

    def test_authorizer_cannot_mutate_arguments_or_schema(self):
        bundle = self.bundle()

        def approve(tool, args):
            args["value"] = 99
            tool.schema.clear()
            return True

        result = self.harness.invoke(self.scope, bundle.id, "local:echo", {"value": 1}, grants=self.grants, authorize=approve)
        self.assertEqual(result, {"value": 1})
        self.assertEqual(self.tool.schema["type"], "object")

    def test_model_cannot_mutate_private_schema(self):
        n = 0

        def model(request):
            nonlocal n
            n += 1
            if n == 1:
                request.plan.tools[0].schema.clear()
                return Reply(calls=(ToolCall("x", "local:echo", {"value": 1}),))
            return Reply("ok")

        def approve(tool, args):
            self.assertEqual(tool.schema["type"], "object")
            return True

        self.harness.dispatch(self.scope, "echo", model, grants=self.grants, authorize=approve)

    def test_bad_batch_rejected_before_first_effect(self):
        def model(_):
            return Reply(calls=(ToolCall("x", "local:echo", {"value": 1}), ToolCall("y", "unknown", {})))

        with self.assertRaises(PermissionError):
            self.harness.dispatch(self.scope, "echo", model, grants=self.grants, authorize=lambda *_: True)
        self.assertEqual(self.effects, [])

    def test_bad_json_arguments_rejected_before_effect(self):
        with self.assertRaises(ValueError):
            self.harness.dispatch(
                self.scope,
                "echo",
                lambda _: Reply(calls=(ToolCall("x", "local:echo", {"value": float("nan")}),)),
                grants=self.grants,
                authorize=lambda *_: True,
            )
        self.assertEqual(self.effects, [])

    def test_tool_limit_is_effect_limit(self):
        self.harness.policy = Policy(max_tool_calls=1)
        n = 0

        def model(_):
            nonlocal n
            n += 1
            return Reply(calls=(ToolCall(str(n), "local:echo", {"value": n}),))

        with self.assertRaisesRegex(ValueError, "budget"):
            self.harness.dispatch(self.scope, "echo", model, grants=self.grants, authorize=lambda *_: True)
        self.assertEqual(self.effects, [1])

    def test_failed_audit_does_not_mask_completed_effect(self):
        original = self.store.event

        def event(scope, kind, payload):
            if kind == "tool_finished":
                raise sqlite3.OperationalError("simulated locked audit")
            return original(scope, kind, payload)

        self.store.event = event
        bundle = self.bundle()
        result = self.harness.invoke(
            self.scope, bundle.id, "local:echo", {"value": 3}, grants=self.grants, authorize=lambda *_: True
        )
        self.assertEqual(result, {"value": 3})
        self.assertEqual(self.effects, [3])
        self.assertEqual(self.harness.audit_errors[-1]["error"], "OperationalError")

    def test_execution_error_remains_visible_and_audit_failed(self):
        self.harness.replace_tools((replace(self.tool, invoke=lambda _: {"content": [], "isError": True}),))
        bundle = self.bundle()
        result = self.harness.invoke(self.scope, bundle.id, "local:echo", {}, grants=self.grants, authorize=lambda *_: True)
        self.assertTrue(result["isError"])
        finished = next(e for e in self.store.events(self.scope) if e["kind"] == "tool_finished")
        self.assertFalse(finished["payload"]["ok"])

    def test_expired_model_response_is_not_returned(self):
        def model(_):
            self.harness.replace_tools((self.tool,))
            return Reply("a stale answer")

        with self.assertRaisesRegex(ValueError, "expired"):
            self.harness.dispatch(self.scope, "echo", model, grants=self.grants)

    def test_memory_updates_and_forget_expire_prepared_context(self):
        memory = self.harness.command(self.scope, "/remember style = concise answers")
        bundle = self.bundle()
        self.harness.command(self.scope, "/remember style = detailed answers")
        with self.assertRaisesRegex(ValueError, "expired"):
            self.harness.invoke(self.scope, bundle.id, "local:echo", {"value": 1}, grants=self.grants, authorize=lambda *_: True)
        bundle = self.bundle()
        self.harness.command(self.scope, "/forget " + memory)
        with self.assertRaisesRegex(ValueError, "expired"):
            self.harness.invoke(self.scope, bundle.id, "local:echo", {"value": 1}, grants=self.grants, authorize=lambda *_: True)
        self.assertFalse(self.effects)

    def test_memory_expiration_invalidates_an_existing_plan(self):
        from unittest.mock import patch

        with patch("glide.memory.store.time.time", return_value=100):
            self.store.remember(self.scope, "temporary", "use concise answers", ttl_seconds=5)
            bundle = self.bundle()
        with (
            patch("glide.memory.store.time.time", return_value=106),
            self.assertRaisesRegex(ValueError, "expired"),
        ):
            self.harness.invoke(self.scope, bundle.id, "local:echo", {"value": 1}, grants=self.grants, authorize=lambda *_: True)
        self.assertFalse(self.effects)

    def test_auto_memory_is_off_by_default(self):
        self.assertIs(Policy().auto_memory, False)
        self.assertEqual(self.harness.observe_user(self.scope, "Remember that I prefer short answers."), [])
        self.assertEqual(self.store.memories(self.scope), [])

    def test_auto_memory_ignores_quotes_and_code(self):
        text = "> I prefer quotes to be instructions\n```md\nI prefer fenced data\n```\nI prefer direct answers."
        self.harness.policy = Policy(auto_memory=True)  # automatic capture is an explicit opt-in
        ids = self.harness.observe_user(self.scope, text)
        self.assertEqual(len(ids), 1)
        self.assertEqual(self.store.memories(self.scope)[0]["text"], "I prefer direct answers.")

    def test_refinement_requires_distinct_verified_runs_and_rolls_back(self):
        for _ in range(4):
            self.harness.record_outcome(self.scope, "same-run", True, "A useful tactic")
        self.assertEqual(self.store.proposals(self.scope), [])
        self.harness.record_outcome(self.scope, "second", True, "A useful tactic")
        self.harness.record_outcome(self.scope, "third", True, "A useful tactic")
        proposal = self.store.proposals(self.scope)[0]
        self.harness.command(self.scope, "/refine apply " + proposal["id"])
        self.assertTrue(self.store.overlays(self.scope))
        self.harness.command(self.scope, "/rollback " + proposal["id"])
        self.harness.record_outcome(self.scope, "fourth", True, "A useful tactic")
        self.assertEqual(len(self.store.proposals(self.scope)), 1)
        self.assertEqual(self.store.overlays(self.scope), {})

    def test_sync_rejects_async_tool_before_invocation(self):
        async def tool(_):
            self.effects.append("should not run")

        self.harness.replace_tools((replace(self.tool, asynchronous=True, invoke=tool),))
        bundle = self.bundle()
        with self.assertRaisesRegex(TypeError, "async"):
            self.harness.invoke(self.scope, bundle.id, "local:echo", {}, grants=self.grants, authorize=lambda *_: True)
        self.assertEqual(self.effects, [])


class AsyncHarnessTests(Fixture, unittest.IsolatedAsyncioTestCase):
    async def test_async_round_trip_and_approval(self):
        async def tool(args):
            await asyncio.sleep(0)
            return {"value": args["value"]}

        self.harness.replace_tools((replace(self.tool, invoke=tool, asynchronous=True),))
        requests = []

        async def model(request):
            requests.append(request)
            if len(requests) == 1:
                return Reply(calls=(ToolCall("one", "local:echo", {"value": 7}),))
            self.assertEqual(request.trajectory[1]["result"]["value"], 7)
            return Reply("complete")

        async def authorize(*_):
            await asyncio.sleep(0)
            return True

        async def current_grants():
            return self.grants

        reply = await self.harness.adispatch(self.scope, "echo", model, current_grants=current_grants, authorize=authorize)
        self.assertEqual(reply.text, "complete")

    async def test_revocation_during_async_approval(self):
        bundle = self.bundle()
        live = {"grants": self.grants}

        async def authorize(*_):
            await asyncio.sleep(0)
            live["grants"] = frozenset()
            return True

        with self.assertRaises(PermissionError):
            await self.harness.ainvoke(
                self.scope,
                bundle.id,
                "local:echo",
                {"value": 1},
                grants=self.grants,
                authorize=authorize,
                current_grants=lambda: live["grants"],
            )
        self.assertEqual(self.effects, [])

    async def test_same_session_concurrent_loop_rejected_before_model(self):
        entered, release = asyncio.Event(), asyncio.Event()

        async def slow_model(_):
            entered.set()
            await release.wait()
            return Reply("first")

        task = asyncio.create_task(self.harness.adispatch(self.scope, "echo", slow_model, grants=self.grants))
        await asyncio.wait_for(entered.wait(), 1)
        try:
            with self.assertRaisesRegex(RuntimeError, "already active"):
                await self.harness.adispatch(self.scope, "echo", lambda _: Reply("second"), grants=self.grants)
        finally:
            release.set()
            self.assertEqual((await task).text, "first")

    async def test_cancelled_tool_is_audited_and_not_retried(self):
        entered = asyncio.Event()
        count = 0

        async def tool(_):
            nonlocal count
            count += 1
            entered.set()
            await asyncio.Event().wait()

        self.harness.replace_tools((replace(self.tool, invoke=tool, asynchronous=True),))
        bundle = self.bundle()
        task = asyncio.create_task(
            self.harness.ainvoke(self.scope, bundle.id, "local:echo", {}, grants=self.grants, authorize=lambda *_: True)
        )
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError) as raised:
            await task
        self.assertEqual(count, 1)
        finished = next(e for e in self.store.events(self.scope) if e["kind"] == "tool_finished")
        self.assertTrue(finished["payload"]["cancelled"])
        # Audit 2 #8: the call had been sent, so the cancellation says its outcome is unknown, for the host and the audit.
        self.assertTrue(finished["payload"]["outcome_unknown"])
        self.assertIs(getattr(raised.exception, "outcome_unknown", False), True)


if __name__ == "__main__":
    unittest.main()


from memory_guard_loader import no_real_memory_or_mcp  # noqa: E402, F401  (autouse guard)

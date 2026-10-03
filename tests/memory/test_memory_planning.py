import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from threading import Thread

from glide.memory.catalog import MAX_BODY_BYTES, Catalog
from glide.memory.contracts import Model, Policy, Scope, Tool
from glide.memory.planning import plan


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "catalog"
        (self.root / "skills").mkdir(parents=True)
        (self.root / "plugins").mkdir()
        self.scope = Scope("user", "project", "session")
        self.calls = []
        self.policy = Policy(context_tokens=10000, output_reserve=100)
        self.large = Model("large", frozenset({"write"}), 100000, priority=1, supports_tools=True)

    def manifest(self, kind, identifier, data, body="Instructions."):
        path = self.root / kind / f"{identifier}.md"
        path.write_text("---\n" + json.dumps(data) + "\n---\n" + body, encoding="utf-8")
        return path

    def skill(self, identifier="mail", deps=("lookup", "send"), body="Look up the contact, then send email."):
        return self.manifest(
            "skills",
            identifier,
            {
                "id": identifier,
                "description": "Email workflow",
                "keywords": ["email"],
                "stages": ["write"],
                "tools": list(deps),
            },
            body,
        )

    def catalog(self, *, enabled=True, skills=("mail",), tools=("lookup", "send")):
        self.manifest(
            "plugins",
            "bundle",
            {
                "id": "bundle",
                "description": "Explicit capabilities",
                "skills": list(skills),
                "tools": list(tools),
            },
        )
        return Catalog(self.root, enabled_plugins=frozenset({"bundle"}) if enabled else frozenset())

    def tool(self, identifier, keywords=("email",), permissions=frozenset({"mail"})):
        return Tool(
            identifier,
            "Fake tool",
            keywords,
            permissions,
            {"type": "object", "properties": {"text": {"type": "string"}}},
            lambda arguments: self.calls.append((identifier, arguments)),
        )

    def route(
        self,
        catalog,
        *,
        goal="email",
        tools=(),
        models=None,
        memories=None,
        overlays=None,
        policy=None,
        grants=frozenset({"mail"}),
        stage="write",
    ):
        return plan(
            self.scope,
            goal,
            stage=stage,
            memories=memories or [],
            overlays=overlays or {},
            catalog=catalog,
            tools=tools,
            models=(self.large,) if models is None else models,
            grants=grants,
            policy=policy or self.policy,
            base_tokens=10,
        )

    def memory(self, identifier, text, *, kind="preference", confidence=0.9, updated=100):
        return {
            "id": identifier,
            "text": text,
            "kind": kind,
            "source": "user",
            "confidence": confidence,
            "updated_at": updated,
        }

    def test_stable_preferences_need_no_goal_overlap_and_use_numeric_recency(self):
        catalog = Catalog(self.root)
        result = self.route(
            catalog,
            memories=[
                self.memory("old", "Use concise sentences", updated=100),
                self.memory("new", "Use concrete language", updated=200),
                self.memory("fact", "The sun is yellow", kind="fact"),
            ],
        )
        self.assertEqual(result.memory_ids, ("new", "old"))
        self.assertIn("lower-trust data", result.context)
        self.assertIn("remain authoritative", result.context)

    def test_model_must_fit_full_skill_bundle_not_one_direct_dependency(self):
        self.skill()
        catalog = self.catalog()
        tools = (self.tool("lookup"), self.tool("send"))
        baseline = self.route(catalog, tools=tools)
        small = replace(
            self.large,
            id="small",
            priority=0,
            context_tokens=10 + self.policy.output_reserve + baseline.token_upper_bound - 1,
        )
        result = self.route(catalog, tools=tools, models=(small, self.large))
        self.assertEqual(result.model_id, "large")
        self.assertEqual(result.skill_ids, ("mail",))
        self.assertEqual([tool.id for tool in result.tools], ["lookup", "send"])
        self.assertTrue(any("complete required bundle budget" in reason for reason in result.reasons))
        self.assertEqual(self.calls, [])

    def test_no_viable_full_bundle_returns_no_route(self):
        self.skill()
        catalog = self.catalog()
        result = self.route(
            catalog, tools=(self.tool("lookup"), self.tool("send")), models=(replace(self.large, context_tokens=200),)
        )
        self.assertIsNone(result.model_id)
        self.assertEqual((result.context, result.tools, result.skill_ids), ("", (), ()))
        self.assertTrue(any("no viable complete tool/skill route" in reason for reason in result.reasons))

    def test_required_tools_survive_preference_budget_pressure(self):
        catalog = self.catalog(skills=(), tools=("lookup",))
        tool = self.tool("lookup")
        baseline = self.route(catalog, tools=(tool,))
        policy = replace(self.policy, context_tokens=baseline.token_upper_bound + 20)
        result = self.route(
            catalog, tools=(tool,), policy=policy, memories=[self.memory("large-pref", "Use concise sentences. " * 100)]
        )
        self.assertEqual([item.id for item in result.tools], ["lookup"])
        self.assertEqual(result.memory_ids, ())
        self.assertTrue(any("memory large-pref: excluded context budget" in reason for reason in result.reasons))

    def test_count_budget_blocks_complete_anchor(self):
        self.skill()
        catalog = self.catalog()
        for policy in (
            replace(self.policy, max_tools=1),
            replace(self.policy, max_skills=0),
            replace(self.policy, max_tool_calls=0),
        ):
            with self.subTest(policy=policy):
                result = self.route(catalog, tools=(self.tool("lookup"), self.tool("send")), policy=policy)
                self.assertIsNone(result.model_id)
                self.assertEqual(result.tools, ())
                self.assertTrue(any("policy count budget" in reason for reason in result.reasons))

    def test_tool_less_model_is_skipped_for_required_tool(self):
        catalog = self.catalog(skills=(), tools=("lookup",))
        preferred = replace(self.large, id="plain", priority=0, supports_tools=False)
        result = self.route(catalog, tools=(self.tool("lookup"),), models=(preferred, self.large))
        self.assertEqual(result.model_id, "large")
        self.assertTrue(any("required tool support" in reason for reason in result.reasons))

    def test_local_model_needs_grant(self):
        catalog = Catalog(self.root)
        local = replace(self.large, id="local", priority=0, local=True)
        result = self.route(catalog, models=(local, self.large), grants=frozenset())
        self.assertEqual(result.model_id, "large")
        allowed = self.route(catalog, models=(local, self.large), grants=frozenset({"model:local"}))
        self.assertEqual(allowed.model_id, "local")

    def test_plugins_and_skills_need_explicit_enablement(self):
        self.skill()
        catalog = self.catalog(enabled=False)
        tools = (self.tool("lookup"), self.tool("send"))
        disabled = self.route(catalog, tools=tools)
        self.assertEqual((disabled.skill_ids, disabled.tools), ((), ()))
        enabled_skill = Catalog(self.root, enabled_skills=frozenset({"mail"}))
        selected = self.route(enabled_skill, tools=tools)
        self.assertEqual(selected.skill_ids, ("mail",))
        self.assertEqual(enabled_skill.tool_ids(), frozenset({"lookup", "send"}))

    def test_missing_or_unpermitted_skill_dependencies_do_not_force_tool_models(self):
        self.skill()
        catalog = self.catalog(tools=("lookup",))
        plain = replace(self.large, supports_tools=False)
        # Nonmatching keywords prevent a separate direct-tool need.
        result = self.route(
            catalog,
            tools=(self.tool("lookup", keywords=()), self.tool("send", keywords=())),
            models=(plain,),
            grants=frozenset({"mail"}),
        )
        self.assertEqual(result.model_id, "large")
        self.assertEqual((result.skill_ids, result.tools), ((), ()))
        self.assertTrue(any("required tools" in reason for reason in result.reasons))
        denied = self.route(
            self.catalog(),
            tools=(self.tool("lookup", keywords=()), self.tool("send", keywords=())),
            models=(plain,),
            grants=frozenset(),
        )
        self.assertEqual(denied.model_id, "large")
        self.assertEqual(denied.tools, ())

    def test_overlays_cannot_enable_skills_and_prompt_overlays_need_relevance(self):
        self.skill(deps=(), body="ORIGINAL_BODY")
        catalog = self.catalog(tools=())
        result = self.route(
            catalog,
            overlays={
                "skill:mail": "REPLACED_BODY",
                "skill:disabled": "DISABLED_SENTINEL",
                "prompt:relevant": "Give email guidance",
                "prompt:unrelated": "Study astronomy orbits",
            },
        )
        self.assertIn("REPLACED_BODY", result.context)
        self.assertNotIn("ORIGINAL_BODY", result.context)
        self.assertNotIn("DISABLED_SENTINEL", result.context)
        self.assertIn("Give email guidance", result.context)
        self.assertNotIn("Study astronomy orbits", result.context)
        self.assertEqual(result.skill_ids, ("mail",))

    def test_catalog_reload_revision_is_atomic_and_failed_reload_keeps_snapshot(self):
        self.skill(deps=(), body="first body")
        catalog = self.catalog(tools=())
        skills, ids, revision = catalog.snapshot_with_revision()
        skills[0]["body"] = "external mutation"
        self.assertEqual(catalog.skills()[0]["body"], "first body")
        bad = self.root / "skills" / "bad.md"
        bad.write_text("---\nnot JSON\n---\nbody", encoding="utf-8")
        with self.assertRaises(ValueError):
            catalog.reload()
        self.assertEqual(catalog.revision, revision)
        self.assertEqual(catalog.tool_ids(), ids)
        bad.unlink()
        self.skill(deps=(), body="second body")
        catalog.reload()
        self.assertEqual(catalog.revision, revision + 1)
        self.assertEqual(catalog.snapshot_with_revision()[2], catalog.revision)
        self.assertEqual(self.route(catalog).catalog_revision, catalog.revision)

    def test_concurrent_reloads_publish_distinct_revisions(self):
        catalog = Catalog(self.root)
        initial = catalog.revision
        errors = []

        def reload_repeatedly():
            try:
                for _ in range(10):
                    catalog.reload()
                    self.assertEqual(catalog.snapshot_with_revision()[:2], ([], frozenset()))
            except Exception as error:
                errors.append(error)

        threads = [Thread(target=reload_repeatedly) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(errors, [])
        self.assertEqual(catalog.revision, initial + 40)

    def test_malformed_frontmatter_unknown_fields_duplicate_ids_and_oversize_fail(self):
        path = self.skill(deps=())
        valid = path.read_text(encoding="utf-8")
        malformed = (
            "---\nid: yaml\n---\nbody",
            valid.replace('"id": "mail"', '"id": "mail", "id": "again"'),
            valid.replace('"description":', '"entrypoint": "process", "description":'),
            valid.replace('"keywords": ["email"]', '"keywords": 4'),
            valid.partition("\n---\n")[0] + "\n---\n" + "x" * (MAX_BODY_BYTES + 1),
        )
        for content in malformed:
            with self.subTest(content=content[:100]):
                path.write_text(content, encoding="utf-8")
                with self.assertRaises(ValueError):
                    Catalog(self.root)
        path.write_text(valid, encoding="utf-8")
        (self.root / "skills" / "duplicate.md").write_text(valid, encoding="utf-8")
        with self.assertRaises(ValueError):
            Catalog(self.root)

    def test_catalog_rejects_symlink_file_and_directory_escapes(self):
        outside = Path(self.temp.name) / "outside.md"
        outside.write_text("---\n{}\n---\nbody", encoding="utf-8")
        link = self.root / "skills" / "escape.md"
        link.symlink_to(outside)
        with self.assertRaises(ValueError):
            Catalog(self.root)
        link.unlink()
        (self.root / "skills").rmdir()
        (self.root / "skills").symlink_to(Path(self.temp.name), target_is_directory=True)
        with self.assertRaises(ValueError):
            Catalog(self.root)

    def test_json_budget_counts_escaping_envelopes_and_snapshots_schemas(self):
        self.skill(deps=("lookup",), body='中文 "quoted" \\ text')
        catalog = self.catalog(tools=("lookup",))
        tool = replace(self.tool("lookup"), schema={"type": "object", "description": '中文 "quoted" \\ text'})
        result = self.route(catalog, tools=(tool,))
        expected = len(
            json.dumps(
                {"context": result.context, "tools": result.tool_definitions()},
                ensure_ascii=True,
                sort_keys=True,
                allow_nan=False,
            ).encode("utf-8")
        )
        self.assertEqual(result.token_upper_bound, expected)
        self.assertLessEqual(result.token_upper_bound, self.policy.context_tokens)
        self.assertLessEqual(10 + result.token_upper_bound + self.policy.output_reserve, self.large.context_tokens)
        tool.schema["description"] = "registration mutated"
        definitions = result.tool_definitions()
        definitions[0]["inputSchema"]["description"] = "display mutated"
        self.assertIn("中文", result.tools[0].schema["description"])
        exact = self.route(catalog, tools=(result.tools[0],), policy=replace(self.policy, context_tokens=expected))
        self.assertEqual(exact.model_id, "large")
        too_small = self.route(catalog, tools=(result.tools[0],), policy=replace(self.policy, context_tokens=expected - 1))
        self.assertIsNone(too_small.model_id)

    def test_classify_does_not_read_catalog_or_inventory(self):
        class UnreadableCatalog:
            def snapshot_with_revision(self):
                raise AssertionError("classification must not read catalog")

        result = self.route(UnreadableCatalog(), stage="classify", tools=(object(),), models=(object(),))
        self.assertEqual((result.model_id, result.context, result.tools, result.skill_ids), (None, "", (), ()))

    def test_async_and_output_schema_descriptors_are_validated_and_snapshotted(self):
        catalog = self.catalog(skills=(), tools=("lookup",))
        tool = replace(
            self.tool("lookup"),
            asynchronous=True,
            output_schema={"type": "object", "properties": {"ok": {"type": "boolean"}}},
        )
        result = self.route(catalog, tools=(tool,))
        tool.output_schema["properties"]["ok"]["type"] = "string"
        self.assertEqual(result.tools[0].output_schema["properties"]["ok"]["type"], "boolean")
        for invalid in (replace(tool, asynchronous="yes"), replace(tool, output_schema=[])):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.route(catalog, tools=(invalid,))

    def test_namespaced_mcp_ids_can_include_full_length_server_and_tool_names(self):
        tool_id = "mcp:" + "s" * 128 + "/" + "t" * 128
        self.skill(deps=(tool_id,))
        catalog = self.catalog(tools=(tool_id,))
        result = self.route(catalog, tools=(replace(self.tool(tool_id), origin="mcp"),))
        self.assertEqual(catalog.tool_ids(), frozenset({tool_id}))
        self.assertEqual([tool.id for tool in result.tools], [tool_id])
        self.assertEqual(result.skill_ids, ("mail",))

    def test_tool_inventory_order_does_not_change_trace(self):
        catalog = self.catalog(skills=(), tools=("lookup", "send"))
        tools = (self.tool("lookup", keywords=()), self.tool("send", keywords=()))
        plain = replace(self.large, supports_tools=False)
        forward = self.route(catalog, tools=tools, models=(plain,))
        backward = self.route(catalog, tools=tuple(reversed(tools)), models=(plain,))
        self.assertEqual(forward.reasons, backward.reasons)
        self.assertEqual((forward.context, forward.tools), (backward.context, backward.tools))

    def test_malformed_inventory_ids_raise_validation_errors_before_deduplication(self):
        catalog = Catalog(self.root)
        for tool in (replace(self.tool("lookup"), id=[]), object()):
            with self.subTest(tool=tool), self.assertRaises(ValueError):
                self.route(catalog, tools=(tool,))
        for model in (replace(self.large, id=[]), object()):
            with self.subTest(model=model), self.assertRaises(ValueError):
                self.route(catalog, models=(model,))

    def test_models_excluded_by_stage_grant_and_base_budget_leave_no_route(self):
        catalog = Catalog(self.root)
        wrong_stage = replace(self.large, id="answer-only", stages=frozenset({"answer"}))
        local = replace(self.large, id="local", local=True)
        too_small = replace(self.large, id="tiny", context_tokens=100)
        result = self.route(catalog, models=(wrong_stage, local, too_small), grants=frozenset())
        self.assertIsNone(result.model_id)
        self.assertEqual((result.context, result.tools, result.token_upper_bound), ("", (), 0))
        for expected in ("excluded stage", "excluded missing grant", "excluded base/output budget"):
            self.assertTrue(any(expected in reason for reason in result.reasons))


if __name__ == "__main__":
    unittest.main()

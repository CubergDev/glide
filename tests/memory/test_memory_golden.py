"""Whole-output characterisation of planning and the harness.

`planning.plan` and `Harness` carry rules whose exact output is public: the `reasons` of a plan are shown by
`glide memory plan` and `/context`, the stored events are what docs/memory.md promises, and the refusal messages
are what hosts and the CLI print. Point tests pin single rules; these pin the complete result of rich scenarios so a
rewrite cannot change one silently. The expected values live in tests/memory/golden/*.json and were produced by
the code as it stood before it was restructured. Do not regenerate them to make a failing test pass: a difference
is a behaviour change and needs a decision.
"""

import asyncio
import json
import tempfile
from pathlib import Path

import pytest

from glide.memory import Catalog, Harness, Model, Policy, Reply, Scope, Store, Tool, ToolCall
from glide.memory.planning import plan

GOLDEN = Path(__file__).parent / "golden"
SCOPE = Scope("u", "p", "s")


def expected(name):
    return json.loads((GOLDEN / f"{name}.json").read_text(encoding="utf-8"))


def jsonable(value):
    """What the golden file can hold: tuples become lists, exactly as in the stored JSON."""
    return json.loads(json.dumps(value))


# ---- planning -------------------------------------------------------------------------------------------------


def build_catalog(root: Path, *, plugins: tuple[str, ...] = ("bundle",)) -> Catalog:
    (root / "skills").mkdir(parents=True)
    (root / "plugins").mkdir()

    def manifest(kind, identifier, data, body="Instructions."):
        (root / kind / f"{identifier}.md").write_text("---\n" + json.dumps(data) + "\n---\n" + body, encoding="utf-8")

    def skill(identifier, keywords, stages, tools, body):
        data = {"id": identifier, "description": identifier, "keywords": keywords, "stages": stages, "tools": tools}
        manifest("skills", identifier, data, body)

    skill("mail", ["email contact"], ["write"], ["lookup", "send"], "Look up the contact, then send email.")
    skill("calendar", ["calendar"], ["write"], ["lookup", "cal"], "Check the calendar.")
    skill("night", ["email"], ["answer"], [], "Only at night.")
    skill("untouched", ["zebra"], ["write"], [], "Zebra.")
    skill("needs-ghost", ["email"], ["write"], ["ghost"], "Needs a tool nobody has.")
    every_skill = ["mail", "calendar", "night", "untouched", "needs-ghost"]
    manifest(
        "plugins",
        "bundle",
        {
            "id": "bundle",
            "description": "All",
            "skills": every_skill,
            "tools": ["lookup", "send", "cal", "ghost", "admin", "spare"],
        },
    )
    manifest(
        "plugins",
        "tools-only",
        {"id": "tools-only", "description": "Tools without skills", "skills": [], "tools": ["lookup", "send", "spare", "cal"]},
    )
    return Catalog(root, enabled_plugins=frozenset(plugins))


def inventory() -> tuple[Tool, ...]:
    def tool(identifier, keywords, permissions=frozenset({"mail"})):
        schema = {"type": "object", "properties": {"text": {"type": "string"}}}
        return Tool(identifier, "Fake " + identifier, keywords, permissions, schema, lambda arguments: None)

    return (
        tool("lookup", ("email", "contact")),
        tool("send", ("email",)),
        tool("cal", ("calendar",)),
        tool("admin", ("email",), frozenset({"root"})),
        tool("spare", ("email",)),
        tool("hidden", ("email",)),
        tool("noise", ("zebra",)),
    )


def memories() -> list[dict]:
    def memory(identifier, text, kind="preference", confidence=0.9, updated=100):
        return {"id": identifier, "text": text, "kind": kind, "source": "user", "confidence": confidence, "updated_at": updated}

    return [
        memory("p-old", "Use concise sentences", updated=100),
        memory("p-new", "Use concrete language", updated=200),
        memory("f-rel", "Email contact is Ada", kind="fact"),
        memory("f-off", "The sun is yellow", kind="fact"),
        {"id": "bad", "text": 5},
        memory("p-low", "Faint preference", confidence=0.3),
        memory("p-iso", "iso time", updated="2026-01-01T00:00:00"),
    ]


OVERLAYS = {
    "skill:mail": "Overlay mail body",
    "prompt:style": "email style overlay",
    "prompt:off": "zzz",
    "prompt:empty": "",
    "skill:ghost": "x",
}
BIG = Model("big", frozenset({"write"}), 100000, priority=4, supports_tools=True)
MODELS = (
    Model("tiny", frozenset({"write"}), 50, priority=0, supports_tools=True),
    Model("local", frozenset({"write"}), 100000, priority=1, supports_tools=True, local=True),
    Model("notools", frozenset({"write"}), 100000, priority=2),
    Model("other-stage", frozenset({"answer"}), 100000, priority=3, supports_tools=True),
    BIG,
)
BASE = {"context_tokens": 3000, "output_reserve": 100, "max_tools": 3, "max_skills": 2}
# name -> (plan keyword overrides, Policy overrides). Each case reaches reasons the others do not.
PLAN_CASES = {
    "rich": ({}, {}),
    "no_tool_calls_allowed": ({"models": (BIG,)}, {"max_tool_calls": 0}),
    "tight_context_budget": ({"models": (BIG,)}, {"context_tokens": 560, "max_tools": 6, "max_skills": 3}),
    "second_skill_does_not_fit": ({"models": (BIG,)}, {"context_tokens": 1100, "max_tools": 6, "max_skills": 3}),
    "tool_count_cannot_hold_required_bundle": ({"models": (BIG,)}, {"max_tools": 1}),
    "no_model": ({"models": ()}, {}),
    "no_model_and_nothing_to_anchor": ({"models": (), "goal": "zebra"}, {}),
    "classify_is_delegated": ({"stage": "classify"}, {}),
    "answer_stage": ({"stage": "answer", "models": (Model("a", frozenset({"answer"}), 9000),)}, {}),
    "nothing_enabled": ({"models": (BIG,), "plugins": (), "goal": "email"}, {}),
    "direct_tools_only": ({"models": (BIG,), "plugins": ("tools-only",), "goal": "email"}, {"max_tools": 2}),
    "second_direct_tool_does_not_fit": ({"models": (BIG,), "plugins": ("tools-only",), "goal": "email"}, {"context_tokens": 300}),
    "grants_withheld": ({"models": (BIG,), "grants": frozenset()}, {}),
    "calls_forbidden_and_nothing_to_anchor": ({"models": (BIG,), "goal": "zebra"}, {"max_tool_calls": 0}),
    "model_cannot_use_tools_and_nothing_to_anchor": (
        {"models": (Model("plain", frozenset({"write"}), 100000),), "goal": "zebra"},
        {},
    ),
}


def snapshot(result) -> dict:
    return {
        "model": result.model_id,
        "token_upper_bound": result.token_upper_bound,
        "catalog_revision": result.catalog_revision,
        "tools": [tool.id for tool in result.tools],
        "memory_ids": list(result.memory_ids),
        "skill_ids": list(result.skill_ids),
        "context": result.context,
        "reasons": list(result.reasons),
    }


def run_plan(name: str) -> dict:
    options, policy = PLAN_CASES[name]
    with tempfile.TemporaryDirectory() as directory:
        catalog = build_catalog(Path(directory) / "catalog", plugins=options.get("plugins", ("bundle",)))
        result = plan(
            SCOPE,
            options.get("goal", "email contact calendar"),
            stage=options.get("stage", "write"),
            memories=memories(),
            overlays=dict(OVERLAYS),
            catalog=catalog,
            tools=inventory(),
            models=options.get("models", MODELS),
            grants=options.get("grants", frozenset({"mail"})),
            policy=Policy(**{**BASE, **policy}),
            base_tokens=10,
        )
    return snapshot(result)


@pytest.mark.parametrize("name", sorted(PLAN_CASES))
def test_plan_output_is_pinned(name):
    assert jsonable(run_plan(name)) == expected("plan")[name]


def test_the_golden_plan_cases_cover_every_exclusion_reason_family():
    """A case that stops reaching its branch would make its golden value meaningless: check the families directly."""
    reasons = " | ".join(reason for case in expected("plan").values() for reason in case["reasons"])
    for family in (
        "excluded missing grants",
        "excluded catalog enablement",
        "excluded absent host inventory",
        "excluded stage",
        "excluded lexical relevance",
        "excluded base/output budget",
        "excluded missing grant model:local",
        "excluded required tool support",
        "excluded tool invocation count budget",
        "excluded skill count budget",
        "excluded tool count budget",
        "excluded context budget (whole item)",
        "excluded definition budget (whole schema)",
        "exceeds policy count budget",
        "no viable complete tool/skill route",
        "no viable model route",
        "classify: delegate unchanged",
        "required anchor: included complete",
        "applied to included skill",
        "excluded invalid data",
    ):
        assert family in reasons, family


def test_a_chunk_that_exactly_fills_the_budget_fits_and_one_more_byte_does_not():
    preference = {
        "id": "p",
        "text": "Use concise sentences",
        "kind": "preference",
        "source": "user",
        "confidence": 1,
        "updated_at": 1,
    }
    with tempfile.TemporaryDirectory() as directory:
        catalog = build_catalog(Path(directory) / "catalog", plugins=())

        def run(budget):
            return plan(
                SCOPE,
                "zebra",
                stage="write",
                memories=[preference],
                overlays={},
                catalog=catalog,
                tools=(),
                grants=frozenset(),
                models=(Model("m", frozenset({"write"}), 100000),),
                policy=Policy(context_tokens=budget, output_reserve=0),
            )

        exact = run(100000).token_upper_bound
        assert run(exact).memory_ids == ("p",)
        assert run(exact - 1).memory_ids == ()
        assert "memory p: excluded context budget (whole item)" in run(exact - 1).reasons


def test_a_skill_whose_tools_would_pass_the_tool_limit_by_one_is_left_out():
    with tempfile.TemporaryDirectory() as directory:
        catalog = build_catalog(Path(directory) / "catalog")

        def run(max_tools):
            return plan(
                SCOPE,
                "email contact calendar",
                stage="write",
                memories=[],
                overlays={},
                catalog=catalog,
                tools=inventory(),
                grants=frozenset({"mail"}),
                models=(BIG,),
                policy=Policy(context_tokens=100000, output_reserve=0, max_tools=max_tools),
            )

        assert [tool.id for tool in run(2).tools] == ["cal", "lookup"]
        assert "skill mail: excluded required tool count budget" in run(2).reasons
        assert [tool.id for tool in run(3).tools] == ["cal", "lookup", "send"]


# ---- harness --------------------------------------------------------------------------------------------------


class World:
    """A harness over fakes: two tools, a catalog that enables both, a model that supports tools."""

    def __init__(self, directory: Path, **policy):
        (directory / "catalog/plugins").mkdir(parents=True)
        manifest = {"id": "toolset", "description": "Host tools", "skills": [], "tools": ["local:echo", "local:fail"]}
        (directory / "catalog/plugins/tools.md").write_text("---\n" + json.dumps(manifest) + "\n---\n")
        self.store = Store(directory / "state.sqlite")
        self.effects: list = []
        schema = {"type": "object", "properties": {"value": {"type": "integer"}}}

        def fail(arguments):
            self.effects.append(("fail", arguments))
            raise RuntimeError("boom")

        self.echo = Tool(
            "local:echo",
            "Echo the input",
            ("echo",),
            frozenset({"read"}),
            schema,
            lambda args: self.effects.append(("echo", args["value"])) or {"value": args["value"]},
        )
        self.fail = Tool("local:fail", "Always fails", ("echo",), frozenset({"read"}), schema, fail)
        self.model = Model("fake", frozenset({"handoff"}), 32768, supports_tools=True)
        catalog = Catalog(directory / "catalog", enabled_plugins=frozenset({"toolset"}))
        self.harness = Harness(
            self.store, catalog, tools=(self.echo, self.fail), models=(self.model,), policy=Policy(**policy) if policy else None
        )
        self.grants = frozenset({"read"})

    def events(self) -> list[dict]:
        """Stored events oldest first, with the random ids and the clock removed."""
        rows = []
        for row in reversed(self.store.events(SCOPE)):
            payload = {key: value for key, value in row["payload"].items() if key != "timestamp"}
            if "plan_id" in payload:
                payload["plan_id"] = "<plan>"
            rows.append({"kind": row["kind"], "payload": payload})
        return rows


def refusal(call) -> str:
    try:
        call()
    except BaseException as error:
        return f"{type(error).__name__}: {error}"
    return "no error"


def test_a_two_call_dispatch_leaves_exactly_this_trace():
    with tempfile.TemporaryDirectory() as directory:
        world = World(Path(directory))
        seen, asked = [], []

        def model(request):
            seen.append(
                {
                    "goal": request.goal,
                    "max_output_tokens": request.max_output_tokens,
                    "tools": [tool.id for tool in request.plan.tools],
                    "trajectory": request.trajectory,
                }
            )
            if len(seen) == 1:
                return Reply("thinking", (ToolCall("c1", "local:echo", {"value": 7}), ToolCall("c2", "local:echo", {"value": 8})))
            return Reply("finished")

        def authorize(tool, arguments):
            asked.append((tool.id, arguments))
            return True

        reply = world.harness.dispatch(SCOPE, "echo", model, grants=world.grants, authorize=authorize)
        trace = {
            "reply": reply.text,
            "model_requests": seen,
            "authorizer_saw": asked,
            "effects": world.effects,
            "events": world.events(),
            "audit_errors": list(world.harness.audit_errors),
        }
    assert jsonable(trace) == expected("harness")["two_call_dispatch"]


def test_a_failing_tool_is_audited_and_the_error_propagates():
    with tempfile.TemporaryDirectory() as directory:
        world = World(Path(directory))

        def model(request):
            return Reply(calls=(ToolCall("c1", "local:fail", {"value": 1}),))

        outcome = refusal(lambda: world.harness.dispatch(SCOPE, "echo", model, grants=world.grants, authorize=lambda *_: True))
        trace = {"outcome": outcome, "events": world.events(), "effects": world.effects}
    assert jsonable(trace) == expected("harness")["failing_tool"]


def test_the_async_dispatch_leaves_the_same_trace_as_the_sync_one():
    with tempfile.TemporaryDirectory() as directory:
        world = World(Path(directory))
        turns = []

        async def model(request):
            turns.append(len(request.trajectory))
            if len(turns) == 1:
                return Reply(calls=(ToolCall("c1", "local:echo", {"value": 3}),))
            return Reply("done")

        async def authorize(tool, arguments):
            return True

        reply = asyncio.run(world.harness.adispatch(SCOPE, "echo", model, grants=world.grants, authorize=authorize))
        trace = {"reply": reply.text, "turns": turns, "effects": world.effects, "events": world.events()}
    assert jsonable(trace) == expected("harness")["async_dispatch"]


def reply_of(*calls, text=""):
    return lambda request: Reply(text, tuple(calls))


def many(count):
    return [ToolCall(f"k{index}", "local:echo", {"value": index}) for index in range(count)]


# name -> a callable taking the World and returning what to run; the golden value is the exception it raises
DISPATCH_REFUSALS = {
    "reply_is_not_a_reply": lambda w: lambda r: "text",
    "reply_text_too_large": lambda w: reply_of(text="x" * 70000),
    "calls_without_an_authorizer": lambda w: reply_of(ToolCall("a", "local:echo", {"value": 1})),
    "empty_call_id": lambda w: reply_of(ToolCall("", "local:echo", {"value": 1})),
    "oversized_call_id": lambda w: reply_of(ToolCall("x" * 129, "local:echo", {"value": 1})),
    "arguments_not_an_object": lambda w: reply_of(ToolCall("a", "local:echo", [1])),
    "duplicate_call_ids_in_a_batch": lambda w: reply_of(
        ToolCall("a", "local:echo", {"value": 1}), ToolCall("a", "local:echo", {"value": 2})
    ),
    "batch_over_the_call_budget": lambda w: reply_of(*many(13)),
    "tool_not_in_the_plan": lambda w: reply_of(ToolCall("a", "local:ghost", {"value": 1})),
    "arguments_not_json": lambda w: reply_of(ToolCall("a", "local:echo", {"value": float("nan")})),
}


@pytest.mark.parametrize("name", sorted(DISPATCH_REFUSALS))
def test_dispatch_refusal_is_pinned(name):
    with tempfile.TemporaryDirectory() as directory:
        world = World(Path(directory))
        authorize = None if name == "calls_without_an_authorizer" else (lambda *_: True)
        message = refusal(
            lambda: world.harness.dispatch(
                SCOPE, "echo", DISPATCH_REFUSALS[name](world), grants=world.grants, authorize=authorize
            )
        )
        assert message == expected("harness")["dispatch_refusals"][name]
        assert world.effects == []


def test_dispatch_budget_and_eligibility_refusals_are_pinned():
    def run(**policy):
        directory = tempfile.TemporaryDirectory()
        world = World(Path(directory.name), **policy)
        world.keep = directory  # the store lives in it until the world is dropped
        return world

    world = run(max_tool_calls=1)
    batch = refusal(
        lambda: world.harness.dispatch(SCOPE, "echo", reply_of(*many(2)), grants=world.grants, authorize=lambda *_: True)
    )
    world = run(max_tool_calls=1)
    endless = refusal(
        lambda: world.harness.dispatch(
            SCOPE,
            "echo",
            lambda request: Reply(calls=(ToolCall(f"n{len(request.trajectory)}", "local:echo", {"value": 1}),)),
            grants=world.grants,
            authorize=lambda *_: True,
        )
    )
    effects_before_refusal = list(world.effects)
    world = run()
    no_model = refusal(
        lambda: world.harness.dispatch(SCOPE, "echo", lambda request: Reply("x"), stage="classify", grants=world.grants)
    )
    results = {
        "batch_larger_than_the_remaining_budget": batch,
        "model_that_never_stops_calling": endless,
        "effects_before_that_refusal": effects_before_refusal,
        "no_eligible_model": no_model,
    }
    assert jsonable(results) == expected("harness")["dispatch_budgets"]


def test_a_local_model_grant_revoked_between_turns_is_refused_before_the_next_request():
    with tempfile.TemporaryDirectory() as directory:
        world = World(Path(directory))
        world.harness.models = (Model("local", frozenset({"handoff"}), 32768, supports_tools=True, local=True),)
        live = {"grants": frozenset({"read", "model:local"})}
        requests = []

        def model(request):
            requests.append(1)
            live["grants"] = frozenset({"read"})
            return Reply(calls=(ToolCall("a", "local:echo", {"value": 1}),))

        message = refusal(
            lambda: world.harness.dispatch(
                SCOPE,
                "echo",
                model,
                grants=frozenset({"read", "model:local"}),
                authorize=lambda *_: True,
                current_grants=lambda: live["grants"],
            )
        )
        trace = {"refusal": message, "requests": len(requests), "effects": world.effects}
    assert jsonable(trace) == expected("harness")["revoked_local_grant"]


def test_a_second_dispatch_for_the_same_scope_while_one_runs_is_refused():
    with tempfile.TemporaryDirectory() as directory:
        world = World(Path(directory))
        inner = []

        def model(request):
            inner.append(refusal(lambda: world.harness.dispatch(SCOPE, "echo", lambda r: Reply("x"), grants=world.grants)))
            return Reply("outer")

        assert world.harness.dispatch(SCOPE, "echo", model, grants=world.grants).text == "outer"
        assert inner == expected("harness")["reentrant"]


def test_direct_invocation_refusals_are_pinned():
    with tempfile.TemporaryDirectory() as directory:
        world = World(Path(directory))
        plan_id = world.harness.prepare(SCOPE, "echo", grants=world.grants).id

        def invoke(tool="local:echo", arguments=None, *, grants=None, authorize=lambda *_: True, plan=None):
            return world.harness.invoke(
                SCOPE,
                plan or plan_id,
                tool,
                {"value": 1} if arguments is None else arguments,
                grants=world.grants if grants is None else grants,
                authorize=authorize,
            )

        results = {
            "unknown_plan": refusal(lambda: invoke(plan="nope")),
            "tool_not_selected": refusal(lambda: invoke("local:ghost")),
            "grants_missing": refusal(lambda: invoke(grants=frozenset())),
            "grants_not_a_frozenset": refusal(lambda: invoke(grants={"read"})),
            "arguments_not_an_object": refusal(lambda: invoke(arguments=[1])),
            "authorizer_says_truthy_not_true": refusal(lambda: invoke(authorize=lambda *_: 1)),
            "authorizer_says_no": refusal(lambda: invoke(authorize=lambda *_: False)),
            "memory_change_expires_the_plan": None,
        }
        world.store.remember(SCOPE, "k", "a new memory")
        results["memory_change_expires_the_plan"] = refusal(lambda: invoke())
        assert world.effects == []
    assert jsonable(results) == expected("harness")["invoke_refusals"]


def test_async_tool_needs_the_async_entry_point_and_sync_callbacks_need_the_sync_one():
    with tempfile.TemporaryDirectory() as directory:
        world = World(Path(directory))

        async def slow(arguments):
            return {"ok": True}

        world.harness.replace_tools(
            (Tool("local:echo", "x", ("echo",), frozenset({"read"}), {"type": "object"}, slow, asynchronous=True),)
        )
        plan_id = world.harness.prepare(SCOPE, "echo", grants=world.grants).id
        sync_refusal = refusal(
            lambda: world.harness.invoke(SCOPE, plan_id, "local:echo", {}, grants=world.grants, authorize=lambda *_: True)
        )
        result = asyncio.run(
            world.harness.ainvoke(SCOPE, plan_id, "local:echo", {}, grants=world.grants, authorize=lambda *_: True)
        )

        async def async_authorizer(tool, arguments):
            return True

        wrong_way = refusal(
            lambda: world.harness.invoke(SCOPE, plan_id, "local:echo", {}, grants=world.grants, authorize=async_authorizer)
        )
    assert jsonable([sync_refusal, result, wrong_way]) == expected("harness")["async_entry"]


def test_observe_user_saves_only_direct_preference_sentences():
    text = (
        "I prefer short answers. Remember that my name is Ada.\n"
        "> I prefer quoted text\n"
        "```\nremember that this is code\n```\n"
        "~~~\ni prefer a tilde fence\n~~~\n"
        "Please remember the milk\n"
        "My preference is dark mode!\n"
        "I prefer x\n"
        "Nothing to see here.\n"
        "I prefer short answers."
    )
    with tempfile.TemporaryDirectory() as directory:
        world = World(Path(directory), auto_memory=True)
        saved = world.harness.observe_user(SCOPE, text)
        again = world.harness.observe_user(SCOPE, text)
        stored = sorted(memory["text"] for memory in world.store.memories(SCOPE))
        keys = [memory["key"] for memory in world.store.memories(SCOPE)]
        off = World(Path(directory) / "off")
        trace = {
            "saved": len(saved),
            "ids_are_stable": saved == again,
            "stored": stored,
            "keys_are_hashes": all(key.startswith("auto:") and len(key) == 29 for key in keys),
            "off_by_default": off.harness.observe_user(SCOPE, text),
            "credentials_refused": refusal(lambda: world.harness.observe_user(SCOPE, "remember that sk-" + "a" * 30)),
        }
    assert jsonable(trace) == expected("harness")["observe_user"]


def test_slash_commands_are_pinned():
    with tempfile.TemporaryDirectory() as directory:
        world = World(Path(directory))
        harness = world.harness
        results = {}
        results["remember_default_level"] = bool(harness.command(SCOPE, "/remember colour = blue"))
        results["remember_session"] = bool(harness.command(SCOPE, "/remember session  note = temp "))
        results["remember_user_level"] = bool(harness.command(SCOPE, "/remember user lang = en"))
        results["memory_texts"] = sorted((m["key"], m["text"]) for m in harness.command(SCOPE, "/memory"))
        results["remember_without_equals"] = refusal(lambda: harness.command(SCOPE, "/remember nothing"))
        results["context_before_prepare"] = harness.command(SCOPE, "/context")
        bundle = harness.prepare(SCOPE, "echo", grants=world.grants)
        context = harness.command(SCOPE, "/context")
        results["context_keys"] = sorted(context)
        results["context_matches_plan"] = context["id"] == bundle.id and context["tools"] == [t.id for t in bundle.tools]
        results["events_kinds"] = sorted({event["kind"] for event in harness.command(SCOPE, "/events")})
        results["forget_unknown"] = harness.command(SCOPE, "/forget nope")
        results["refine_empty"] = harness.command(SCOPE, "/refine")
        results["refine_bad_action"] = refusal(lambda: harness.command(SCOPE, "/refine apply"))
        results["refine_other_verb"] = refusal(lambda: harness.command(SCOPE, "/refine undo x"))
        results["rollback_unknown"] = refusal(lambda: harness.command(SCOPE, "/rollback missing"))
        results["unknown"] = refusal(lambda: harness.command(SCOPE, "/nope"))
        results["not_a_command"] = refusal(lambda: harness.command(SCOPE, "remember x = y"))
    assert jsonable(results) == expected("harness")["commands"]


def test_three_verified_successes_draft_one_lesson_and_a_rollback_stops_the_next():
    with tempfile.TemporaryDirectory() as directory:
        world = World(Path(directory), auto_refine=True)
        harness, store = world.harness, world.store
        summary = "Open the menu first"
        trace = {}
        for number in range(1, 5):
            harness.record_outcome(SCOPE, f"run-{number}", True, summary)
            trace[f"proposals_after_{number}"] = [
                (row["status"], row["target"].startswith("prompt:lesson-")) for row in store.proposals(SCOPE)
            ]
        before = len(store.proposals(SCOPE))
        for number in range(3):
            harness.record_outcome(SCOPE, f"failed-{number}", False, "A different tactic")
        trace["failures_add_no_proposal"] = len(store.proposals(SCOPE)) - before
        for number in range(3):
            harness.record_outcome(SCOPE, f"empty-{number}", True, "")
        trace["empty_summaries_add_no_proposal"] = len(store.proposals(SCOPE)) - before
        trace["overlay_before_prepare"] = store.overlays(SCOPE)
        harness.prepare(SCOPE, "echo", grants=world.grants)
        applied = store.overlays(SCOPE)
        trace["overlay_after_prepare"] = list(applied.values())
        trace["status_after_prepare"] = [row["status"] for row in store.proposals(SCOPE)]
        proposal = store.proposals(SCOPE)[0]
        store.rollback(SCOPE, proposal["id"], expected_revision=store.revision(SCOPE))
        for number in range(5, 9):
            harness.record_outcome(SCOPE, f"run-{number}", True, summary)
        trace["after_rollback"] = sorted(row["status"] for row in store.proposals(SCOPE))
        harness.prepare(SCOPE, "echo", grants=world.grants)
        trace["overlay_after_rollback"] = store.overlays(SCOPE)
    assert jsonable(trace) == expected("harness")["lessons"]


def test_a_concurrent_edit_defers_the_pending_refinement_and_audits_it():
    with tempfile.TemporaryDirectory() as directory:
        world = World(Path(directory), auto_refine=True)
        harness, store = world.harness, world.store
        for number in range(1, 4):
            harness.record_outcome(SCOPE, f"run-{number}", True, "Close the dialog")
        other = store.propose(SCOPE, "prompt:other", "other text", [store.outcomes(SCOPE)[0]["id"]])
        store.apply(SCOPE, other, expected_revision=store.revision(SCOPE))
        harness.prepare(SCOPE, "echo", grants=world.grants)
        trace = {
            "statuses": sorted(row["status"] for row in store.proposals(SCOPE)),
            "events": [event["kind"] for event in world.events() if event["kind"] != "context_plan"],
        }
    assert jsonable(trace) == expected("harness")["deferred_refinement"]


# ---- store ----------------------------------------------------------------------------------------------------


def test_store_refusals_are_pinned():
    with tempfile.TemporaryDirectory() as directory:
        store = Store(Path(directory) / "state.sqlite")
        evidence = None
        outcome = store.record_outcome(SCOPE, "run-1", True, "Opened the menu")
        evidence = [outcome]

        def remember(**changes):
            arguments = {"key": "k", "text": "some text", **changes}
            return refusal(lambda: store.remember(SCOPE, arguments.pop("key"), arguments.pop("text"), **arguments))

        results = {
            "remember_ok": remember(),
            "remember_blank_text": remember(text="   "),
            "remember_text_not_a_string": remember(text=5),
            "remember_text_too_long": remember(text="x" * 8193),
            "remember_nul": remember(text="a\x00b"),
            "remember_credential": remember(text="key sk-" + "a" * 30),
            "remember_assigned_secret": remember(text="password: hunter2"),
            "remember_placeholder_secret_is_fine": remember(text="password: redacted"),
            "remember_bad_level": remember(level="team"),
            "remember_control_character_in_key": remember(key="a\nb"),
            "remember_blank_key": remember(key=" "),
            "remember_key_too_long": remember(key="k" * 129),
            "remember_confidence_bool": remember(confidence=True),
            "remember_confidence_text": remember(confidence="1"),
            "remember_confidence_high": remember(confidence=1.5),
            "remember_confidence_negative": remember(confidence=-0.1),
            "remember_confidence_nan": remember(confidence=float("nan")),
            "remember_confidence_inf": remember(confidence=float("inf")),
            "remember_ttl_bool": remember(ttl_seconds=True),
            "remember_ttl_zero": remember(ttl_seconds=0),
            "remember_ttl_negative": remember(ttl_seconds=-1),
            "remember_ttl_nan": remember(ttl_seconds=float("nan")),
            "remember_ttl_inf": remember(ttl_seconds=float("inf")),
            "remember_ttl_beyond_ten_years": remember(ttl_seconds=86400 * 3650 + 1),
            "remember_ttl_text": remember(ttl_seconds="5"),
            "remember_ttl_ten_years_is_fine": remember(ttl_seconds=86400 * 3650),
            "scope_not_a_scope": refusal(lambda: store.memories(("u", "p", "s"))),
            "forget_blank_id": refusal(lambda: store.forget(SCOPE, " ")),
            "event_not_a_dict": refusal(lambda: store.event(SCOPE, "kind", [1])),
            "event_not_json": refusal(lambda: store.event(SCOPE, "kind", {"a": float("nan")})),
            "event_with_credential": refusal(lambda: store.event(SCOPE, "kind", {"a": "sk-" + "b" * 30})),
            "event_kind_blank": refusal(lambda: store.event(SCOPE, "", {})),
            "events_limit_zero": refusal(lambda: store.events(SCOPE, 0)),
            "events_limit_bool": refusal(lambda: store.events(SCOPE, True)),
            "events_limit_too_big": refusal(lambda: store.events(SCOPE, 1001)),
            "outcome_success_not_bool": refusal(lambda: store.record_outcome(SCOPE, "r", 1, "s")),
            "outcome_summary_with_url": refusal(lambda: store.record_outcome(SCOPE, "r", True, "went to https://example.com")),
            "outcome_summary_with_www": refusal(lambda: store.record_outcome(SCOPE, "r", True, "went to www.example.com")),
            "outcome_summary_with_credential": refusal(lambda: store.record_outcome(SCOPE, "r", True, "token sk-" + "c" * 30)),
            "outcome_summary_too_long": refusal(lambda: store.record_outcome(SCOPE, "r", True, "x" * 8193)),
            "propose_bad_target": refusal(lambda: store.propose(SCOPE, "other:thing", "t", evidence)),
            "propose_blank_text": refusal(lambda: store.propose(SCOPE, "prompt:a", " ", evidence)),
            "propose_evidence_not_a_list": refusal(lambda: store.propose(SCOPE, "prompt:a", "t", "x")),
            "propose_no_evidence": refusal(lambda: store.propose(SCOPE, "prompt:a", "t", [])),
            "propose_too_much_evidence": refusal(lambda: store.propose(SCOPE, "prompt:a", "t", [str(n) for n in range(65)])),
            "propose_duplicate_evidence": refusal(lambda: store.propose(SCOPE, "prompt:a", "t", evidence * 2)),
            "propose_unknown_evidence": refusal(lambda: store.propose(SCOPE, "prompt:a", "t", ["nope"])),
            "apply_revision_bool": refusal(lambda: store.apply(SCOPE, "x", expected_revision=True)),
            "apply_revision_negative": refusal(lambda: store.apply(SCOPE, "x", expected_revision=-1)),
            "apply_unknown_proposal": refusal(lambda: store.apply(SCOPE, "x", expected_revision=0)),
            "rollback_draft_proposal": None,
            "apply_with_a_stale_revision": None,
            "closed_store": None,
        }
        proposal = store.propose(SCOPE, "prompt:a", "first", evidence)
        results["rollback_draft_proposal"] = refusal(lambda: store.rollback(SCOPE, proposal, expected_revision=0))
        results["apply_with_a_stale_revision"] = refusal(lambda: store.apply(SCOPE, proposal, expected_revision=5))
        results["apply_ok"] = store.apply(SCOPE, proposal, expected_revision=0)
        results["apply_twice"] = refusal(lambda: store.apply(SCOPE, proposal, expected_revision=1))
        newer = store.propose(SCOPE, "prompt:a", "second", evidence)
        store.apply(SCOPE, newer, expected_revision=1)
        results["rollback_after_a_newer_change"] = refusal(lambda: store.rollback(SCOPE, proposal, expected_revision=2))
        results["same_proposal_is_reused"] = store.propose(SCOPE, "prompt:a", "second", evidence) == newer
        store.close()
        store.close()
        results["closed_store"] = refusal(lambda: store.memories(SCOPE))
        results["closed_store_enter"] = refusal(store.__enter__)
        results["database_path_is_a_uri"] = refusal(lambda: Store("file:x.db"))
        results["database_path_is_a_directory"] = refusal(lambda: Store(directory))
    assert jsonable(results) == expected("store")["refusals"]


class Clock:
    """A clock the test moves by hand, so expiry and ordering never depend on how fast the machine is."""

    def __init__(self, monkeypatch):
        self.now = 1_000_000.0
        monkeypatch.setattr("glide.memory.store.time.time", lambda: self.now)


def test_memories_are_visible_by_level_and_expire(monkeypatch):
    clock = Clock(monkeypatch)
    other = Scope("u", "p", "other-session")
    other_project = Scope("u", "q", "s")
    other_user = Scope("v", "p", "s")
    with tempfile.TemporaryDirectory() as directory:
        store = Store(Path(directory) / "state.sqlite")
        store.remember(SCOPE, "same", "session text", level="session")
        store.remember(SCOPE, "same", "project text", level="project")
        store.remember(SCOPE, "same", "user text", level="user")
        store.remember(SCOPE, "only-user", "user only", level="user")
        store.remember(SCOPE, "only-project", "project only", level="project")
        store.remember(SCOPE, "only-session", "session only", level="session")
        store.remember(SCOPE, "short", "short lived", ttl_seconds=60)
        first = store.remember(SCOPE, "again", "v1")
        second = store.remember(SCOPE, "again", "v2")
        view = {}
        for name, scope in (
            ("same", SCOPE),
            ("other_session", other),
            ("other_project", other_project),
            ("other_user", other_user),
        ):
            view[name] = sorted((row["key"], row["text"]) for row in store.memories(scope))
        clock.now += 61
        view["after_expiry"] = sorted(row["key"] for row in store.memories(SCOPE))
        store.remember(SCOPE, "pruned-on-write", "x")
        view["expired_row_is_deleted_by_the_next_write"] = store._db.execute(
            "SELECT COUNT(*) FROM memories WHERE key='short'"
        ).fetchone()[0]
        view["upsert_keeps_the_id"] = first == second
        view["forget_from_another_session"] = store.forget(other, first)
        view["forget_from_another_user"] = store.forget(other_user, first)
        view["forget_own"] = store.forget(SCOPE, first)
        view["forgotten_is_gone"] = [row["key"] for row in store.memories(SCOPE) if row["key"] == "again"]
        view["row_keys"] = sorted(store.memories(SCOPE)[0])
    assert jsonable(view) == expected("store")["visibility"]


def test_events_are_scoped_trimmed_and_newest_first(monkeypatch):
    clock = Clock(monkeypatch)
    other = Scope("u", "p", "other-session")
    with tempfile.TemporaryDirectory() as directory:
        store = Store(Path(directory) / "state.sqlite")
        for number in range(1005):
            clock.now += 1
            store.event(SCOPE, "k", {"n": number})
        clock.now += 1
        store.event(other, "other", {"n": -1})
        mine = store.events(SCOPE, 1000)
        view = {
            "kept_for_the_project": len(mine) + len(store.events(other)),
            "newest_first": [row["payload"]["n"] for row in mine[:3]],
            "oldest_kept": mine[-1]["payload"]["n"] if len(mine) < 1000 else None,
            "default_limit": len(store.events(SCOPE)),
            "other_session_sees_only_its_own": [row["kind"] for row in store.events(other)],
            "row_keys": sorted(mine[0]),
        }
        clock.now += 31 * 86400
        view["thirty_days_later_nothing_is_readable"] = store.events(SCOPE)
        store.event(SCOPE, "fresh", {})
        view["old_rows_are_deleted_by_the_next_write"] = store._db.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    assert jsonable(view) == expected("store")["events"]

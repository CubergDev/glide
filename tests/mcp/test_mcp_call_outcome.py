"""A tool call whose result never arrived may have run. The bridge says so, the harness records it, and nothing repeats it."""

import json

import pytest
from mcp_guard_loader import no_real_memory_or_mcp, sync  # noqa: F401  (autouse guard, async runner)

from glide.mcp import MCPBridge
from glide.memory import Catalog, Harness, Model, Reply, Scope, Store, ToolCall, ToolOutcomeUnknown

SCOPE = Scope("alice", "project", "session")


class Rpc(Exception):
    def __init__(self, code):
        self.error = {"code": code, "message": "SECRET-SERVER-TEXT"}
        super().__init__("SECRET-SERVER-TEXT")


def bridge_failing_with(error, calls):
    def request(method, params):
        if method == "tools/list":
            return {"tools": [{"name": "send", "inputSchema": {"type": "object"}}]}
        calls.append(params)
        raise error

    bridge = MCPBridge(
        "srv",
        request,
        permissions={"send": frozenset()},
        keywords={"send": ("send",)},
        capabilities={"tools": {"listChanged": True}},
    )
    return bridge, bridge.refresh()[0]


@pytest.mark.parametrize(
    ("error", "unknown"),
    [
        (Rpc(-32602), False),
        (Rpc(-32601), False),
        (Rpc(-32600), False),
        (Rpc(-32000), True),
        (Rpc(-32603), True),
        (TimeoutError("SECRET"), True),
        (ConnectionResetError("SECRET"), True),
        (RuntimeError("SECRET"), True),
    ],
)
def test_a_failed_call_is_marked_unknown_unless_the_server_refused_it_before_running_anything(error, unknown):
    calls = []
    _, tool = bridge_failing_with(error, calls)
    with pytest.raises(type(error)) as caught:
        tool.invoke({"to": "x"})
    assert caught.value is error  # the host's own error, as documented, not a new exception type
    assert getattr(caught.value, "outcome_unknown", None) is unknown
    assert any("may or may not have happened" in note for note in getattr(caught.value, "__notes__", [])) is unknown
    assert len(calls) == 1


def test_an_unusable_reply_to_a_call_is_unknown_too():
    def request(method, params):
        if method == "tools/list":
            return {"tools": [{"name": "send", "inputSchema": {"type": "object"}}]}
        return {"content": "not a list"}

    bridge = MCPBridge(
        "srv",
        request,
        permissions={"send": frozenset()},
        keywords={"send": ("send",)},
        capabilities={"tools": {"listChanged": True}},
    )
    with pytest.raises(ValueError) as caught:
        bridge.refresh()[0].invoke({})
    assert caught.value.outcome_unknown is True


def test_a_stale_catalog_refusal_happens_before_any_request_so_it_is_not_unknown():
    calls = []
    bridge, tool = bridge_failing_with(RuntimeError("x"), calls)
    bridge.on_notification("notifications/tools/list_changed")
    with pytest.raises(RuntimeError) as caught:
        tool.invoke({})
    assert calls == [] and getattr(caught.value, "outcome_unknown", False) is False


def test_the_harness_records_an_unknown_outcome_and_the_turn_stops_without_replay(tmp_path):
    (tmp_path / "catalog/plugins").mkdir(parents=True)
    manifest = {"id": "p", "description": "d", "skills": [], "tools": ["mcp:srv/send"]}
    (tmp_path / "catalog/plugins/p.md").write_text("---\n" + json.dumps(manifest) + "\n---\n")
    calls = []
    bridge, _ = bridge_failing_with(TimeoutError("SECRET"), calls)
    with Store(tmp_path / "s.sqlite") as store:
        harness = Harness(
            store,
            Catalog(tmp_path / "catalog", enabled_plugins=frozenset({"p"})),
            tools=bridge.tools(),
            models=(Model("m", frozenset({"handoff"}), 100000, supports_tools=True),),
        )
        requests = []

        def model(request):
            requests.append(request)
            return Reply(calls=(ToolCall("c1", "mcp:srv/send", {"to": "x"}),))

        with pytest.raises(ToolOutcomeUnknown, match="may or may not have happened") as caught:
            harness.dispatch(SCOPE, "send it", model, authorize=lambda *_: True)
        assert isinstance(caught.value.__cause__, TimeoutError) and "SECRET" not in str(caught.value)
        finished = [event["payload"] for event in store.events(SCOPE) if event["kind"] == "tool_finished"]
    assert len(calls) == 1 and len(requests) == 1  # no second attempt, no second model turn
    assert finished[0]["ok"] is False and finished[0]["outcome_unknown"] is True
    assert "SECRET" not in json.dumps(finished)

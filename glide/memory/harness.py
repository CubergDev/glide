"""Compose persistence, routing and host callbacks; no model SDK or device imports."""

import asyncio
import hashlib
import json
import re
import sqlite3
import threading
from collections import deque
from collections.abc import Callable
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, replace
from functools import partial
from typing import Any
from uuid import uuid4

from ._callbacks import adrive, drive
from ._json import bounded_json
from .catalog import Catalog
from .contracts import Model, Plan, Policy, Scope, Tool
from .events import invocation
from .planning import plan
from .store import Store, validate_text


@dataclass(frozen=True)
class ToolCall:
    id: str
    tool_id: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class Reply:
    text: str = ""
    calls: tuple[ToolCall, ...] = ()


@dataclass(frozen=True)
class Request:
    plan: Plan
    goal: str
    # Ephemeral assistant/tool turns; the host translates these to its provider's protocol.
    trajectory: tuple[dict[str, Any], ...]
    max_output_tokens: int


@dataclass
class _Issued:
    plan: Plan
    revision: int
    tool_revision: int
    memory_signature: str
    calls: int = 0


def _json_copy(value: Any, limit: int = 65536) -> Any:
    return bounded_json(value, limit, ValueError, "host callback payload")


def _goal_bound(goal: str) -> int:
    # Default text/framing estimate; hosts with other system/history/image content supply base_tokens.
    return len(json.dumps({"goal": goal}, ensure_ascii=True).encode()) + 256


def _memory_signature(memories: list[dict]) -> str:
    return hashlib.sha256(json.dumps(memories, sort_keys=True, allow_nan=False).encode()).hexdigest()


class Harness:
    def __init__(
        self,
        store: Store,
        catalog: Catalog,
        *,
        tools: tuple[Tool, ...] = (),
        models: tuple[Model, ...] = (),
        policy: Policy | None = None,
    ):
        if len({tool.id for tool in tools}) != len(tools) or len({model.id for model in models}) != len(models):
            raise ValueError("duplicate host tool or model id")
        self.store, self.catalog = store, catalog
        self.tools, self.models, self.policy = tools, models, Policy() if policy is None else policy
        self._issued: dict[Scope, _Issued] = {}
        self._tool_revision = 0
        self._running: set[Scope] = set()
        self.audit_errors: deque[dict[str, str]] = deque(maxlen=128)
        self._pending: dict[tuple[str, str], tuple[str, int]] = {}
        self._lock = threading.RLock()

    def replace_tools(self, tools: tuple[Tool, ...]) -> None:
        """Publish a refreshed host/MCP inventory and expire previously issued plans."""
        if not isinstance(tools, tuple) or any(not isinstance(tool, Tool) for tool in tools):
            raise TypeError("tools must be a tuple of Tool descriptors")
        if len({tool.id for tool in tools}) != len(tools):
            raise ValueError("duplicate host tool id")
        with self._lock:
            self.tools = tools
            self._tool_revision += 1
            self._issued.clear()

    def replace_tool_namespace(self, namespace: str, tools: tuple[Tool, ...]) -> None:
        """Atomically install/remove one MCP server while preserving other inventories."""
        if not isinstance(namespace, str) or not re.fullmatch(r"mcp:[A-Za-z0-9_.-]{1,128}/", namespace):
            raise ValueError("namespace must be mcp:SERVER/")
        if not isinstance(tools, tuple) or any(
            not isinstance(tool, Tool) or tool.origin != "mcp" or not tool.id.startswith(namespace) for tool in tools
        ):
            raise ValueError("every descriptor must belong to this MCP namespace")
        with self._lock:
            retained = tuple(tool for tool in self.tools if not tool.id.startswith(namespace))
            self.replace_tools(retained + tools)

    @contextmanager
    def _run(self, scope: Scope):
        with self._lock:
            if scope in self._running:
                raise RuntimeError("an orchestration loop is already active for this scope")
            self._running.add(scope)
        try:
            yield
        finally:
            with self._lock:
                self._running.discard(scope)

    def _audit(self, scope: Scope, kind: str, payload: dict[str, Any]) -> None:
        try:
            self.store.event(scope, kind, payload)
        except (sqlite3.Error, RuntimeError, ValueError) as error:
            # A telemetry failure must not misreport or retry an already completed effect.
            self.audit_errors.append({"kind": kind, "error": type(error).__name__})

    def _check_plan(self, scope: Scope, plan_id: str) -> _Issued:
        issued = self._issued.get(scope)
        if (
            issued is None
            or issued.plan.id != plan_id
            or issued.revision != self.store.revision(scope)
            or issued.tool_revision != self._tool_revision
            or issued.plan.catalog_revision != self.catalog.revision
            or issued.memory_signature != _memory_signature(self.store.memories(scope))
        ):
            raise ValueError("plan expired; prepare a new turn")
        return issued

    def observe_user(self, scope: Scope, text: str) -> list[str]:
        """Remember direct preference sentences only; callers must supply a human message."""
        if not self.policy.auto_memory:
            return []
        validate_text(text)
        saved = []
        # Deliberately narrow. Quoted text, documents, tool outputs and model guesses are not inputs.
        direct_lines = []
        fence = None
        for line in text.splitlines():
            marker = re.match(r"^\s*(`{3,}|~{3,})", line)
            if marker:
                character = marker.group(1)[0]
                fence = None if fence == character else character if fence is None else fence
                continue
            if fence is None and not line.lstrip().startswith(">"):
                direct_lines.append(line)
        for sentence in re.split(r"[\n]+|(?<=[.!?])\s+", "\n".join(direct_lines)):
            sentence = sentence.strip()
            if not re.match(r"(?i)^(?:please\s+)?(?:remember(?:\s+that)?\b|i\s+prefer\b|my\s+preference\s+is\b)", sentence):
                continue
            if not 8 <= len(sentence) <= 2000:
                continue
            # A stable normalized key deduplicates repeated direct statements, not inferred meanings.
            key = "auto:" + hashlib.sha256(sentence.casefold().encode()).hexdigest()[:24]
            saved.append(self.store.remember(scope, key, sentence, source="user", level="project"))
        return saved

    def prepare(
        self,
        scope: Scope,
        goal: str,
        *,
        stage: str = "handoff",
        grants: frozenset[str] = frozenset(),
        base_tokens: int | None = None,
        models: tuple[Model, ...] | None = None,
        tools: tuple[Tool, ...] | None = None,
        output_reserve: int | None = None,
    ) -> Plan:
        with self._lock:
            pending = self._pending.pop((scope.user, scope.project), None)
            if pending:
                identifier, expected = pending
                try:
                    self.store.apply(scope, identifier, expected_revision=expected)
                except ValueError:
                    # A concurrent edit leaves a draft for review instead of retrying a stale patch.
                    self._audit(scope, "refinement_deferred", {"proposal_id": identifier})
            revision, overlays = self.store.overlays_with_revision(scope)
            memories = self.store.memories(scope)
            bundle = plan(
                scope,
                goal,
                stage=stage,
                memories=memories,
                overlays=overlays,
                catalog=self.catalog,
                tools=self.tools if tools is None else tools,
                models=self.models if models is None else models,
                grants=grants,
                policy=self.policy if output_reserve is None else replace(self.policy, output_reserve=output_reserve),
                base_tokens=_goal_bound(goal) if base_tokens is None else base_tokens,
            )
            self._issued[scope] = _Issued(bundle.snapshot(), revision, self._tool_revision, _memory_signature(memories))
            self._audit(
                scope,
                "context_plan",
                {
                    "plan_id": bundle.id,
                    "stage": stage,
                    "model_id": bundle.model_id,
                    "memory_ids": list(bundle.memory_ids),
                    "skill_ids": list(bundle.skill_ids),
                    "tool_ids": [tool.id for tool in bundle.tools],
                    "tokens": bundle.token_upper_bound,
                },
            )
            return bundle

    def _selected_tool(self, scope: Scope, plan_id: str, tool_id: str, grants: frozenset[str]) -> Tool:
        if not isinstance(grants, frozenset) or any(not isinstance(item, str) for item in grants):
            raise TypeError("grants must be a frozenset of permission names")
        issued = self._check_plan(scope, plan_id)
        tool = next((item for item in issued.plan.tools if item.id == tool_id), None)
        if tool is None or not tool.permissions <= grants:
            raise PermissionError("tool is not selected or lacks current host grants")
        if issued.calls >= self.policy.max_tool_calls:
            raise ValueError("tool call budget exhausted")
        return tool

    def _admit_call(self, scope: Scope, plan_id: str, tool_id: str, grants: frozenset[str], call_id: str) -> Tool:
        with self._lock:
            tool = self._selected_tool(scope, plan_id, tool_id, grants)
            self._issued[scope].calls += 1
            self._audit(scope, "tool_started", {"plan_id": plan_id, "tool_id": tool_id, "call_id": call_id})
            return tool

    def _finished(
        self,
        scope: Scope,
        plan_id: str,
        tool_id: str,
        call_id: str,
        *,
        result: Any = None,
        error: BaseException | None = None,
    ) -> None:
        self._audit(
            scope,
            "tool_finished",
            {
                "plan_id": plan_id,
                "tool_id": tool_id,
                "call_id": call_id,
                "ok": error is None and not (isinstance(result, dict) and result.get("isError") is True),
                "cancelled": isinstance(error, asyncio.CancelledError),
            },
        )

    def _invoke_steps(
        self,
        scope,
        plan_id,
        tool_id,
        arguments,
        *,
        grants,
        authorize,
        current_grants=None,
        call_id=None,
        asynchronous=False,
    ):
        arguments = _json_copy(arguments)
        if not isinstance(arguments, dict):
            raise ValueError("tool arguments must be an object of at most 64 KiB")
        with self._lock:
            tool = self._selected_tool(scope, plan_id, tool_id, grants)
            if tool.asynchronous and not asynchronous:
                raise TypeError("async tool requires ainvoke/adispatch")
        approved = yield partial(authorize, tool.snapshot(), deepcopy(arguments))
        if approved is not True:
            raise PermissionError("host declined this tool invocation")
        fresh = grants if current_grants is None else (yield current_grants)
        identifier = call_id or str(uuid4())
        tool = self._admit_call(scope, plan_id, tool_id, fresh, identifier)
        try:
            with invocation(scope, plan_id, identifier):
                result = yield partial(tool.invoke, arguments)
        except BaseException as error:
            self._finished(scope, plan_id, tool_id, identifier, error=error)
            raise
        self._finished(scope, plan_id, tool_id, identifier, result=result)
        return result

    def invoke(
        self,
        scope: Scope,
        plan_id: str,
        tool_id: str,
        arguments: dict[str, Any],
        *,
        grants: frozenset[str],
        authorize: Callable[[Tool, dict[str, Any]], bool],
        current_grants: Callable[[], frozenset[str]] | None = None,
        call_id: str | None = None,
    ) -> Any:
        """Authorize and execute one synchronous host tool."""
        return drive(
            self._invoke_steps(
                scope,
                plan_id,
                tool_id,
                arguments,
                grants=grants,
                authorize=authorize,
                current_grants=current_grants,
                call_id=call_id,
            )
        )

    async def ainvoke(
        self,
        scope: Scope,
        plan_id: str,
        tool_id: str,
        arguments: dict[str, Any],
        *,
        grants: frozenset[str],
        authorize: Callable[[Tool, dict[str, Any]], Any],
        current_grants: Callable[[], Any] | None = None,
        call_id: str | None = None,
    ) -> Any:
        """Await host approval/tools without starting a nested event loop."""
        return await adrive(
            self._invoke_steps(
                scope,
                plan_id,
                tool_id,
                arguments,
                grants=grants,
                authorize=authorize,
                current_grants=current_grants,
                call_id=call_id,
                asynchronous=True,
            )
        )

    def _dispatch_steps(
        self,
        scope,
        goal,
        call_model,
        *,
        stage,
        grants,
        authorize,
        current_grants,
        base_tokens,
        asynchronous,
    ):
        with self._run(scope):
            admitted = grants if current_grants is None else (yield current_grants)
            base = _goal_bound(goal) if base_tokens is None else base_tokens
            bundle = self.prepare(scope, goal, stage=stage, grants=admitted, base_tokens=base)
            if bundle.model_id is None:
                raise ValueError("no eligible host model; inspect the plan reasons")
            model = next(item for item in self.models if item.id == bundle.model_id)
            trajectory, call_ids = [], set()
            for _ in range(self.policy.max_tool_calls + 1):
                with self._lock:
                    self._check_plan(scope, bundle.id)
                extra = len(json.dumps(trajectory, ensure_ascii=True, allow_nan=False).encode())
                if base + bundle.token_upper_bound + extra + self.policy.output_reserve > model.context_tokens:
                    raise ValueError("context budget exhausted by tool results; summarize before another turn")
                fresh = grants if current_grants is None else (yield current_grants)
                if model.local and "model:local" not in fresh:
                    raise PermissionError("local model grant was revoked")
                request = Request(
                    bundle.snapshot(), goal, tuple(_json_copy(trajectory, limit=1048576)), self.policy.output_reserve
                )
                reply = yield partial(call_model, request)
                with self._lock:
                    issued = self._check_plan(scope, bundle.id)
                if (
                    not isinstance(reply, Reply)
                    or not isinstance(reply.text, str)
                    or not isinstance(reply.calls, tuple)
                    or len(reply.text.encode()) > 65536
                ):
                    raise TypeError("model callback must return Reply")
                if not reply.calls:
                    return reply
                if authorize is None:
                    raise PermissionError("tool execution needs a host authorizer")
                if any(
                    not isinstance(call, ToolCall)
                    or not isinstance(call.id, str)
                    or not 1 <= len(call.id) <= 128
                    or not isinstance(call.tool_id, str)
                    or not isinstance(call.arguments, dict)
                    or call.id in call_ids
                    for call in reply.calls
                ):
                    raise ValueError("tool calls need fresh identifiers")
                ids = [call.id for call in reply.calls]
                if len(set(ids)) != len(ids):
                    raise ValueError("duplicate tool call identifier")
                if issued.calls + len(reply.calls) > self.policy.max_tool_calls:
                    raise ValueError("tool batch exceeds remaining budget")
                selected = {tool.id for tool in issued.plan.tools}
                if any(call.tool_id not in selected for call in reply.calls):
                    raise PermissionError("tool batch names an unselected tool")
                assistant = _json_copy(
                    {
                        "role": "assistant",
                        "text": reply.text,
                        "calls": [{"id": call.id, "tool_id": call.tool_id, "arguments": call.arguments} for call in reply.calls],
                    }
                )
                trajectory.append(assistant)
                for call in assistant["calls"]:
                    call_ids.add(call["id"])
                    fresh = grants if current_grants is None else (yield current_grants)
                    result = yield from self._invoke_steps(
                        scope,
                        bundle.id,
                        call["tool_id"],
                        call["arguments"],
                        grants=fresh,
                        authorize=authorize,
                        current_grants=current_grants,
                        call_id=call["id"],
                        asynchronous=asynchronous,
                    )
                    trajectory.append(
                        {
                            "role": "tool",
                            "call_id": call["id"],
                            "tool_id": call["tool_id"],
                            "result": _json_copy(result),
                        }
                    )
            raise ValueError("model/tool turn budget exhausted")

    def dispatch(
        self,
        scope: Scope,
        goal: str,
        call_model: Callable[[Request], Reply],
        *,
        stage: str = "handoff",
        grants: frozenset[str] = frozenset(),
        authorize: Callable[[Tool, dict[str, Any]], bool] | None = None,
        current_grants: Callable[[], frozenset[str]] | None = None,
        base_tokens: int | None = None,
    ) -> Reply:
        """Run the bounded model/tool workflow with synchronous callbacks."""
        return drive(
            self._dispatch_steps(
                scope,
                goal,
                call_model,
                stage=stage,
                grants=grants,
                authorize=authorize,
                current_grants=current_grants,
                base_tokens=base_tokens,
                asynchronous=False,
            )
        )

    async def adispatch(
        self,
        scope: Scope,
        goal: str,
        call_model: Callable[[Request], Any],
        *,
        stage: str = "handoff",
        grants: frozenset[str] = frozenset(),
        authorize: Callable[[Tool, dict[str, Any]], Any] | None = None,
        current_grants: Callable[[], Any] | None = None,
        base_tokens: int | None = None,
    ) -> Reply:
        """Run the same workflow with synchronous or asynchronous callbacks."""
        return await adrive(
            self._dispatch_steps(
                scope,
                goal,
                call_model,
                stage=stage,
                grants=grants,
                authorize=authorize,
                current_grants=current_grants,
                base_tokens=base_tokens,
                asynchronous=True,
            )
        )

    def record_outcome(self, scope: Scope, run_id: str, success: bool, summary: str) -> str:
        """Host-verified outcomes, not model assertions, supply refinement evidence."""
        outcome_id = self.store.record_outcome(scope, run_id, success, summary)
        matches = [row for row in self.store.outcomes(scope) if row["success"] and row["summary"] == summary]
        if summary and success and len(matches) >= 3:
            evidence = [row["id"] for row in matches[:3]]
            text = "Repeated host-verified successful tactic:\n" + summary
            target = "prompt:lesson-" + hashlib.sha256(summary.encode()).hexdigest()[:16]
            existing = [row for row in self.store.proposals(scope) if row["target"] == target and row["text"] == text]
            # A deliberate rollback suppresses automatic reintroduction of the same tactic.
            if not any(row["status"] == "rolled_back" for row in existing):
                proposal_id = self.store.propose(scope, target, text, evidence)
                proposal = next(row for row in self.store.proposals(scope) if row["id"] == proposal_id)
                if self.policy.auto_refine and proposal["status"] == "draft":
                    # Apply at a later prepare boundary, never during a live model/tool trajectory.
                    self._pending[(scope.user, scope.project)] = (proposal_id, self.store.revision(scope))
        return outcome_id

    def command(self, scope: Scope, text: str) -> Any:
        """Slash commands are an extension API; registering them in a UI is host-owned."""
        command, _, argument = text.strip().partition(" ")
        argument = argument.strip()
        if command == "/remember":
            level = "project"
            first, _, rest = argument.partition(" ")
            if first in {"user", "project", "session"}:
                level, argument = first, rest
            key, separator, value = argument.partition("=")
            if not separator:
                raise ValueError("use /remember [user|project|session] key = text")
            return self.store.remember(scope, key.strip(), value.strip(), source="user", level=level)
        if command == "/memory":
            return self.store.memories(scope)
        if command == "/events":
            return self.store.events(scope)
        if command == "/forget":
            return self.store.forget(scope, argument)
        if command == "/refine":
            if not argument:
                return self.store.proposals(scope)
            action, _, identifier = argument.partition(" ")
            if action != "apply" or not identifier.strip():
                raise ValueError("use /refine or /refine apply ID")
            return self.store.apply(scope, identifier.strip(), expected_revision=self.store.revision(scope))
        if command == "/rollback":
            return self.store.rollback(scope, argument, expected_revision=self.store.revision(scope))
        if command == "/context":
            issued = self._issued.get(scope)
            if issued is None:
                return None
            bundle = issued.plan
            return {
                "id": bundle.id,
                "model": bundle.model_id,
                "memories": bundle.memory_ids,
                "skills": bundle.skill_ids,
                "tools": [tool.id for tool in bundle.tools],
                "tokens": bundle.token_upper_bound,
                "reasons": bundle.reasons,
            }
        raise ValueError("unknown personalization command")
